// libpager.c — pager de experts MoE em user space via userfaultfd.
//
// Uso: LD_PRELOAD=./libpager.so <llama.cpp ...>
// Intercepta o mmap() do arquivo .gguf e devolve uma região anônima registrada
// no userfaultfd. Cada page fault chega pra thread handler, que descobre qual
// (camada, expert) foi tocado, lê do disco e insere com UFFDIO_COPY.
//
// Variáveis de ambiente:
//   PAGER_MODE       trace | fifo | lru      (default: trace)
//     trace: a cada novo token despeja todos os experts -> cada acesso vira um
//            fault -> log = trace completo de acessos (lento, só pra gerar trace).
//            Respeita PAGER_BUDGET_MB dentro do passo (necessário no passo do prompt).
//     fifo : orçamento fixo de RAM pros experts, despejo por ordem de carga.
//            (o pager só vê MISSES; sem ver hits, LRU degenera em FIFO)
//     lru  : orçamento fixo com despejo LRU real. O "bit de acesso" vem do router do modelo: o pager
//            intercepta llama_init_from_model e instala um cb_eval que observa ffn_moe_topk-<L>; a cada
//            camada marca os experts escolhidos como usados e despacha a carga dos que faltam (v4)
//   PAGER_ROUTER     1 = instala o cb_eval do router (default: 1 no modo lru, 0 nos outros)
//   PAGER_PREFETCH   1 = prefetch Markov no callback (v5; default: 1 com lru+router+async; 0 = v4)
//   PAGER_PREFETCH_K experts previstos por camada (default: 8)
//   PAGER_PREFETCH_MIN limiar de confiança (v6): só emite se o voto do expert >= essa fração dos votos da
//                    camada (default: 0.04, escolhido no simulador; 0 = K fixo, como no v5)
//   PAGER_EVICT_ASYNC 1 = madvise dos despejos num worker (v6; default: 1 no async). A contabilidade é imediata;
//                    um contador de geração por unidade garante que um madvise atrasado nunca apague um expert
//                    recarregado: antes de carregar, o pager cancela o despejo pendente (as páginas ainda estão
//                    lá: volta a ser residente sem ler o disco) ou espera o madvise em andamento terminar.
//   PAGER_EVICT_DELAY_US só para teste: o worker dorme entre as fatias do madvise (estado EV_MADVISING com parte
//                    das páginas já apagada: a janela em que uma carga sobreposta seria um erro)
// Espera (v5, todos os modos): uma thread leitora carimba a chegada de cada fault; o fault termina quando a
// thread é acordada (UFFDIO_WAKE no sync/parallel, cópia do bloco no async). espera= no resumo é o tempo de
// relógio com pelo menos uma thread do llama parada (as threads se sincronizam a cada operação).
//   PAGER_BUDGET_MB  orçamento dos experts no modo fifo (default: 3072)
//   PAGER_LOG        arquivo CSV de log (default: pager_trace.csv)
//   PAGER_DIRECT     1 = lê com O_DIRECT, sem page cache (default: 1)
//   PAGER_MIN_MB     só intercepta .gguf maiores que isso (default: 64)
//   PAGER_IO         sync | parallel     (default: sync)
//     sync    : v1 — o handler lê e copia cada fatia em sequência
//     parallel: v2 — um pool de threads lê as fatias (gate/up/down, e blocos de 8 MB dos tensores
//               fixos) em paralelo; cada worker faz pread + UFFDIO_COPY, sobrepondo leitura e cópia
//     async   : v3 — o handler despacha cada fault de expert pro pool e volta a ler o uffd na hora:
//               faults de threads diferentes carregam experts diferentes ao mesmo tempo. Fatias em
//               blocos de PAGER_BLOCK_KB; o UFFDIO_COPY de cada bloco acorda quem espera naquela página.
//               Um segundo fault num expert já em carga não recarrega: espera o bloco dele chegar.
//   PAGER_IO_THREADS workers dos modos parallel/async (default: 4)
//   PAGER_BLOCK_KB   tamanho do bloco de leitura no modo async (default: 256)
//   PAGER_VERBOSE    1 = imprime detalhes no stderr

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "llama.h"     // llama_context_params e ggml_tensor: compilar contra o mesmo commit do llama.cpp

#ifndef UFFD_USER_MODE_ONLY
#define UFFD_USER_MODE_ONLY 1
#endif
#ifndef UFFD_FEATURE_EXACT_ADDRESS
#define UFFD_FEATURE_EXACT_ADDRESS (1 << 11)
#endif
#ifndef USERFAULTFD_IOC_NEW
#define USERFAULTFD_IOC_NEW _IO(0xAA, 0x00)
#endif

#define PG 4096ULL
#define CHUNK (8ULL << 20)
#define MAX_REGIONS 16
#define MAX_SLICES 4

static inline uint64_t pg_down(uint64_t x) { return x & ~(PG - 1); }
static inline uint64_t pg_up(uint64_t x) { return (x + PG - 1) & ~(PG - 1); }
static uint64_t now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000ULL + t.tv_nsec / 1000;
}

// ---------------------------------------------------------------- estruturas
typedef struct { uint64_t off, len; } slice_t;          // offsets no arquivo

typedef struct {
    int region, layer, expert;   // expert = -1 -> tensor fixo (atenção, norm, embedding...)
    int nslices; slice_t sl[MAX_SLICES];
    uint64_t bytes;              // bytes de páginas internas (contabilizadas no orçamento)
    int resident;
    int loading;                 // async: carga em voo (o FIFO não despeja)
    int pending;                 // async: blocos que faltam
    uint64_t t_submit, step_submit, load_bytes;
    int pf;                      // carregado por prefetch e ainda não usado
    uint64_t mark_step;          // step+1 do último uso marcado (0 = nunca); o prefetch não despeja os do token atual
    uint32_t gen;                // v6: muda a cada despejo e a cada resgate (ev_mu)
    int evstate;                 // v6: 0 nada, 1 despejo pendente, 2 madvise em andamento (ev_mu)   // load_bytes: intervalo lido (com bordas), igual ao log do v1/v2
    int prev, next;              // fila FIFO
} unit_t;

typedef struct { uint64_t start, end; int unit; } range_t; // [start,end) no arquivo

typedef struct {
    char *base; uint64_t len, fsize;
    int fd;                       // fd próprio (O_DIRECT se possível)
    int direct;
    range_t *ranges; int nranges;
} region_t;

static region_t R[MAX_REGIONS];
static int nregions;
static unit_t *U; static int nunits, capunits;

static int uffd = -1;
static pthread_t handler_tid;
static pthread_mutex_t setup_mu = PTHREAD_MUTEX_INITIALIZER;
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);

// config
static int mode_fifo;                  // 1 = orçamento fixo (fifo ou lru); 0 = trace
static int mode_lru, router_on;
static uint64_t budget;
static int use_direct = 1, verbose;
static uint64_t min_bytes;
static FILE *logf_;

// estado
static uint64_t used, pinned_bytes, step, nloads, nevict, bytes_read, io_us, nunknown;
static uint64_t load_us;              // tempo de relógio do handler carregando (leitura + cópia)
static int io_parallel, io_async, nworkers = 4;
static uint64_t block_bytes = 256 << 10;
static uint64_t nwait;                // async: faults num expert que já estava em carga

// v5: espera medida = da chegada do fault (thread leitora) até a thread ser acordada
typedef struct { uint64_t addr, t; int ui; } waiter_t;
#define WN 8192
static waiter_t W[WN]; static int nw;
static pthread_mutex_t w_mu = PTHREAD_MUTEX_INITIALIZER;
static int n_waiting; static uint64_t wait_t0;
static uint64_t wait_union_us, wait_sum_us, nwaits, wait_union_gen_us, wait_sum_gen_us, nwaits_gen, nw_overflow;

static void wait_close_locked(uint64_t t, uint64_t now) {     // com w_mu
    uint64_t d = now > t ? now - t : 0;
    wait_sum_us += d; nwaits++;
    if (step >= 1) { wait_sum_gen_us += d; nwaits_gen++; }
    if (--n_waiting == 0) {
        uint64_t u = now > wait_t0 ? now - wait_t0 : 0;
        wait_union_us += u;
        if (step >= 1) wait_union_gen_us += u;
    }
}
static void wait_open(uint64_t t) {
    pthread_mutex_lock(&w_mu);
    if (n_waiting++ == 0) wait_t0 = t;
    pthread_mutex_unlock(&w_mu);
}
static void wait_close(uint64_t t) {
    pthread_mutex_lock(&w_mu);
    wait_close_locked(t, now_us());
    pthread_mutex_unlock(&w_mu);
}
// async: o fault termina quando o bloco da página for copiado. Se já foi (mincore), termina agora.
// O worker copia ANTES de pegar w_mu e procurar esperadores, então a checagem e o registro sob w_mu não perdem nenhum.
static void wait_register(uint64_t addr, uint64_t t, int ui) {
    unsigned char v = 0;
    pthread_mutex_lock(&w_mu);
    if (mincore((void *)pg_down(addr), PG, &v) == 0 && (v & 1)) wait_close_locked(t, now_us());
    else if (nw < WN) W[nw++] = (waiter_t){ pg_down(addr), t, ui };
    else { nw_overflow++; wait_close_locked(t, now_us()); }
    pthread_mutex_unlock(&w_mu);
}
static void wait_wake_range(int ui, uint64_t a, uint64_t b) {   // endereços [a,b) acabaram de ser copiados
    pthread_mutex_lock(&w_mu);
    uint64_t now = now_us();
    for (int i = 0; i < nw; ) {
        if (W[i].ui == ui && W[i].addr >= a && W[i].addr < b) { wait_close_locked(W[i].t, now); W[i] = W[--nw]; }
        else i++;
    }
    pthread_mutex_unlock(&w_mu);
}

// v5: prefetch Markov
static int prefetch_on, prefetch_k = 8;
static uint64_t npf_issued, npf_useful, npf_wasted, npf_gen_issued, npf_gen_useful, npf_gen_wasted;
static uint64_t npf_skipped_budget;       // previsões abandonadas por não caber sem despejar o token atual
// Só instrumentação: prefetch por metade da geração (passos 1-32 e 33+) e madvise dos despejos do prefetch.
static uint64_t pf_half_issued[2], pf_half_useful[2], pf_half_wasted[2];
static int in_prefetch;                   // 1 enquanto markov_prefetch roda (st_mu)
static uint64_t pf_madvise_ns, pf_evict_n;
static inline int half_of_step(void) { return step >= 33; }
static double prefetch_min = 0.04;        // v6: limiar de confiança do prefetch (0 = K fixo)
static uint64_t npf_below_min;            // previsões cortadas pelo limiar

// v6: despejo adiado
enum { EV_NONE = 0, EV_PENDING = 1, EV_MADVISING = 2 };
typedef struct { int ui; uint32_t gen; } evjob_t;
#define EVQ 8192
static evjob_t evq[EVQ]; static int evq_head, evq_count, evq_max;
static pthread_mutex_t ev_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ev_cv = PTHREAD_COND_INITIALIZER, ev_idle = PTHREAD_COND_INITIALIZER;
static int evict_async;
static uint64_t nev_deferred, nev_rescued, nev_waited, nev_stale, nev_sync_full, ev_madvise_ns, ev_madvise_gen_ns;
static uint64_t nfault_resident;          // fault numa página de unidade já residente (várias threads no mesmo expert)
static uint64_t nev_violation;            // invariante: carga iniciada durante o madvise da unidade (tem de ser 0)
static int last_layer = -1;
static pthread_mutex_t st_mu = PTHREAD_MUTEX_INITIALIZER;   // fila, orçamento e passo (handler x callback)
static int *emap_g, emap_layers; static uint64_t emap_maxe;  // (camada, expert) -> unidade
static int cb_seen, cb_last_layer = -1;
static uint64_t ncb, cb_ns, ncb_loads, ncb_touch;
static uint64_t cb_ask_ns, cb_lock_ns, cb_work_ns;      // decomposição do custo (só passos de geração)
static uint64_t cb_gen_ns;                               // custo total nos passos de geração (passo >= 1)
static uint64_t cb_evict_ns, cb_submit_ns, cb_nevict;    // dentro do trabalho: despejos e despacho
static uint64_t cb_pf_ns;                                // dentro do trabalho: prefetch Markov
static uint64_t ns_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec; }
static ggml_backend_sched_eval_callback prev_cb; static void *prev_ud;
static int q_head = -1, q_tail = -1;   // FIFO de experts residentes
static char *iobuf;

#define LOGV(...) do { if (verbose) fprintf(stderr, "[pager] " __VA_ARGS__); } while (0)

// ---------------------------------------------------------------- leitor GGUF
typedef struct { int fd; uint64_t pos, bstart, blen; char buf[1 << 20]; } rd_t;

static int rd(rd_t *r, void *dst, uint64_t n) {
    char *d = dst;
    while (n) {
        if (r->pos < r->bstart || r->pos >= r->bstart + r->blen) {
            ssize_t k = pread(r->fd, r->buf, sizeof r->buf, r->pos);
            if (k <= 0) return -1;
            r->bstart = r->pos; r->blen = k;
        }
        uint64_t o = r->pos - r->bstart, take = r->blen - o;
        if (take > n) take = n;
        memcpy(d, r->buf + o, take);
        d += take; r->pos += take; n -= take;
    }
    return 0;
}
static int rd_u32(rd_t *r, uint32_t *v) { return rd(r, v, 4); }
static int rd_u64(rd_t *r, uint64_t *v) { return rd(r, v, 8); }

static int rd_str(rd_t *r, char *out, size_t cap) {
    uint64_t n; if (rd_u64(r, &n)) return -1;
    if (out && n < cap) { if (rd(r, out, n)) return -1; out[n] = 0; }
    else { if (out && cap) out[0] = 0; r->pos += n; }
    return 0;
}

static int type_size(uint32_t t) {
    switch (t) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return -1;
    }
}

static int skip_value(rd_t *r, uint32_t t) {
    if (t == 8) return rd_str(r, NULL, 0);
    if (t == 9) {
        uint32_t et; uint64_t n;
        if (rd_u32(r, &et) || rd_u64(r, &n)) return -1;
        if (et == 8 || et == 9) { for (uint64_t i = 0; i < n; i++) if (skip_value(r, et)) return -1; return 0; }
        int s = type_size(et); if (s < 0) return -1;
        r->pos += n * (uint64_t)s; return 0;
    }
    int s = type_size(t); if (s < 0) return -1;
    r->pos += s; return 0;
}

typedef struct { char name[96]; uint32_t ndims; uint64_t ne[4]; uint64_t off; } tinfo_t;

static int cmp_tinfo(const void *a, const void *b) {
    uint64_t x = ((const tinfo_t *)a)->off, y = ((const tinfo_t *)b)->off;
    return x < y ? -1 : x > y;
}
static int cmp_range(const void *a, const void *b) {
    uint64_t x = ((const range_t *)a)->start, y = ((const range_t *)b)->start;
    return x < y ? -1 : x > y;
}

static int new_unit(int region, int layer, int expert) {
    if (nunits == capunits) {
        capunits = capunits ? capunits * 2 : 4096;
        U = realloc(U, sizeof(unit_t) * capunits);
    }
    unit_t *u = &U[nunits];
    memset(u, 0, sizeof *u);
    u->region = region; u->layer = layer; u->expert = expert; u->prev = u->next = -1;
    return nunits++;
}

static void add_slice(int ui, uint64_t off, uint64_t len) {
    unit_t *u = &U[ui];
    if (u->nslices < MAX_SLICES) { u->sl[u->nslices].off = off; u->sl[u->nslices].len = len; u->nslices++; }
    uint64_t a = pg_up(off), b = pg_down(off + len);
    if (b > a) u->bytes += b - a;
}

// Monta as unidades: um expert lógico = (camada, expert) agrupando gate/up/down.
static int parse_gguf(int ri, int fd) {
    region_t *g = &R[ri];
    rd_t *r = calloc(1, sizeof *r); r->fd = fd;
    uint32_t magic, ver; uint64_t nt, nkv; uint32_t align = 32;
    if (rd_u32(r, &magic) || magic != 0x46554747 /* "GGUF" */) goto fail;
    if (rd_u32(r, &ver) || rd_u64(r, &nt) || rd_u64(r, &nkv)) goto fail;
    for (uint64_t i = 0; i < nkv; i++) {
        char key[256]; uint32_t t;
        if (rd_str(r, key, sizeof key) || rd_u32(r, &t)) goto fail;
        if (!strcmp(key, "general.alignment") && t == 4) { if (rd_u32(r, &align)) goto fail; }
        else if (skip_value(r, t)) goto fail;
    }
    tinfo_t *T = calloc(nt, sizeof *T);
    for (uint64_t i = 0; i < nt; i++) {
        uint32_t ty;
        if (rd_str(r, T[i].name, sizeof T[i].name) || rd_u32(r, &T[i].ndims)) { free(T); goto fail; }
        if (T[i].ndims > 4) { free(T); goto fail; }
        for (uint32_t d = 0; d < T[i].ndims; d++) if (rd_u64(r, &T[i].ne[d])) { free(T); goto fail; }
        if (rd_u32(r, &ty) || rd_u64(r, &T[i].off)) { free(T); goto fail; }
    }
    uint64_t data = (r->pos + align - 1) / align * align;
    qsort(T, nt, sizeof *T, cmp_tinfo);

    int maxl = -1; uint64_t maxe = 0;
    for (uint64_t i = 0; i < nt; i++) {
        int l = -1; sscanf(T[i].name, "blk.%d.", &l);
        if (l > maxl) maxl = l;
        if (T[i].ndims == 3 && strstr(T[i].name, "_exps") && T[i].ne[2] > maxe) maxe = T[i].ne[2];
    }
    int *emap = NULL;
    if (maxl >= 0 && maxe) { emap = malloc(sizeof(int) * (maxl + 1) * maxe); memset(emap, -1, sizeof(int) * (maxl + 1) * maxe); }

    g->ranges = malloc(sizeof(range_t) * (nt + (maxl + 1) * maxe * 3 + 16));
    g->nranges = 0;
    int nexp_t = 0;
    for (uint64_t i = 0; i < nt; i++) {
        uint64_t start = data + T[i].off;
        uint64_t end = (i + 1 < nt) ? data + T[i + 1].off : g->fsize;
        uint64_t size = end - start;
        int l = -1; sscanf(T[i].name, "blk.%d.", &l);
        if (emap && l >= 0 && T[i].ndims == 3 && strstr(T[i].name, "_exps") && T[i].ne[2] > align) {
            uint64_t ne2 = T[i].ne[2];
            uint64_t sl = size / ne2;   // padding < alignment < ne2 -> divisão exata da fatia
            nexp_t++;
            for (uint64_t e = 0; e < ne2; e++) {
                int *slot = &emap[(uint64_t)l * maxe + e];
                if (*slot < 0) *slot = new_unit(ri, l, (int)e);
                add_slice(*slot, start + e * sl, sl);
                g->ranges[g->nranges++] = (range_t){ start + e * sl, start + (e + 1) * sl, *slot };
            }
        } else {
            int ui = new_unit(ri, l, -1);
            add_slice(ui, start, size);
            g->ranges[g->nranges++] = (range_t){ start, end, ui };
        }
    }
    qsort(g->ranges, g->nranges, sizeof(range_t), cmp_range);
    fprintf(stderr, "[pager] %llu tensores (%d de experts), %d camadas, %llu experts/camada, alinhamento %u, dados em %llu\n",
            (unsigned long long)nt, nexp_t, maxl + 1, (unsigned long long)maxe, align, (unsigned long long)data);
    if (emap && !emap_g) { emap_g = emap; emap_layers = maxl + 1; emap_maxe = maxe; emap = NULL; }
    free(T); free(emap); free(r);
    return 0;
fail:
    fprintf(stderr, "[pager] falha ao ler o header GGUF\n");
    free(r);
    return -1;
}

// ---------------------------------------------------------------- I/O + inserção
static void read_file(region_t *g, uint64_t off, uint64_t n, char *dst) {
    uint64_t t0 = now_us(), done = 0;
    while (done < n) {
        ssize_t k = pread(g->fd, dst + done, n - done, off + done);
        if (k < 0 && errno == EINTR) continue;
        if (k < 0 && g->direct && errno == EINVAL) {      // fs não aceita O_DIRECT: cai pro normal
            int fl = fcntl(g->fd, F_GETFL); fcntl(g->fd, F_SETFL, fl & ~O_DIRECT); g->direct = 0;
            continue;
        }
        if (k <= 0) break;                                // EOF
        done += k;
    }
    if (done < n) memset(dst + done, 0, n - done);
    if (!g->direct && done) posix_fadvise(g->fd, off, done, POSIX_FADV_DONTNEED);
    __atomic_add_fetch(&bytes_read, done, __ATOMIC_RELAXED);
    __atomic_add_fetch(&io_us, now_us() - t0, __ATOMIC_RELAXED);   // soma por thread (no parallel se sobrepõe)
}

// Copia páginas pra região; ignora as que já existem (EEXIST).
static void uffd_copy(char *dst, char *src, uint64_t len) {
    uint64_t pos = 0;
    while (pos < len) {
        struct uffdio_copy c = { .dst = (uint64_t)(dst + pos), .src = (uint64_t)(src + pos), .len = len - pos, .mode = 0 };
        if (ioctl(uffd, UFFDIO_COPY, &c) == 0) return;
        if (errno == EAGAIN && c.copy > 0) { pos += c.copy; continue; }
        if (errno == EEXIST || errno == ENOENT) { if (c.copy > 0) pos += c.copy; pos += PG; continue; }
        if (errno == EAGAIN) continue;
        fprintf(stderr, "[pager] UFFDIO_COPY: %s\n", strerror(errno));
        return;
    }
}

static void load_file_range(region_t *g, uint64_t a, uint64_t b) {   // a,b alinhados a página
    if (b > g->len) b = pg_up(g->len);
    for (uint64_t p = a; p < b; p += CHUNK) {
        uint64_t n = b - p < CHUNK ? b - p : CHUNK;
        read_file(g, p, n, iobuf);
        uffd_copy(g->base + p, iobuf, n);
    }
}

// ---------------------------------------------------------------- pool de I/O (PAGER_IO=parallel)
#define QN 16384
typedef struct { region_t *g; uint64_t a, b; int ui; } job_t;   // ui >= 0: bloco de uma carga async
static job_t jobq[QN];
static int jq_head, jq_count, jq_pending;
static pthread_mutex_t jq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t jq_cv = PTHREAD_COND_INITIALIZER, jq_space = PTHREAD_COND_INITIALIZER,
                      jq_done = PTHREAD_COND_INITIALIZER;
#define PFN 4096
static job_t pfq[PFN]; static int pf_count;   // fila de prefetch: os workers só pegam se a de demanda estiver vazia
static int inflight;                  // async: unidades em carga
static uint64_t busy_t0;              // async: início do período com >= 1 carga em voo

static void log_load_at(unit_t *u, uint64_t st, uint64_t bytes, uint64_t us) {
    if (logf_) fprintf(logf_, "%llu,%d,%d,%llu,%llu\n", (unsigned long long)st, u->layer, u->expert,
                       (unsigned long long)bytes, (unsigned long long)us);
}

static void unit_done(int ui) {       // chamado pelo worker que copiou o último bloco
    unit_t *u = &U[ui];
    log_load_at(u, u->step_submit, u->load_bytes, now_us() - u->t_submit);
    __atomic_store_n(&u->resident, 1, __ATOMIC_RELEASE);
    __atomic_store_n(&u->loading, 0, __ATOMIC_RELEASE);
    pthread_mutex_lock(&jq_mu);
    if (--inflight == 0) load_us += now_us() - busy_t0;
    pthread_mutex_unlock(&jq_mu);
}

static void *io_worker(void *arg) {
    (void)arg;
    sigset_t all; sigfillset(&all); pthread_sigmask(SIG_BLOCK, &all, NULL);
    char *buf;
    if (posix_memalign((void **)&buf, PG, CHUNK)) return NULL;   // block_bytes <= CHUNK
    for (;;) {
        pthread_mutex_lock(&jq_mu);
        while (!jq_count && !pf_count) pthread_cond_wait(&jq_cv, &jq_mu);
        job_t j;
        if (jq_count) {
            j = jobq[jq_head]; jq_head = (jq_head + 1) % QN; jq_count--;
            pthread_cond_signal(&jq_space);
        } else {
            j = pfq[0]; memmove(pfq, pfq + 1, sizeof(job_t) * --pf_count);
        }
        pthread_mutex_unlock(&jq_mu);
        read_file(j.g, j.a, j.b - j.a, buf);
        uffd_copy(j.g->base + j.a, buf, j.b - j.a);       // mode=0: acorda quem espera nessas páginas
        if (j.ui >= 0) {
            wait_wake_range(j.ui, (uint64_t)j.g->base + j.a, (uint64_t)j.g->base + j.b);
            if (__atomic_sub_fetch(&U[j.ui].pending, 1, __ATOMIC_ACQ_REL) == 0) unit_done(j.ui);
            continue;
        }
        pthread_mutex_lock(&jq_mu);
        if (--jq_pending == 0) pthread_cond_signal(&jq_done);
        pthread_mutex_unlock(&jq_mu);
    }
    return NULL;
}

static void submit(region_t *g, uint64_t a, uint64_t b) {     // a,b alinhados a página; quebra em CHUNK
    if (b > g->len) b = pg_up(g->len);
    for (uint64_t p = a; p < b; p += CHUNK) {
        pthread_mutex_lock(&jq_mu);
        while (jq_count == QN) pthread_cond_wait(&jq_space, &jq_mu);
        jobq[(jq_head + jq_count) % QN] = (job_t){ g, p, b - p < CHUNK ? b : p + CHUNK, -1 };
        jq_count++; jq_pending++;
        pthread_cond_signal(&jq_cv);
        pthread_mutex_unlock(&jq_mu);
    }
}

static void wait_jobs(void) {
    pthread_mutex_lock(&jq_mu);
    while (jq_pending) pthread_cond_wait(&jq_done, &jq_mu);
    pthread_mutex_unlock(&jq_mu);
}

static void enqueue(job_t j) {
    pthread_mutex_lock(&jq_mu);
    while (jq_count == QN) pthread_cond_wait(&jq_space, &jq_mu);
    jobq[(jq_head + jq_count) % QN] = j;
    jq_count++;
    pthread_cond_signal(&jq_cv);
    pthread_mutex_unlock(&jq_mu);
}

// Sobe pra fila de demanda os blocos de prefetch ainda não lidos de uma unidade que passou a ser pedida.
static void promote(int ui) {
    pthread_mutex_lock(&jq_mu);
    int k = 0;
    for (int i = 0; i < pf_count; i++) {
        if (pfq[i].ui == ui && jq_count < QN) { jobq[(jq_head + jq_count) % QN] = pfq[i]; jq_count++; }
        else pfq[k++] = pfq[i];
    }
    pf_count = k;
    pthread_cond_broadcast(&jq_cv);
    pthread_mutex_unlock(&jq_mu);
}

// async: despacha a unidade inteira em blocos e retorna sem esperar. prefetch=1: fila de baixa prioridade,
// tudo ou nada (devolve 0 se não couber, sem mexer na unidade).
static int submit_unit_prio(region_t *g, int ui, int prefetch);
static void submit_unit(region_t *g, int ui) { submit_unit_prio(g, ui, 0); }
static int submit_unit_prio(region_t *g, int ui, int prefetch) {
    unit_t *u = &U[ui];
    int n = 0;
    u->load_bytes = 0;
    for (int s = 0; s < u->nslices; s++) {
        uint64_t a = pg_down(u->sl[s].off), b = pg_up(u->sl[s].off + u->sl[s].len);
        if (b > g->len) b = pg_up(g->len);
        if (b > a) { n += (int)((b - a + block_bytes - 1) / block_bytes); u->load_bytes += b - a; }
    }
    if (!n) { __atomic_store_n(&u->resident, 1, __ATOMIC_RELEASE); return 1; }
    if (prefetch) {
        pthread_mutex_lock(&jq_mu);
        int room = pf_count + n <= PFN;
        pthread_mutex_unlock(&jq_mu);
        if (!room) return 0;                     // só o handler/callback (sob st_mu) enfileiram prefetch
    }
    u->t_submit = now_us(); u->step_submit = step;
    __atomic_store_n(&u->pending, n, __ATOMIC_RELEASE);   // antes de qualquer bloco entrar na fila
    __atomic_store_n(&u->loading, 1, __ATOMIC_RELEASE);
    pthread_mutex_lock(&jq_mu);
    if (inflight++ == 0) busy_t0 = now_us();
    pthread_mutex_unlock(&jq_mu);
    for (int s = 0; s < u->nslices; s++) {
        uint64_t a = pg_down(u->sl[s].off), b = pg_up(u->sl[s].off + u->sl[s].len);
        if (b > g->len) b = pg_up(g->len);
        for (uint64_t p = a; p < b; p += block_bytes) {
            job_t j = { g, p, b - p < block_bytes ? b : p + block_bytes, ui };
            if (!prefetch) { enqueue(j); continue; }
            pthread_mutex_lock(&jq_mu);
            pfq[pf_count++] = j;
            pthread_cond_signal(&jq_cv);
            pthread_mutex_unlock(&jq_mu);
        }
    }
    return 1;
}

static void q_remove(int i) {
    unit_t *u = &U[i];
    if (u->prev >= 0) U[u->prev].next = u->next; else q_head = u->next;
    if (u->next >= 0) U[u->next].prev = u->prev; else q_tail = u->prev;
    u->prev = u->next = -1;
}
static void q_push(int i) {
    U[i].prev = q_tail; U[i].next = -1;
    if (q_tail >= 0) U[q_tail].next = i; else q_head = i;
    q_tail = i;
}

static int evict_delay_us;
static void madvise_unit(unit_t *u, int test_delay) {
    region_t *g = &R[u->region];
    for (int s = 0; s < u->nslices; s++) {
        uint64_t a = pg_up(u->sl[s].off), b = pg_down(u->sl[s].off + u->sl[s].len);
        if (b > a) madvise(g->base + a, b - a, MADV_DONTNEED);   // páginas de borda ficam
        // teste: dorme com parte das fatias já apagadas e o estado em EV_MADVISING (a janela perigosa)
        if (test_delay && evict_delay_us && s + 1 < u->nslices) usleep(evict_delay_us);
    }
}

// Worker de despejo: só age se a unidade continua com o despejo pendente da MESMA geração do pedido.
static void *evictor(void *arg) {
    (void)arg;
    sigset_t all; sigfillset(&all); pthread_sigmask(SIG_BLOCK, &all, NULL);
    for (;;) {
        pthread_mutex_lock(&ev_mu);
        while (!evq_count) pthread_cond_wait(&ev_cv, &ev_mu);
        evjob_t j = evq[evq_head]; evq_head = (evq_head + 1) % EVQ; evq_count--;
        unit_t *u = &U[j.ui];
        if (u->evstate != EV_PENDING || u->gen != j.gen) { nev_stale++; pthread_mutex_unlock(&ev_mu); continue; }
        u->evstate = EV_MADVISING;                 // a partir daqui, quem quiser carregar espera (reclaim)
        pthread_mutex_unlock(&ev_mu);
        uint64_t t0 = now_us();
        madvise_unit(u, 1);
        uint64_t dt = (now_us() - t0) * 1000;
        pthread_mutex_lock(&ev_mu);
        ev_madvise_ns += dt; if (step >= 1) ev_madvise_gen_ns += dt;
        // ninguém pode ter começado a carregar esta unidade enquanto o madvise rodava (o reclaim espera)
        if (__atomic_load_n(&u->loading, __ATOMIC_ACQUIRE) || __atomic_load_n(&u->resident, __ATOMIC_ACQUIRE))
            nev_violation++;
        u->evstate = EV_NONE;
        pthread_cond_broadcast(&ev_idle);
        pthread_mutex_unlock(&ev_mu);
    }
    return NULL;
}

// Antes de carregar uma unidade ausente (com st_mu): se o despejo ainda está pendente, cancela e a unidade
// volta a ser residente sem ler o disco (devolve 1); se o madvise está em andamento, espera ele acabar (0).
static void q_push(int i);
static int reclaim(int ui) {
    unit_t *u = &U[ui];
    if (!evict_async) return 0;
    pthread_mutex_lock(&ev_mu);
    while (u->evstate == EV_MADVISING) { nev_waited++; pthread_cond_wait(&ev_idle, &ev_mu); }
    if (u->evstate == EV_PENDING) {
        u->evstate = EV_NONE; u->gen++;           // invalida o pedido que está na fila
        pthread_mutex_unlock(&ev_mu);
        __atomic_store_n(&u->resident, 1, __ATOMIC_RELEASE);
        used += u->bytes; q_push(ui); nev_rescued++;
        return 1;
    }
    pthread_mutex_unlock(&ev_mu);
    return 0;
}

static void evict(int i) {
    unit_t *u = &U[i];
    int deferred = 0;
    // deixa de ser residente ANTES de o pedido entrar na fila: o worker pode fazer o madvise na hora
    __atomic_store_n(&u->resident, 0, __ATOMIC_RELEASE);
    if (evict_async) {
        pthread_mutex_lock(&ev_mu);
        if (evq_count < EVQ) {
            u->gen++; u->evstate = EV_PENDING;
            evq[(evq_head + evq_count) % EVQ] = (evjob_t){ i, u->gen };
            if (++evq_count > evq_max) evq_max = evq_count;
            pthread_cond_signal(&ev_cv);
            nev_deferred++; deferred = 1;
        } else nev_sync_full++;
        pthread_mutex_unlock(&ev_mu);
    }
    struct timespec m0, m1;
    if (in_prefetch) clock_gettime(CLOCK_MONOTONIC, &m0);
    if (!deferred) madvise_unit(u, 0);
    if (in_prefetch && step >= 1) {
        clock_gettime(CLOCK_MONOTONIC, &m1);
        pf_madvise_ns += (uint64_t)((m1.tv_sec - m0.tv_sec) * 1000000000LL + (m1.tv_nsec - m0.tv_nsec));
        pf_evict_n++;
    }
    if (u->pf) {
        npf_wasted++;
        if (step >= 1) { npf_gen_wasted++; pf_half_wasted[half_of_step()]++; }
        u->pf = 0;
    }
    used -= u->bytes; nevict++;
    q_remove(i);
}

// Despeja o mais antigo que não esteja em carga (no sync/parallel nunca há carga em voo: é o head).
static int evict_oldest(void) {
    for (int i = q_head; i >= 0; i = U[i].next)
        if (!__atomic_load_n(&U[i].loading, __ATOMIC_ACQUIRE)) { evict(i); return 1; }
    return 0;
}

// Prefetch: despeja o mais antigo que não esteja em carga nem tenha sido usado no token atual.
static int evict_oldest_unmarked(void) {
    for (int i = q_head; i >= 0; i = U[i].next)
        if (!__atomic_load_n(&U[i].loading, __ATOMIC_ACQUIRE) && U[i].mark_step != step + 1) { evict(i); return 1; }
    return 0;
}

// Uso de verdade de uma unidade (fault ou router): marca no token e, se veio de prefetch, conta como útil.
static void mark_used(int ui) {
    unit_t *u = &U[ui];
    u->mark_step = step + 1;
    if (u->pf) {
        npf_useful++;
        if (step >= 1) { npf_gen_useful++; pf_half_useful[half_of_step()]++; }
        u->pf = 0;
        if (__atomic_load_n(&u->loading, __ATOMIC_ACQUIRE)) promote(ui);
    }
}

static void log_load(unit_t *u, uint64_t bytes, uint64_t us) {
    if (logf_) fprintf(logf_, "%llu,%d,%d,%llu,%llu\n", (unsigned long long)step, u->layer, u->expert,
                       (unsigned long long)bytes, (unsigned long long)us);
}

static int find_unit(region_t *g, uint64_t off) {
    int lo = 0, hi = g->nranges - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        if (off < g->ranges[m].start) hi = m - 1;
        else if (off >= g->ranges[m].end) lo = m + 1;
        else return g->ranges[m].unit;
    }
    return -1;
}

static void handle_fault_locked(uint64_t addr, uint64_t t);
static void handle_fault(uint64_t addr, uint64_t t) {
    pthread_mutex_lock(&st_mu);
    handle_fault_locked(addr, t);
    pthread_mutex_unlock(&st_mu);
}

// Toda saída fecha a espera do fault exatamente uma vez: wait_close (acordou aqui) ou wait_register (o worker fecha).
static void handle_fault_locked(uint64_t addr, uint64_t t) {
    region_t *g = NULL;
    for (int i = 0; i < nregions; i++)
        if (addr >= (uint64_t)R[i].base && addr < (uint64_t)R[i].base + R[i].len) { g = &R[i]; break; }
    if (!g) { wait_close(t); return; }
    uint64_t off = addr - (uint64_t)g->base;
    int ui = find_unit(g, off);

    if (ui >= 0 && io_async && __atomic_load_n(&U[ui].loading, __ATOMIC_ACQUIRE)) {
        nwait++;                                 // já em carga: o UFFDIO_COPY do bloco acorda a thread
        if (U[ui].expert >= 0) mark_used(ui);    // se era prefetch, vira demanda (promove os blocos)
        wait_register(addr, t, ui);
        return;
    }
    if (ui < 0 || __atomic_load_n(&U[ui].resident, __ATOMIC_ACQUIRE)) {   // header/padding, ou página de borda
page:;
        uint64_t tp = now_us();
        load_file_range(g, pg_down(off), pg_down(off) + PG);
        load_us += now_us() - tp;
        if (ui < 0) nunknown++; else nfault_resident++;
        goto wake;
    }
    unit_t *u = &U[ui];
    if (u->expert >= 0) {
        mark_used(ui);
        if (!cb_seen && u->layer < last_layer) { // camada voltou -> novo token (sem router; com ele, o passo vem do callback)
            step++;
            if (!mode_fifo) while (evict_oldest()) ;
        }
        last_layer = u->layer;
        // Também no trace: o passo do prompt toca milhares de experts (~7,5 GB no Qwen3.6 com 97 tokens)
        // e estoura o MemoryMax se nada for despejado antes da troca de token. Nos passos de geração o
        // conjunto do token (~590 MB) cabe no orçamento, então o trace desses passos continua completo;
        // só o passo 0 pode ter recargas, e o simulate.py o descarta (--skip 1).
        while (used + u->bytes > budget && evict_oldest()) ;
    }
    if (reclaim(ui)) goto page;                  // despejo pendente cancelado: as páginas da unidade ainda estão lá
    if (io_async) {
        nloads++;
        if (u->expert >= 0) { used += u->bytes; q_push(ui); }
        else pinned_bytes += u->bytes;
        submit_unit(g, ui);
        wait_register(addr, t, ui);
        return;                                  // sem UFFDIO_WAKE: a cópia do bloco acorda
    }
    uint64_t t0 = now_us(), total = 0;
    for (int s = 0; s < u->nslices; s++) {
        uint64_t a = pg_down(u->sl[s].off), b = pg_up(u->sl[s].off + u->sl[s].len);
        if (io_parallel) submit(g, a, b); else load_file_range(g, a, b);
        total += b - a;
    }
    if (io_parallel) wait_jobs();
    load_us += now_us() - t0;
    u->resident = 1; nloads++;
    if (u->expert >= 0) { used += u->bytes; q_push(ui); }
    else pinned_bytes += u->bytes;
    log_load(u, total, now_us() - t0);
wake:;
    struct uffdio_range wr = { .start = pg_down(addr), .len = PG };
    ioctl(uffd, UFFDIO_WAKE, &wr);
    wait_close(t);
}

// Leitora: só lê o uffd e carimba a chegada (sem ela, no sync/parallel os faults esperariam na fila do
// kernel enquanto o handler carrega, e a espera medida sairia menor que a real).
#define FQN 4096
static struct { uint64_t addr, t; } fq[FQN];
static int fq_head, fq_count;
static pthread_mutex_t fq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t fq_cv = PTHREAD_COND_INITIALIZER, fq_space = PTHREAD_COND_INITIALIZER;
static pthread_t reader_tid;

static void *reader(void *arg) {
    (void)arg;
    sigset_t all; sigfillset(&all); pthread_sigmask(SIG_BLOCK, &all, NULL);
    for (;;) {
        struct uffd_msg msg[16];
        ssize_t n = read(uffd, msg, sizeof msg);
        if (n < 0) { if (errno == EINTR || errno == EAGAIN) continue; break; }
        uint64_t t = now_us();
        for (ssize_t i = 0; i < n / (ssize_t)sizeof(struct uffd_msg); i++) {
            if (msg[i].event != UFFD_EVENT_PAGEFAULT) continue;
            wait_open(t);
            pthread_mutex_lock(&fq_mu);
            while (fq_count == FQN) pthread_cond_wait(&fq_space, &fq_mu);
            fq[(fq_head + fq_count) % FQN].addr = msg[i].arg.pagefault.address;
            fq[(fq_head + fq_count) % FQN].t = t;
            fq_count++;
            pthread_cond_signal(&fq_cv);
            pthread_mutex_unlock(&fq_mu);
        }
    }
    return NULL;
}

static void *handler(void *arg) {
    (void)arg;
    sigset_t all; sigfillset(&all); pthread_sigmask(SIG_BLOCK, &all, NULL);
    for (;;) {
        pthread_mutex_lock(&fq_mu);
        while (!fq_count) pthread_cond_wait(&fq_cv, &fq_mu);
        uint64_t addr = fq[fq_head].addr, t = fq[fq_head].t;
        fq_head = (fq_head + 1) % FQN; fq_count--;
        pthread_cond_signal(&fq_space);
        pthread_mutex_unlock(&fq_mu);
        handle_fault(addr, t);
        if (logf_ && nloads % 256 == 0) fflush(logf_);
    }
    return NULL;
}

// ---------------------------------------------------------------- setup
static int open_uffd(int flags) {
    int fd = syscall(SYS_userfaultfd, flags | UFFD_USER_MODE_ONLY);
    if (fd < 0) fd = syscall(SYS_userfaultfd, flags);
    if (fd < 0) {
        int dev = open("/dev/userfaultfd", O_RDWR | O_CLOEXEC);
        if (dev >= 0) { fd = ioctl(dev, USERFAULTFD_IOC_NEW, flags | UFFD_USER_MODE_ONLY); close(dev); }
    }
    return fd;
}

static int init_uffd(void) {
    int probe = open_uffd(O_CLOEXEC);
    if (probe < 0) { fprintf(stderr, "[pager] userfaultfd indisponível: %s\n", strerror(errno)); return -1; }
    struct uffdio_api api = { .api = UFFD_API, .features = 0 };
    if (ioctl(probe, UFFDIO_API, &api)) { close(probe); return -1; }
    uint64_t want = api.features & UFFD_FEATURE_EXACT_ADDRESS;
    close(probe);
    uffd = open_uffd(O_CLOEXEC);
    api = (struct uffdio_api){ .api = UFFD_API, .features = want };
    if (uffd < 0 || ioctl(uffd, UFFDIO_API, &api)) { fprintf(stderr, "[pager] UFFDIO_API falhou\n"); return -1; }
    LOGV("userfaultfd ok (endereço exato: %s)\n", want ? "sim" : "não");

    const char *m = getenv("PAGER_MODE");
    mode_lru = m && !strcmp(m, "lru");
    mode_fifo = (m && !strcmp(m, "fifo")) || mode_lru;
    router_on = getenv("PAGER_ROUTER") ? atoi(getenv("PAGER_ROUTER")) : mode_lru;
    budget = (uint64_t)(getenv("PAGER_BUDGET_MB") ? atoll(getenv("PAGER_BUDGET_MB")) : 3072) << 20;
    if (getenv("PAGER_DIRECT")) use_direct = atoi(getenv("PAGER_DIRECT"));
    io_parallel = getenv("PAGER_IO") && !strcmp(getenv("PAGER_IO"), "parallel");
    io_async = getenv("PAGER_IO") && !strcmp(getenv("PAGER_IO"), "async");
    if (getenv("PAGER_BLOCK_KB")) block_bytes = pg_up((uint64_t)atoll(getenv("PAGER_BLOCK_KB")) << 10);
    if (block_bytes < PG) block_bytes = PG;
    if (block_bytes > CHUNK) block_bytes = CHUNK;
    if (getenv("PAGER_IO_THREADS")) nworkers = atoi(getenv("PAGER_IO_THREADS"));
    if (nworkers < 1) nworkers = 1;
    prefetch_on = getenv("PAGER_PREFETCH") ? atoi(getenv("PAGER_PREFETCH")) : (mode_lru && router_on && io_async);
    if (prefetch_on && !(router_on && io_async)) {
        fprintf(stderr, "[pager] PAGER_PREFETCH=1 exige router e PAGER_IO=async: prefetch desligado\n");
        prefetch_on = 0;
    }
    if (getenv("PAGER_PREFETCH_K")) prefetch_k = atoi(getenv("PAGER_PREFETCH_K"));
    if (prefetch_k < 1) prefetch_k = 1;
    if (prefetch_k > 64) prefetch_k = 64;
    if (getenv("PAGER_PREFETCH_MIN")) prefetch_min = atof(getenv("PAGER_PREFETCH_MIN"));
    if (prefetch_min < 0) prefetch_min = 0;
    evict_async = getenv("PAGER_EVICT_ASYNC") ? atoi(getenv("PAGER_EVICT_ASYNC")) : io_async;
    if (getenv("PAGER_EVICT_DELAY_US")) evict_delay_us = atoi(getenv("PAGER_EVICT_DELAY_US"));
    const char *lp = getenv("PAGER_LOG") ? getenv("PAGER_LOG") : "pager_trace.csv";
    logf_ = fopen(lp, "w");
    if (logf_) fprintf(logf_, "step,layer,expert,bytes,load_us\n");
    if (posix_memalign((void **)&iobuf, PG, CHUNK)) return -1;
    fprintf(stderr, "[pager] modo=%s orçamento=%llu MB direct=%d io=%s threads_io=%d bloco=%llu KB log=%s\n",
            mode_lru ? "lru" : mode_fifo ? "fifo" : "trace", (unsigned long long)(budget >> 20), use_direct,
            io_async ? "async" : io_parallel ? "parallel" : "sync", (io_parallel || io_async) ? nworkers : 0,
            (unsigned long long)(io_async ? block_bytes >> 10 : 0), lp);
    return 0;
}

static void report(void) {
    if (logf_) fflush(logf_);
    if (!nregions) return;
    fprintf(stderr, "[pager] passos=%llu cargas=%llu despejos=%llu lidos=%.2f GB io=%.1f s carga=%.1f s "
            "(%.0f MB/s) io_modo=%s faults_em_carga=%llu experts_residentes=%.0f MB fixos=%.0f MB paginas_avulsas=%llu\n",
            (unsigned long long)step, (unsigned long long)nloads, (unsigned long long)nevict,
            bytes_read / 1e9, io_us / 1e6, load_us / 1e6, load_us ? bytes_read / (double)load_us : 0.0,
            io_async ? "async" : io_parallel ? "parallel" : "sync", (unsigned long long)nwait,
            used / 1048576.0, pinned_bytes / 1048576.0,
            (unsigned long long)nunknown);
    double gen = step ? (double)step : 1.0;
    fprintf(stderr, "[pager] router=%s politica=%s callbacks=%llu cb=%.2f ms/token (geração; ask=%.2f lock=%.2f "
            "trabalho=%.2f: despejo=%.2f [%llu] despacho=%.2f) cb_total=%.1f ms toques=%llu cargas_router=%llu\n",
            router_on ? (cb_seen ? "ativo" : "sem callbacks") : "desligado",
            mode_lru ? "lru" : mode_fifo ? "fifo" : "trace", (unsigned long long)ncb,
            cb_gen_ns / 1e6 / gen, cb_ask_ns / 1e6 / gen, cb_lock_ns / 1e6 / gen, cb_work_ns / 1e6 / gen,
            cb_evict_ns / 1e6 / gen, (unsigned long long)cb_nevict, cb_submit_ns / 1e6 / gen,
            cb_ns / 1e6, (unsigned long long)ncb_touch, (unsigned long long)ncb_loads);
    fprintf(stderr, "[pager] espera=%.1f ms/token (geração, relógio com >= 1 thread parada; soma=%.1f ms/token, "
            "%.1f faults/token) espera_total=%.1f s esperadores_pendentes=%d estouro=%llu\n",
            wait_union_gen_us / 1e3 / gen, wait_sum_gen_us / 1e3 / gen, nwaits_gen / gen,
            wait_union_us / 1e6, nw, (unsigned long long)nw_overflow);
    fprintf(stderr, "[pager] despejo=%s adiados=%llu resgatados=%llu esperas_madvise=%llu obsoletos=%llu "
            "fila_cheia=%llu fila_max=%d madvise_worker=%.2f ms/token (geração) faults_em_residente=%llu "
            "violacoes=%llu\n",
            evict_async ? "async" : "sync", (unsigned long long)nev_deferred, (unsigned long long)nev_rescued,
            (unsigned long long)nev_waited, (unsigned long long)nev_stale, (unsigned long long)nev_sync_full,
            evq_max, ev_madvise_gen_ns / 1e6 / gen, (unsigned long long)nfault_resident,
            (unsigned long long)nev_violation);
    fprintf(stderr, "[pager] prefetch=%s K=%d min=%.3f emitidos=%.1f uteis=%.1f desperdicados=%.1f por token "
            "(geração; total %llu/%llu/%llu) abaixo_do_limiar=%llu sem_orcamento=%llu cb_prefetch=%.2f ms/token\n",
            prefetch_on ? "markov" : "desligado", prefetch_k, prefetch_min, npf_gen_issued / gen, npf_gen_useful / gen,
            npf_gen_wasted / gen, (unsigned long long)npf_issued, (unsigned long long)npf_useful,
            (unsigned long long)npf_wasted, (unsigned long long)npf_below_min, (unsigned long long)npf_skipped_budget,
            cb_pf_ns / 1e6 / gen);
    if (prefetch_on) {
        double h0 = step < 32 ? (double)step : 32.0, h1 = step > 32 ? (double)(step - 32) : 0.0;
        fprintf(stderr, "[pager] prefetch_metades=passos 1-32: emitidos=%.1f uteis=%.1f desperdicados=%.1f por token | "
                "passos 33-%llu: emitidos=%.1f uteis=%.1f desperdicados=%.1f por token | "
                "madvise_no_prefetch=%.2f ms/token (%llu despejos) de cb_prefetch=%.2f ms/token\n",
                pf_half_issued[0] / (h0 ? h0 : 1), pf_half_useful[0] / (h0 ? h0 : 1), pf_half_wasted[0] / (h0 ? h0 : 1),
                (unsigned long long)step, pf_half_issued[1] / (h1 ? h1 : 1), pf_half_useful[1] / (h1 ? h1 : 1),
                pf_half_wasted[1] / (h1 ? h1 : 1), pf_madvise_ns / 1e6 / gen, (unsigned long long)pf_evict_n,
                cb_pf_ns / 1e6 / gen);
    }
}

// ---------------------------------------------------------------- router (v4): bit de acesso pelo cb_eval
static int in_queue(int ui) { return U[ui].prev >= 0 || U[ui].next >= 0 || q_head == ui; }

// Chamado com st_mu: experts da camada L escolhidos pelo router (ids[], n distintos).
static uint16_t *trans;                          // Markov: trans[L][a][b] = vezes que a (camada L) levou a b (L+1)
static int32_t prev_tok[8192]; static int prev_L = -1, prev_ntok, prev_nused; static uint64_t prev_step;

static void markov_learn(int L, const int32_t *tok, int nused, int ntok) {
    if (!trans) trans = calloc((size_t)emap_layers * emap_maxe * emap_maxe, sizeof(uint16_t));
    if (!trans) return;
    if (L >= 1 && prev_L == L - 1 && prev_step == step && prev_ntok == ntok) {
        for (int j = 0; j < ntok; j++)               // pares do mesmo token: experts de L-1 -> experts de L
            for (int a = 0; a < prev_nused; a++)
                for (int b = 0; b < nused; b++) {
                    int ea = prev_tok[j * prev_nused + a], eb = tok[j * nused + b];
                    if (ea < 0 || eb < 0 || (uint64_t)ea >= emap_maxe || (uint64_t)eb >= emap_maxe) continue;
                    uint16_t *c = &trans[((size_t)(L - 1) * emap_maxe + ea) * emap_maxe + eb];
                    if (*c < UINT16_MAX) (*c)++;
                }
    }
    if (nused * ntok <= (int)(sizeof prev_tok / sizeof prev_tok[0])) {
        memcpy(prev_tok, tok, sizeof(int32_t) * nused * ntok);
        prev_L = L; prev_ntok = ntok; prev_nused = nused; prev_step = step;
    } else prev_L = -1;
}

// Prevê os experts de L+1 a partir dos de L e despacha os top-K com prioridade baixa.
static void markov_prefetch(int L, const int *ids, int n) {
    if (!trans || L + 1 >= emap_layers || L + 1 >= 40) return;
    uint32_t votes[4096] = {0};
    uint64_t total = 0;
    for (int i = 0; i < n; i++) {
        const uint16_t *row = &trans[((size_t)L * emap_maxe + ids[i]) * emap_maxe];
        for (uint64_t e = 0; e < emap_maxe; e++) { votes[e] += row[e]; total += row[e]; }
    }
    double thr = prefetch_min * (double)total;       // limiar de confiança (0 = K fixo)
    for (int k = 0; k < prefetch_k; k++) {           // seleção dos top-K com voto > 0
        uint64_t best = 0; uint32_t bv = 0;
        for (uint64_t e = 0; e < emap_maxe; e++) if (votes[e] > bv) { bv = votes[e]; best = e; }
        if (!bv) break;
        if ((double)bv < thr) { npf_below_min += (uint64_t)(prefetch_k - k); break; }
        votes[best] = 0;
        int ui = emap_g[(uint64_t)(L + 1) * emap_maxe + best];
        if (ui < 0) continue;
        unit_t *u = &U[ui];
        if (__atomic_load_n(&u->resident, __ATOMIC_ACQUIRE) || __atomic_load_n(&u->loading, __ATOMIC_ACQUIRE)) continue;
        int ok = 1;
        while (used + u->bytes > budget) if (!evict_oldest_unmarked()) { ok = 0; break; }
        if (!ok) { npf_skipped_budget++; return; }   // nunca despeja o que o token atual já usou
        if (reclaim(ui)) continue;                   // estava com despejo pendente: voltou sem ler o disco
        if (!submit_unit_prio(&R[u->region], ui, 1)) return;
        nloads++; npf_issued++;
        if (step >= 1) { npf_gen_issued++; pf_half_issued[half_of_step()]++; }
        u->pf = 1; used += u->bytes; q_push(ui);
    }
}

static void router_layer(int L, const int *ids, int n, const int32_t *tok, int nused, int ntok) {
    if (L < cb_last_layer) {                     // camada voltou -> novo token
        step++;
        if (!mode_fifo) while (evict_oldest()) ;
    }
    cb_last_layer = L; cb_seen = 1;
    for (int i = 0; i < n; i++) {                // 1) bit de acesso: no LRU, vai pro fim da fila
        int ui = emap_g[(uint64_t)L * emap_maxe + ids[i]];
        if (ui < 0) continue;
        mark_used(ui);
        if (mode_lru && in_queue(ui)) { q_remove(ui); q_push(ui); ncb_touch++; }
    }
    if (prefetch_on) markov_learn(L, tok, nused, ntok);
    if (!io_async) return;                       // carga antecipada só no async; nos outros, o fault carrega
    for (int i = 0; i < n; i++) {                // 2) despacha os que faltam, antes do mul_mat_id
        int ui = emap_g[(uint64_t)L * emap_maxe + ids[i]];
        if (ui < 0) continue;
        unit_t *u = &U[ui];
        if (__atomic_load_n(&u->resident, __ATOMIC_ACQUIRE) || __atomic_load_n(&u->loading, __ATOMIC_ACQUIRE)) continue;
        uint64_t te = ns_now(); uint64_t ne = nevict;
        while (used + u->bytes > budget && evict_oldest()) ;
        uint64_t ts = ns_now();
        if (reclaim(ui)) { if (step >= 1) { cb_evict_ns += ts - te; cb_nevict += nevict - ne; } continue; }
        nloads++; ncb_loads++;
        used += u->bytes; q_push(ui);
        submit_unit(&R[u->region], ui);
        if (step >= 1) { cb_evict_ns += ts - te; cb_submit_ns += ns_now() - ts; cb_nevict += nevict - ne; }
    }
    if (prefetch_on && step >= 1) {              // 3) Markov: prevê L+1 (só na geração, como o simulador)
        uint64_t tp = ns_now();
        in_prefetch = 1;
        markov_prefetch(L, ids, n);
        in_prefetch = 0;
        cb_pf_ns += ns_now() - tp;
    }
}

static bool pager_cb(struct ggml_tensor *t, bool ask, void *ud) {
    (void)ud;
    uint64_t t0 = ns_now();
    int mine = 0, L = -1;
    if (!strncmp(t->name, "ffn_moe_topk-", 13)) {
        L = atoi(t->name + 13);
        mine = emap_g && L >= 0 && L < emap_layers && L < 40 && t->type == GGML_TYPE_I32;
    }
    bool prev_wants = prev_cb && prev_cb(t, true, prev_ud);
    bool ret = true;
    if (ask) {
        ret = mine || prev_wants;
        if (step >= 1) __atomic_add_fetch(&cb_ask_ns, ns_now() - t0, __ATOMIC_RELAXED);
    } else {
        if (mine) {
            // ffn_moe_topk é uma view do argsort: [n_expert_used, n_tokens] com strides nb[0], nb[1]
            static int32_t tok[8192];            // só a thread principal do llama chama o callback
            int ids[1024], n = 0; unsigned char seen[4096] = {0};
            int nused = (int)t->ne[0], ntok = (int)t->ne[1];
            if ((int64_t)nused * ntok > (int64_t)(sizeof tok / sizeof tok[0])) ntok = (int)(sizeof tok / sizeof tok[0]) / nused;
            for (int64_t j = 0; j < t->ne[1]; j++)
                for (int64_t i = 0; i < t->ne[0]; i++) {
                    int e = *(const int32_t *)((const char *)t->data + i * t->nb[0] + j * t->nb[1]);
                    if (j < ntok) tok[j * nused + i] = e;
                    if (e >= 0 && (uint64_t)e < emap_maxe && e < 4096 && !seen[e] && n < 1024) { seen[e] = 1; ids[n++] = e; }
                }
            uint64_t tl = ns_now();
            pthread_mutex_lock(&st_mu);
            uint64_t tw = ns_now();
            router_layer(L, ids, n, tok, nused, ntok);
            ncb++;
            if (step >= 1) { cb_lock_ns += tw - tl; cb_work_ns += ns_now() - tw; }
            pthread_mutex_unlock(&st_mu);
        }
        if (prev_wants) ret = prev_cb(t, false, prev_ud);
    }
    uint64_t dt = ns_now() - t0;
    __atomic_add_fetch(&cb_ns, dt, __ATOMIC_RELAXED);
    if (step >= 1) __atomic_add_fetch(&cb_gen_ns, dt, __ATOMIC_RELAXED);
    return ret;
}

// Intercepta a criação do contexto pra encadear o cb_eval (preserva o que já existir).
struct llama_context *llama_init_from_model(struct llama_model *model, struct llama_context_params params) {
    static struct llama_context *(*real)(struct llama_model *, struct llama_context_params);
    if (!real) real = (struct llama_context *(*)(struct llama_model *, struct llama_context_params))
                      dlsym(RTLD_NEXT, "llama_init_from_model");
    if (uffd >= 0 && router_on && emap_g && params.cb_eval != pager_cb) {
        prev_cb = params.cb_eval; prev_ud = params.cb_eval_user_data;
        params.cb_eval = pager_cb; params.cb_eval_user_data = NULL;
        fprintf(stderr, "[pager] router: cb_eval instalado (%s), %d camadas x %llu experts\n",
                prev_cb ? "encadeado com o existente" : "sem callback anterior", emap_layers,
                (unsigned long long)emap_maxe);
    }
    return real(model, params);
}

static void *setup_region(int fd, size_t len) {
    char lnk[64], path[4096];
    snprintf(lnk, sizeof lnk, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(lnk, path, sizeof path - 1);
    if (n <= 5) return NULL;
    path[n] = 0;
    if (strcmp(path + n - 5, ".gguf")) return NULL;
    struct stat st;
    if (fstat(fd, &st) || (uint64_t)st.st_size < min_bytes) return NULL;

    pthread_mutex_lock(&setup_mu);
    void *ret = NULL;
    if (nregions == MAX_REGIONS) goto out;
    if (uffd < 0) { if (init_uffd()) goto out; atexit(report); }

    int ri = nregions;
    region_t *g = &R[ri];
    memset(g, 0, sizeof *g);
    g->len = len; g->fsize = st.st_size;
    g->fd = use_direct ? open(path, O_RDONLY | O_DIRECT | O_CLOEXEC) : -1;
    g->direct = g->fd >= 0;
    if (g->fd < 0) g->fd = open(path, O_RDONLY | O_CLOEXEC);
    if (g->fd < 0) goto out;
    int pfd = open(path, O_RDONLY | O_CLOEXEC);
    int ok = parse_gguf(ri, pfd); close(pfd);
    if (ok) { close(g->fd); goto out; }

    g->base = real_mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (g->base == MAP_FAILED) { close(g->fd); goto out; }
    madvise(g->base, len, MADV_NOHUGEPAGE);
    struct uffdio_register reg = { .range = { .start = (uint64_t)g->base, .len = pg_up(len) }, .mode = UFFDIO_REGISTER_MODE_MISSING };
    if (ioctl(uffd, UFFDIO_REGISTER, &reg)) {
        fprintf(stderr, "[pager] UFFDIO_REGISTER: %s\n", strerror(errno));
        munmap(g->base, len); close(g->fd); goto out;
    }
    if (nregions == 0) {
        pthread_create(&handler_tid, NULL, handler, NULL);
        pthread_create(&reader_tid, NULL, reader, NULL);
        if (evict_async) { pthread_t t; pthread_create(&t, NULL, evictor, NULL); }
        if (io_parallel || io_async)
            for (int i = 0; i < nworkers; i++) { pthread_t t; pthread_create(&t, NULL, io_worker, NULL); }
    }
    nregions++;
    fprintf(stderr, "[pager] interceptado %s (%.2f GB)\n", path, len / 1e9);
    ret = g->base;
out:
    pthread_mutex_unlock(&setup_mu);
    return ret;
}

// ---------------------------------------------------------------- hooks
__attribute__((constructor)) static void pager_init(void) {
    real_mmap = dlsym(RTLD_NEXT, "mmap");
    verbose = getenv("PAGER_VERBOSE") && atoi(getenv("PAGER_VERBOSE"));
    min_bytes = (uint64_t)(getenv("PAGER_MIN_MB") ? atoll(getenv("PAGER_MIN_MB")) : 64) << 20;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!real_mmap) real_mmap = dlsym(RTLD_NEXT, "mmap");
    if (fd >= 0 && off == 0 && !(prot & PROT_WRITE) && !(flags & MAP_FIXED)) {
        void *r = setup_region(fd, len);
        if (r) return r;
    }
    return real_mmap(addr, len, prot, flags, fd, off);
}
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) __attribute__((alias("mmap")));
