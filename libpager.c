// libpager.c — pager de experts MoE em user space via userfaultfd.
//
// Uso: LD_PRELOAD=./libpager.so <llama.cpp ...>
// Intercepta o mmap() do arquivo .gguf e devolve uma região anônima registrada
// no userfaultfd. Cada page fault chega pra thread handler, que descobre qual
// (camada, expert) foi tocado, lê do disco e insere com UFFDIO_COPY.
//
// Variáveis de ambiente:
//   PAGER_MODE       trace | fifo            (default: trace)
//     trace: a cada novo token despeja todos os experts -> cada acesso vira um
//            fault -> log = trace completo de acessos (lento, só pra gerar trace).
//            Respeita PAGER_BUDGET_MB dentro do passo (necessário no passo do prompt).
//     fifo : orçamento fixo de RAM pros experts, despejo por ordem de carga.
//            (o pager só vê MISSES; sem ver hits, LRU degenera em FIFO)
//   PAGER_BUDGET_MB  orçamento dos experts no modo fifo (default: 3072)
//   PAGER_LOG        arquivo CSV de log (default: pager_trace.csv)
//   PAGER_DIRECT     1 = lê com O_DIRECT, sem page cache (default: 1)
//   PAGER_MIN_MB     só intercepta .gguf maiores que isso (default: 64)
//   PAGER_IO         sync | parallel     (default: sync)
//     sync    : v1 — o handler lê e copia cada fatia em sequência
//     parallel: v2 — um pool de threads lê as fatias (gate/up/down, e blocos de 8 MB dos tensores
//               fixos) em paralelo; cada worker faz pread + UFFDIO_COPY, sobrepondo leitura e cópia
//   PAGER_IO_THREADS workers do modo parallel (default: 4)
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
static int mode_fifo;
static uint64_t budget;
static int use_direct = 1, verbose;
static uint64_t min_bytes;
static FILE *logf_;

// estado
static uint64_t used, pinned_bytes, step, nloads, nevict, bytes_read, io_us, nunknown;
static uint64_t load_us;              // tempo de relógio do handler carregando (leitura + cópia)
static int io_parallel, nworkers = 4;
static int last_layer = -1;
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
#define QN 4096
typedef struct { region_t *g; uint64_t a, b; } job_t;
static job_t jobq[QN];
static int jq_head, jq_count, jq_pending;
static pthread_mutex_t jq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t jq_cv = PTHREAD_COND_INITIALIZER, jq_space = PTHREAD_COND_INITIALIZER,
                      jq_done = PTHREAD_COND_INITIALIZER;

static void *io_worker(void *arg) {
    (void)arg;
    sigset_t all; sigfillset(&all); pthread_sigmask(SIG_BLOCK, &all, NULL);
    char *buf;
    if (posix_memalign((void **)&buf, PG, CHUNK)) return NULL;
    for (;;) {
        pthread_mutex_lock(&jq_mu);
        while (!jq_count) pthread_cond_wait(&jq_cv, &jq_mu);
        job_t j = jobq[jq_head]; jq_head = (jq_head + 1) % QN; jq_count--;
        pthread_cond_signal(&jq_space);
        pthread_mutex_unlock(&jq_mu);
        read_file(j.g, j.a, j.b - j.a, buf);
        uffd_copy(j.g->base + j.a, buf, j.b - j.a);
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
        jobq[(jq_head + jq_count) % QN] = (job_t){ g, p, b - p < CHUNK ? b : p + CHUNK };
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

static void evict(int i) {
    unit_t *u = &U[i]; region_t *g = &R[u->region];
    for (int s = 0; s < u->nslices; s++) {
        uint64_t a = pg_up(u->sl[s].off), b = pg_down(u->sl[s].off + u->sl[s].len);
        if (b > a) madvise(g->base + a, b - a, MADV_DONTNEED);   // páginas de borda ficam
    }
    u->resident = 0; used -= u->bytes; nevict++;
    q_remove(i);
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

static void handle_fault(uint64_t addr) {
    region_t *g = NULL;
    for (int i = 0; i < nregions; i++)
        if (addr >= (uint64_t)R[i].base && addr < (uint64_t)R[i].base + R[i].len) { g = &R[i]; break; }
    if (!g) return;
    uint64_t off = addr - (uint64_t)g->base;
    int ui = find_unit(g, off);

    if (ui < 0 || U[ui].resident) {             // header/padding, ou página de borda: só a página
        uint64_t tp = now_us();
        load_file_range(g, pg_down(off), pg_down(off) + PG);
        load_us += now_us() - tp;
        if (ui < 0) nunknown++;
        goto wake;
    }
    unit_t *u = &U[ui];
    if (u->expert >= 0) {
        if (u->layer < last_layer) {             // camada voltou -> novo token (passo)
            step++;
            if (!mode_fifo) while (q_head >= 0) evict(q_head);
        }
        last_layer = u->layer;
        // Também no trace: o passo do prompt toca milhares de experts (~7,5 GB no Qwen3.6 com 97 tokens)
        // e estoura o MemoryMax se nada for despejado antes da troca de token. Nos passos de geração o
        // conjunto do token (~590 MB) cabe no orçamento, então o trace desses passos continua completo;
        // só o passo 0 pode ter recargas, e o simulate.py o descarta (--skip 1).
        while (used + u->bytes > budget && q_head >= 0) evict(q_head);
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
}

static void *handler(void *arg) {
    (void)arg;
    sigset_t all; sigfillset(&all); pthread_sigmask(SIG_BLOCK, &all, NULL);
    for (;;) {
        struct uffd_msg msg[16];
        ssize_t n = read(uffd, msg, sizeof msg);
        if (n < 0) { if (errno == EINTR || errno == EAGAIN) continue; break; }
        for (ssize_t i = 0; i < n / (ssize_t)sizeof(struct uffd_msg); i++)
            if (msg[i].event == UFFD_EVENT_PAGEFAULT) handle_fault(msg[i].arg.pagefault.address);
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
    mode_fifo = m && !strcmp(m, "fifo");
    budget = (uint64_t)(getenv("PAGER_BUDGET_MB") ? atoll(getenv("PAGER_BUDGET_MB")) : 3072) << 20;
    if (getenv("PAGER_DIRECT")) use_direct = atoi(getenv("PAGER_DIRECT"));
    io_parallel = getenv("PAGER_IO") && !strcmp(getenv("PAGER_IO"), "parallel");
    if (getenv("PAGER_IO_THREADS")) nworkers = atoi(getenv("PAGER_IO_THREADS"));
    if (nworkers < 1) nworkers = 1;
    const char *lp = getenv("PAGER_LOG") ? getenv("PAGER_LOG") : "pager_trace.csv";
    logf_ = fopen(lp, "w");
    if (logf_) fprintf(logf_, "step,layer,expert,bytes,load_us\n");
    if (posix_memalign((void **)&iobuf, PG, CHUNK)) return -1;
    fprintf(stderr, "[pager] modo=%s orçamento=%llu MB direct=%d io=%s threads_io=%d log=%s\n",
            mode_fifo ? "fifo" : "trace", (unsigned long long)(budget >> 20), use_direct,
            io_parallel ? "parallel" : "sync", io_parallel ? nworkers : 0, lp);
    return 0;
}

static void report(void) {
    if (logf_) fflush(logf_);
    if (!nregions) return;
    fprintf(stderr, "[pager] passos=%llu cargas=%llu despejos=%llu lidos=%.2f GB io=%.1f s carga=%.1f s "
            "(%.0f MB/s) io_modo=%s experts_residentes=%.0f MB fixos=%.0f MB paginas_avulsas=%llu\n",
            (unsigned long long)step, (unsigned long long)nloads, (unsigned long long)nevict,
            bytes_read / 1e9, io_us / 1e6, load_us / 1e6, load_us ? bytes_read / (double)load_us : 0.0,
            io_parallel ? "parallel" : "sync", used / 1048576.0, pinned_bytes / 1048576.0,
            (unsigned long long)nunknown);
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
        if (io_parallel) for (int i = 0; i < nworkers; i++) { pthread_t t; pthread_create(&t, NULL, io_worker, NULL); }
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
