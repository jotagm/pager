// Simula a inferência: mmap do .gguf (interceptado pelo libpager), e a cada
// "token" percorre as camadas lendo 8 experts aleatórios por camada.
// Confere cada byte lido contra pread() do arquivo.
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { int l, e; long off, len; } sl_t;
static sl_t S[100000]; static int ns;
static unsigned char *map; static int fd; static long fsz;
static int nlayers, nexp, ntok = 20, nthreads = 4;
static volatile int errors;

static void check(long off, long len) {
    unsigned char *ref = malloc(len);
    if (pread(fd, ref, len, off) != len) { errors++; free(ref); return; }
    if (memcmp(ref, map + off, len)) { fprintf(stderr, "ERRO em off=%ld len=%ld\n", off, len); errors++; }
    free(ref);
}

static int chosen[64][8];
static pthread_barrier_t bar;

static void *worker(void *arg) {
    long id = (long)arg;
    for (int t = 0; t < ntok; t++) {
        for (int l = 0; l < nlayers; l++) {
            if (id == 0) for (int k = 0; k < 8; k++) chosen[l][k] = rand() % nexp;
            pthread_barrier_wait(&bar);
            for (int i = 0; i < ns; i++)              // cada thread lê uma parte das fatias escolhidas
                if (S[i].l == l)
                    for (int k = 0; k < 8; k++)
                        if (S[i].e == chosen[l][k] && (i + k) % nthreads == id) check(S[i].off, S[i].len);
            pthread_barrier_wait(&bar);
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s fake.gguf\n", argv[0]); return 2; }
    char lp[4096]; snprintf(lp, sizeof lp, "%s.layout", argv[1]);
    FILE *f = fopen(lp, "r");
    while (fscanf(f, "%d %d %ld %ld", &S[ns].l, &S[ns].e, &S[ns].off, &S[ns].len) == 4) {
        if (S[ns].l + 1 > nlayers) nlayers = S[ns].l + 1;
        if (S[ns].e + 1 > nexp) nexp = S[ns].e + 1;
        ns++;
    }
    fclose(f);
    fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st); fsz = st.st_size;
    map = mmap(NULL, fsz, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    check(0, 4096);                    // header
    srand(42);
    pthread_barrier_init(&bar, NULL, nthreads);
    pthread_t th[16];
    for (long i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, (void *)i);
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    check(0, fsz);                     // varredura completa no final
    printf("%s: %d erros (%d tokens, %d camadas, %d experts)\n", errors ? "FALHOU" : "OK", errors, ntok, nlayers, nexp);
    return errors != 0;
}
