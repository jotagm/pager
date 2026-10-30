// libnopopulate.c — baseline justo: o kernel decide a paginação, sem carga antecipada forçada.
// O llama.cpp mapeia o .gguf com MAP_POPULATE e pede POSIX_MADV_WILLNEED no arquivo inteiro
// (src/llama-mmap.cpp, construtor do llama_mmap). Com um modelo maior que a RAM, isso lê o arquivo
// todo no carregamento. Este shim tira o MAP_POPULATE e ignora WILLNEED nas regiões de .gguf.
// NOPOP_RANDOM=1: aplica também MADV_RANDOM na região, desligando o read-around do kernel nos faults
// (read_ahead_kb=4096 neste NVMe/btrfs: cada fault lia até 4 MB em volta).
// Uso: [NOPOP_RANDOM=1] LD_PRELOAD=./libnopopulate.so <llama.cpp ...>
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define MAXR 16
static struct { uintptr_t a, b; } R[MAXR];
static int nr, npop, nwill, nrand;

static int is_gguf(int fd) {
    char lnk[64], p[4096];
    snprintf(lnk, sizeof lnk, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(lnk, p, sizeof p - 1);
    return n > 5 && !memcmp(p + n - 5, ".gguf", 5);   // readlink não termina com NUL
}
static int in_gguf(void *addr) {
    for (int i = 0; i < nr; i++) if ((uintptr_t)addr >= R[i].a && (uintptr_t)addr < R[i].b) return 1;
    return 0;
}
static void report(void) {
    fprintf(stderr, "[nopopulate] regiões .gguf=%d MAP_POPULATE removido=%d WILLNEED ignorados=%d MADV_RANDOM=%d\n",
            nr, npop, nwill, nrand);
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    static void *(*real)(void *, size_t, int, int, int, off_t);
    if (!real) real = dlsym(RTLD_NEXT, "mmap");
    int g = fd >= 0 && is_gguf(fd);
    if (g && (flags & MAP_POPULATE)) { flags &= ~MAP_POPULATE; npop++; }
    void *r = real(addr, len, prot, flags, fd, off);
    if (g && r != MAP_FAILED && nr < MAXR) {
        if (!nr) atexit(report);
        R[nr].a = (uintptr_t)r; R[nr].b = (uintptr_t)r + len; nr++;
        const char *e = getenv("NOPOP_RANDOM");
        if (e && atoi(e) && madvise(r, len, MADV_RANDOM) == 0) nrand++;
    }
    return r;
}
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) __attribute__((alias("mmap")));

int madvise(void *addr, size_t len, int advice) {
    static int (*real)(void *, size_t, int);
    if (!real) real = dlsym(RTLD_NEXT, "madvise");
    if (advice == MADV_WILLNEED && in_gguf(addr)) { nwill++; return 0; }
    return real(addr, len, advice);
}

int posix_madvise(void *addr, size_t len, int advice) {
    static int (*real)(void *, size_t, int);
    if (!real) real = dlsym(RTLD_NEXT, "posix_madvise");
    if (advice == POSIX_MADV_WILLNEED && in_gguf(addr)) { nwill++; return 0; }
    return real(addr, len, advice);
}
