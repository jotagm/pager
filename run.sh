#!/usr/bin/env bash
# Roda uma geração do Qwen no llama.cpp com memória limitada por cgroup e mede.
#   ./run.sh baseline          -> mmap normal via libnopopulate: o kernel decide a paginação, sem
#                                 MAP_POPULATE nem WILLNEED na carga (referência para comparar com o pager)
#   ./run.sh baseline-populate -> comportamento padrão do llama.cpp (MAP_POPULATE + WILLNEED), só pra registro
#   ./run.sh baseline-random   -> como baseline, mais MADV_RANDOM no .gguf (sem read-around nos faults)
#   ./run.sh trace      -> libpager em modo trace: grava o trace completo de acessos (lento)
#   ./run.sh fifo       -> libpager com orçamento fixo e despejo FIFO
#   ./run.sh lru        -> libpager com orçamento fixo e LRU real pelo router (cb_eval); use PAGER_IO=async
set -euo pipefail

MODE=${1:?uso: ./run.sh baseline|baseline-populate|baseline-random|trace|fifo|lru}
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:?defina MODEL=/caminho/do/modelo.gguf}
LLAMA=${LLAMA:-$HOME/llama.cpp/build/bin/llama-completion}
MEM_MAX=${MEM_MAX:-6G}          # teto do cgroup (processo inteiro: pesos + KV + buffers)
BUDGET_MB=${BUDGET_MB:-3072}    # orçamento só dos experts (modo fifo)
NTOK=${NTOK:-64}
THREADS=${THREADS:-4}
PROMPT=${PROMPT:-$HERE/prompt.txt}
TIMEOUT=${TIMEOUT:-}            # ex.: 30m. Manda SIGINT: o llama-completion imprime as estatísticas e sai (130)
# flags conferidas com --help do llama-completion (llama.cpp 5047748, 2026-10-30):
#   -no-cnv           o modo conversa liga sozinho quando o GGUF tem chat template (e é interativo)
#   --load-mode mmap  explícito: o default "auto" pode escolher outro modo ("dio") em algum dispositivo
#   --fit off         o default "on" ajusta argumentos não definidos conforme a memória livre
LLAMA_ARGS=${LLAMA_ARGS:---no-warmup --no-repack -no-cnv --load-mode mmap --fit off -c 2048 --temp 0 --seed 1}

OUT="$HERE/results/${MODE}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

# disco onde está o modelo -> contador de setores lidos
SRC=$(df --output=source "$MODEL" | tail -1)
DEV=${DEV:-$(lsblk -no PKNAME "$SRC" 2>/dev/null | head -1)}
[ -z "$DEV" ] && DEV=$(basename "$SRC")
sectors() { awk '{print $3}' "/sys/block/$DEV/stat"; }

echo "== $MODE | modelo=$(basename "$MODEL") | MemoryMax=$MEM_MAX | disco=$DEV | tokens=$NTOK"

# começa frio: tira do page cache todos os .gguf do diretório do modelo (cobre modelos divididos em partes).
# posix_fadvise(DONTNEED) não precisa de root para páginas limpas; o fincore confirma o que sobrou.
# Páginas mapeadas por outro processo vivo não saem: por isso o total residente vai para o summary.
GGUFS=("$(dirname "$MODEL")"/*.gguf)
python3 -I -c '
import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY)
    try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally: os.close(fd)
' "${GGUFS[@]}"
CACHE_RES=$(fincore -b -n -o RES "${GGUFS[@]}" | awk '{s+=$1} END{printf "%.1f", s/1048576}')
fincore -o RES,SIZE,FILE "${GGUFS[@]}"
echo "page cache dos .gguf após o DONTNEED: ${CACHE_RES} MB residentes"

TMO=()
[ -n "$TIMEOUT" ] && TMO=(timeout --signal=INT --kill-after=60s "$TIMEOUT")

ENV=()
case "$MODE" in
  baseline) ENV=(LD_PRELOAD="$HERE/libnopopulate.so") ;;
  baseline-populate) ;;
  baseline-random) ENV=(LD_PRELOAD="$HERE/libnopopulate.so" NOPOP_RANDOM=1) ;;
  trace) ENV=(LD_PRELOAD="$HERE/libpager.so" PAGER_MODE=trace PAGER_LOG="$OUT/trace.csv") ;;
  fifo)  ENV=(LD_PRELOAD="$HERE/libpager.so" PAGER_MODE=fifo PAGER_BUDGET_MB="$BUDGET_MB" PAGER_LOG="$OUT/trace.csv") ;;
  lru)   ENV=(LD_PRELOAD="$HERE/libpager.so" PAGER_MODE=lru PAGER_BUDGET_MB="$BUDGET_MB" PAGER_LOG="$OUT/trace.csv") ;;
  *) echo "modo inválido"; exit 2 ;;
esac

S0=$(sectors); T0=$(date +%s.%N)
# O bash dentro do scope é o pai do llama: num OOM o kernel mata o llama (o maior processo) e o bash
# sobrevive pra salvar memory.events/memory.peak antes de o scope sumir.
# OOMPolicy=continue: sem isso o systemd derruba o scope inteiro (e o bash) quando o OOM mata o llama.
# LC_NUMERIC=C: o llama.cpp chama setlocale e, em pt_BR, imprime "1,23 tokens per second" (o pager também).
# shellcheck disable=SC2086,SC2016
systemd-run --user --scope --quiet -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 -p OOMPolicy=continue \
  bash -c 'cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup); "$@"; rc=$?
           cat "$cg/memory.events" > "'"$OUT"'/memory.events"; cat "$cg/memory.peak" > "'"$OUT"'/memory.peak"; exit $rc' _ \
  /usr/bin/time -v -o "$OUT/time.txt" \
  "${TMO[@]}" env LC_NUMERIC=C "${ENV[@]}" "$LLAMA" -m "$MODEL" -f "$PROMPT" -n "$NTOK" -t "$THREADS" $LLAMA_ARGS \
  >"$OUT/output.txt" 2>"$OUT/llama.log" || echo "aviso: llama saiu com erro (veja $OUT/llama.log)"
T1=$(date +%s.%N); S1=$(sectors)

GB=$(awk -v a="$S0" -v b="$S1" 'BEGIN{printf "%.2f", (b-a)*512/1e9}')
WALL=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.1f", b-a}')
MAJ=$(awk -F': ' '/Major/{print $2}' "$OUT/time.txt" 2>/dev/null || echo "?")
RSS=$(awk -F': ' '/Maximum resident/{printf "%.0f", $2/1024}' "$OUT/time.txt" 2>/dev/null || echo "?")
TPS=$(grep -E "eval time" "$OUT/llama.log" | grep -v prompt | grep -oE "[0-9]+[.,][0-9]+ tokens per second" | tr , . | head -1 || true)

{
  echo "modo=$MODE mem_max=$MEM_MAX budget_mb=$BUDGET_MB ntok=$NTOK threads=$THREADS io=${PAGER_IO:-sync} timeout=${TIMEOUT:-nenhum}"
  grep -qs "Interrupted by user" "$OUT/llama.log" "$OUT/output.txt" && echo "INTERROMPIDO pelo timeout: $(grep -oE '/ +[0-9]+ runs' "$OUT/llama.log" | head -1) de geração"
  echo "cache inicial: ${CACHE_RES} MB dos .gguf residentes antes de rodar"
  echo "geracao:      ${TPS:-?}"
  echo "tempo total:  ${WALL}s"
  echo "lido do disco: ${GB} GB"
  echo "major faults: $MAJ"
  echo "RSS maximo:   ${RSS} MB"
  echo "scope: pico=$(awk '{printf "%.0f", $1/1048576}' "$OUT/memory.peak" 2>/dev/null || echo ?) MB" \
       "$(grep -E '^(max|oom|oom_kill) ' "$OUT/memory.events" 2>/dev/null | tr '\n' ' ')"
  grep -E "\[(pager|nopopulate)\] (passos|regiões)" "$OUT/llama.log" | tail -1 || true
  grep -E "\[pager\] (router|espera|prefetch|prefetch_metades)=" "$OUT/llama.log" || true
} | tee "$OUT/summary.txt"
echo "resultados em $OUT"
