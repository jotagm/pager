#!/usr/bin/env bash
# Corrida do despejo adiado (v6): o worker dorme entre as fatias do madvise
# (PAGER_EVICT_DELAY_US), e o orçamento de 1 MB força despejos e recargas da mesma unidade.
# Falha se: erro de byte, thread presa (timeout), violação do invariante (carga iniciada durante o
# madvise da unidade) ou se a corrida não aconteceu nenhuma vez (o teste passaria no vazio).
# Uso: race.sh [libpager.so] [repetições]
set -u
LIB=${1:-../libpager.so}; REP=${2:-6}; DELAY=${RACE_DELAY_US:-2000}
cd "$(dirname "$0")"
enc=0; viol=0; fail=0
for i in $(seq "$REP"); do
  for m in fifo lru; do
    out=$(PAGER_IO=async PAGER_MODE=$m PAGER_BUDGET_MB=1 PAGER_EVICT_DELAY_US=$DELAY PAGER_MIN_MB=0 \
          PAGER_LOG=/dev/null LD_PRELOAD=$LIB timeout 60 ./test_reader fake.gguf 2>&1); rc=$?
    e=$(grep -oE 'esperas_madvise=[0-9]+' <<<"$out" | cut -d= -f2 | head -1)
    v=$(grep -oE 'violacoes=[0-9]+' <<<"$out" | cut -d= -f2 | head -1)
    enc=$((enc + ${e:-0})); viol=$((viol + ${v:-0}))
    if [ $rc -ne 0 ] || ! grep -q '^OK: 0 erros' <<<"$out"; then
      fail=$((fail + 1)); echo "FALHA ($m, rc=$rc): $(grep -E 'OK|FALHOU|ERRO' <<<"$out" | head -2 | tr '\n' ' ')"
    fi
  done
done
echo "corrida: $((REP * 2)) execuções, madvise em andamento encontrado $enc vezes, violações $viol, falhas $fail"
[ $fail -eq 0 ] && [ $viol -eq 0 ] && [ $enc -gt 0 ] && echo "OK: corrida do despejo adiado" && exit 0
[ $enc -eq 0 ] && echo "FALHOU: a corrida não foi exercitada"
exit 1
