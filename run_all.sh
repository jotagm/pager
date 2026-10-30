#!/usr/bin/env bash
# Rodada completa (Ollama e Floci parados): teto do disco com fio, baseline, trace, fifo e simulador.
# Uma rodada por vez; uma falha não interrompe as seguintes (cada uma registra o próprio resultado).
set -uo pipefail
cd "$(dirname "$0")"
export MODEL=${MODEL:-$HOME/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
export NTOK=${NTOK:-64}
LOG=results/rodada-$(date +%Y%m%d-%H%M%S).log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "######## $* ($(date +%T))"; }

for qd in 1 16; do
  step "fio randread bs=2M direct iodepth=$qd"
  fio --name=teto --filename="$MODEL" --readonly --direct=1 --rw=randread --bs=2M \
      --ioengine=io_uring --iodepth=$qd --runtime=30 --time_based --output-format=json \
      > "results/fio-qd$qd.json" &&
  python3 -I fio_summary.py "results/fio-qd$qd.json" "fio qd=$qd"
done

step "baseline (MEM_MAX=6G, sem populate)";   MEM_MAX=6G ./run.sh baseline
step "baseline-populate (MEM_MAX=6G)";        MEM_MAX=6G ./run.sh baseline-populate
step "trace (MEM_MAX=6G)";                    MEM_MAX=6G ./run.sh trace
step "fifo (MEM_MAX=6G BUDGET_MB=2700)";      MEM_MAX=6G BUDGET_MB=2700 ./run.sh fifo
step "fifo (MEM_MAX=5G BUDGET_MB=1700)";      MEM_MAX=5G BUDGET_MB=1700 ./run.sh fifo

step "simulate.py"
TRACE=$(ls -td results/trace-*/ | head -1)trace.csv
python3 simulate.py "$TRACE" --budgets-mb 512,1024,1700,2700,4096 | tee results/simulate.txt
echo "fim: $(date +%T)"
