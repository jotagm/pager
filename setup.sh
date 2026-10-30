#!/usr/bin/env bash
# Checa o lab, compila o llama.cpp (só CPU) e o libpager, e roda o teste com GGUF falso.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

echo "== máquina"
uname -r
free -h | head -2
lsblk -d -o NAME,ROTA,TRAN,SIZE,MODEL
df -h "$HOME" | tail -1
echo "unprivileged_userfaultfd=$(cat /proc/sys/vm/unprivileged_userfaultfd 2>/dev/null || echo ?)"
ls -l /dev/userfaultfd 2>/dev/null || true
lscpu | grep -E "Model name|^CPU\(s\)"; grep -qw avx2 /proc/cpuinfo && echo "avx2: sim"

echo "== dependências"
sudo dnf install -y gcc gcc-c++ make cmake git python3 time perf 2>/dev/null || echo "instale manualmente: gcc gcc-c++ make cmake git python3 time"

echo "== libpager + teste"
cd "$HERE" && make clean >/dev/null && make && make test

echo "== llama.cpp (CPU)"
if [ ! -d "$HOME/llama.cpp" ]; then git clone --depth 1 https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"; fi
cd "$HOME/llama.cpp"
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --config Release
ls build/bin | grep -E "llama-(completion|cli|bench|simple)$" || true
