# pager — paginação de experts MoE em user space

Experimento: rodar o **Qwen3.6-35B-A3B** (MoE, 256 experts por camada, 8 roteados por token;
UD-Q4_K_M com 22,13 GB) num lab com **7,6 GiB de RAM**, paginando os pesos do NVMe. A pergunta é se uma
política de paginação que conhece a estrutura (camada, expert) lê menos do disco que o kernel.

O llama.cpp faz todo o cálculo. Só a camada de memória muda:

- **`libpager.so`** (`LD_PRELOAD`): intercepta o `mmap()` do `.gguf`, devolve uma região anônima registrada no
  `userfaultfd` e, a cada fault, lê do disco (O_DIRECT) o expert inteiro (gate+up+down) e o insere com
  `UFFDIO_COPY`. Os tensores fixos (atenção, embedding, expert compartilhado) ficam residentes.
- **`libnopopulate.so`** (`LD_PRELOAD`): baseline justo para o kernel. Tira o `MAP_POPULATE` e ignora o
  `WILLNEED` que o llama.cpp aplica no carregamento; com `NOPOP_RANDOM=1`, aplica também `MADV_RANDOM`.
- **`simulate.py`**: simula LRU, LFU, Markov (prefetch) e Belady sobre um trace de acessos.

## Compilar e testar

```bash
make && make test      # testa trace/fifo/fifo sem O_DIRECT com PAGER_IO=sync e parallel (GGUF falso)
```

O llama.cpp (CPU) é compilado pelo `setup.sh`. Testado no commit `5047748`.

## Rodar

```bash
export MODEL=~/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
MEM_MAX=6G NTOK=64 ./run.sh <modo>
```

**Router (v4).** O `libpager` intercepta `llama_init_from_model` e encadeia um `cb_eval` (preservando o que já
existir) que observa `ffn_moe_topk-<L>` (L < 40): a cada camada, marca os experts escolhidos como usados (LRU) e
despacha a carga dos que faltam antes do `mul_mat_id`. O passo (token) passa a vir do índice da camada. Ligado por
padrão só em `PAGER_MODE=lru` (`PAGER_ROUTER=0/1` muda). O resumo `[pager] router=` traz o custo do callback por
token de geração, decomposto em `ask`, `lock` e `trabalho` (despejo + despacho). O `libpager` compila contra os
headers do llama.cpp (`LLAMA_DIR`, padrão `~/llama.cpp`), do mesmo commit que vai rodar.

**Espera e prefetch (v5).** Uma thread leitora carimba a chegada de cada fault; ele termina quando a thread do
llama é acordada. `[pager] espera=` é o tempo de relógio por token com pelo menos uma thread parada, então
tempo/token = espera + callback + cálculo, tudo medido. No modo `lru` com router e `PAGER_IO=async`, o callback da
camada L aprende as transições L−1 → L (pares do mesmo token) e despacha os top-`PAGER_PREFETCH_K` (8) previstos
para L+1 numa fila de prioridade baixa; um expert em prefetch que passa a ser pedido sobe pra fila de demanda, e o
prefetch nunca despeja um expert usado no token atual. `PAGER_PREFETCH=0` reproduz o v4 no mesmo binário.
`[pager] prefetch=` traz emitidos, úteis e desperdiçados por token.

Modos de I/O do pager: `sync` (v1, o handler lê e copia cada fatia em sequência), `parallel` (v2, as fatias
gate/up/down de um expert são lidas em paralelo por um pool de threads, leitura sobreposta à cópia) e `async`
(v3, o handler despacha cada fault pro pool e volta a ler o uffd: várias cargas de experts em voo, fatias em
blocos de `PAGER_BLOCK_KB`; um segundo fault num expert em carga espera o bloco dele, sem recarregar).

| modo | o que faz |
|---|---|
| `baseline` | mmap normal via `libnopopulate`: o kernel decide (sem populate/WILLNEED na carga) |
| `baseline-populate` | comportamento padrão do llama.cpp (só para registro) |
| `baseline-random` | `baseline` + `MADV_RANDOM` (sem o read-around de 4 MB do kernel) |
| `trace` | `libpager` gravando o trace completo de acessos (`results/trace-*/trace.csv`) |
| `fifo` | `libpager` com orçamento fixo de experts (`BUDGET_MB`) e despejo FIFO |
| `lru` | `libpager` com despejo LRU real: o "bit de acesso" vem do router do modelo (v4, use `PAGER_IO=async`) |

Variáveis: `MEM_MAX` (teto do cgroup), `BUDGET_MB`, `NTOK`, `PAGER_IO=sync|parallel|async`, `PAGER_IO_THREADS`,
`PAGER_BLOCK_KB` (async),
`TIMEOUT` (SIGINT; o llama ainda imprime as estatísticas). Cada rodada parte do page cache frio
(`posix_fadvise` + `fincore`) e grava em `results/<modo>-<data>/` o `summary.txt`, o `llama.log`, o
`time.txt` e o `memory.events`/`memory.peak` do scope. `run_all.sh` roda a sequência da rodada 1.

## Resultados (i7-8565U, NVMe 128 GB, `MEM_MAX=6G`, `NTOK=64`, prompt de 97 tokens)

Teto do disco (`fio`, randread 2M, O_DIRECT): **922 MB/s** com iodepth=1, **1.268 MB/s** com iodepth=16.

| rodada | tok/s (geração) | GB lidos | GB/token (total/64) | experts lidos/token (geração) | MB/s de I/O | KB por major fault |
|---|---|---|---|---|---|---|
| baseline (readahead 4 MB) | 0,06 | 576,0 | 9,00 | — | 545¹ | 2.706 |
| baseline-populate | 0,06 | 596,4 | 9,32 | — | 559¹ | — |
| baseline-random (`MADV_RANDOM`) | 0,31 | 35,0 | 0,55 | — | 121¹ | 4,1 |
| trace (orçamento 2700 MB) | 0,71 | 49,7 | 0,78 | 615 MB | — | — |
| fifo v1 sync, 2700 MB | 1,89–1,90 | 22,0 | 0,34 | 180,5 MB | 533² | — |
| fifo v1 sync, 1700 MB (`MEM_MAX=5G`) | 1,52 | 26,1 | 0,41 | 246,3 MB | — | — |
| **fifo v2 parallel, 2700 MB** | **2,34** | 22,0 | 0,34 | 180,5 MB | **688**² | — |

¹ bytes lidos do disco ÷ tempo total. ² tempo de relógio do pager carregando experts (leitura + cópia).

Simulador sobre o trace (experts lidos do disco por token, só geração):

| orçamento | lru | lfu | markov (demanda + prefetch) | belady |
|---|---|---|---|---|
| 1024 MB | 290,3 | 292,7 | 190 + 167 = 357 | 176,6 |
| 1700 MB | 214,5 | 220,5 | 164 + 67 = 231 | 138,2 |
| 2700 MB | 161,2 | 159,0 | 142 + 24 = 166 | 115,2 |

### Conclusões até aqui

- O pager v1 é **31× mais rápido** que o kernel com mmap, mas quase tudo vem do **readahead de 4 MB**
  (`read_ahead_kb=4096` no NVMe/btrfs), que lê experts vizinhos inúteis em cada fault. Sem ele
  (`baseline-random`), o kernel lê 16× menos e fica 5× mais rápido; o pager ainda é **6–7,5× mais rápido**,
  porque faz ~10 mil leituras de ~1,8 MB em vez de 8,5 milhões de faults de 4 KB.
- O FIFO real fica 12–15% acima do LRU simulado: o pager só vê misses e despeja experts quentes.
- O Markov reduz a demanda (42–88% do caminho LRU→Belady), mas lê mais disco no total; o ganho depende de o
  prefetch se esconder atrás do cálculo.
- O v2 (fatias lidas em paralelo, leitura sobreposta à cópia) dá **+23% de tok/s** sobre o v1; o I/O ainda é
  ~60% do tempo de cada token.

Os detalhes de cada rodada estão em `results/rodada-*.log` e `results/rodada2-*.log`.
