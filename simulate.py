#!/usr/bin/env python3
"""Simulador offline de políticas de paginação de experts, a partir do trace
gerado pelo libpager em PAGER_MODE=trace.

Políticas:
  lru      despeja o menos recente (referência ~ o que o kernel tenta fazer)
  lfu      despeja o menos usado (segura experts "quentes")
  markov   LRU + prefetch: aprende online "expert e na camada L -> experts na
           próxima camada" e pré-carrega o top-K antes da próxima camada
  belady   ótimo teórico (MIN): despeja quem vai demorar mais pra ser usado.
           Precisa conhecer o futuro -> é o teto, não é implementável.

Métricas por token:
  demanda  bytes carregados no momento do acesso (a CPU espera -> stall)
  prefetch bytes carregados antecipadamente (podem ficar escondidos atrás do cálculo)
  disco    demanda + prefetch (o que de fato sai do SSD)
"""
import argparse, csv, heapq
from collections import OrderedDict, Counter, defaultdict

def load(path, skip):
    steps = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if int(r["expert"]) < 0: continue
        steps[int(r["step"])].append((int(r["layer"]), int(r["expert"]), int(r["bytes"])))
    acc = []
    for s in sorted(steps)[skip:]:
        for l, e, b in sorted(steps[s], key=lambda x: x[0]):   # ordem por camada dentro do token
            acc.append((s, l, (l, e), b))
    return acc

def sim_lru(acc, budget, lfu=False):
    cache, used, freq, dem = OrderedDict(), 0, Counter(), 0
    hits = 0
    for _, _, k, b in acc:
        freq[k] += 1
        if k in cache:
            hits += 1; cache.move_to_end(k); continue
        dem += b
        while used + b > budget and cache:
            v = min(cache, key=lambda x: freq[x]) if lfu else next(iter(cache))
            used -= cache.pop(v)
        cache[k] = b; used += b
    return dict(hits=hits, dem=dem, pre=0, waste=0)

def sim_belady(acc, budget):
    nxt, last = [0] * len(acc), {}
    for i in range(len(acc) - 1, -1, -1):
        k = acc[i][2]; nxt[i] = last.get(k, 10**12); last[k] = i
    cache, heap, used, dem, hits = {}, [], 0, 0, 0
    for i, (_, _, k, b) in enumerate(acc):
        if k in cache:
            hits += 1
        else:
            dem += b
            while used + b > budget and cache:
                nu, v = heapq.heappop(heap)
                if v in cache and cache[v][1] == -nu:
                    used -= cache.pop(v)[0]
            used += b
        cache[k] = (b, nxt[i])
        heapq.heappush(heap, (-nxt[i], k))
    return dict(hits=hits, dem=dem, pre=0, waste=0)

def sim_markov(acc, budget, topk):
    trans = defaultdict(Counter)                 # (l,e) -> Counter((l2,e2))
    cache, used, dem, pre, waste, hits = OrderedDict(), 0, 0, 0, 0, 0
    pref = set()                                 # pré-carregados ainda não usados
    sizes = {}
    def insert(k, b):
        nonlocal used, waste
        while used + b > budget and cache:
            v, vb = cache.popitem(last=False); used -= vb
            if v in pref: pref.discard(v); waste += vb
        cache[k] = b; used += b
    i = 0
    while i < len(acc):
        s, l = acc[i][0], acc[i][1]
        j = i
        while j < len(acc) and acc[j][0] == s and acc[j][1] == l: j += 1
        cur = [a[2] for a in acc[i:j]]
        for _, _, k, b in acc[i:j]:
            sizes[k] = b
            if k in cache:
                hits += 1; cache.move_to_end(k); pref.discard(k)
            else:
                dem += b; insert(k, b)
        if j < len(acc) and acc[j][0] == s:     # existe próxima camada neste token
            nl = acc[j][1]
            k2 = j
            while k2 < len(acc) and acc[k2][0] == s and acc[k2][1] == nl: k2 += 1
            for _, _, kk, _ in acc[j:k2]:       # aprende (online: só depois de ver)
                for c in cur: trans[c][kk] += 1
            votes = Counter()
            for c in cur: votes.update(trans[c])
            for kk, _ in votes.most_common(topk):
                if kk not in cache and kk in sizes:
                    pre += sizes[kk]; insert(kk, sizes[kk]); pref.add(kk)
        i = j
    return dict(hits=hits, dem=dem, pre=pre, waste=waste)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budgets-mb", default="1024,2048,3072,4096")
    ap.add_argument("--skip", type=int, default=1, help="passos iniciais a ignorar (prompt)")
    ap.add_argument("--topk", type=int, default=8)
    a = ap.parse_args()
    acc = load(a.trace, a.skip)
    if not acc: print("trace vazio"); return
    ntok = len({x[0] for x in acc})
    per_tok = sum(x[3] for x in acc) / ntok
    nexp = len({x[2] for x in acc})
    print(f"{len(acc)} acessos, {ntok} tokens, {nexp} experts distintos, "
          f"{per_tok/1e6:.1f} MB de experts por token\n")
    print(f"{'orçamento':>10} {'política':>8} {'acerto':>7} {'demanda/tok':>12} {'prefetch/tok':>13} "
          f"{'desperdício':>11} {'disco/tok':>10}")
    for bm in [float(x) for x in a.budgets_mb.split(",")]:
        budget = int(bm * 2**20)
        for name, fn in [("lru", lambda: sim_lru(acc, budget)),
                         ("lfu", lambda: sim_lru(acc, budget, lfu=True)),
                         ("markov", lambda: sim_markov(acc, budget, a.topk)),
                         ("belady", lambda: sim_belady(acc, budget))]:
            r = fn()
            print(f"{bm:>8g}MB {name:>8} {r['hits']/len(acc):>6.1%} {r['dem']/ntok/1e6:>10.1f}MB "
                  f"{r['pre']/ntok/1e6:>11.1f}MB {r['waste']/ntok/1e6:>9.1f}MB "
                  f"{(r['dem']+r['pre'])/ntok/1e6:>8.1f}MB")
        print()

if __name__ == "__main__":
    main()
