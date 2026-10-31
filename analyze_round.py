#!/usr/bin/env python3
"""Resume rodadas do pager: tok/s, experts lidos/token na geração, MB/s de carga, despejos,
custo do callback e a divisão do tempo por token (carga / callback / resto).

Uso: analyze_round.py results/<rodada> [...]
A carga por token é exata no modo sync/parallel (handler serial: soma do load_us dos passos >= 1) e
estimada no async (bytes de experts por token ÷ MB/s de carga da rodada), porque lá as cargas se sobrepõem.
"""
import csv, re, sys

for d in sys.argv[1:]:
    log = open(f"{d}/llama.log").read()
    s = open(f"{d}/summary.txt").read()
    ev = re.search(r"\n[^\n]*?\beval time =\s+([\d.]+) ms /\s+(\d+) runs", log)
    gen_ms, runs = float(ev.group(1)), int(ev.group(2))
    ms_tok = gen_ms / runs
    pg = re.search(r"\[pager\] passos=.*", log).group(0)
    io_mode = re.search(r"io_modo=(\w+)", pg).group(1)
    rate = float(re.search(r"carga=[\d.]+ s \((\d+) MB/s\)", pg).group(1))
    desp = int(re.search(r"despejos=(\d+)", pg).group(1))
    rt = re.search(r"\[pager\] router=.*", log)
    cb = float(re.search(r"cb=([\d.]+) ms/token", rt.group(0)).group(1)) if rt and "ativo" in rt.group(0) else 0.0
    rows = [r for r in csv.DictReader(open(f"{d}/trace.csv")) if int(r["step"]) >= 1 and int(r["expert"]) >= 0]
    steps = len({r["step"] for r in rows}) or 1
    mb_tok = sum(int(r["bytes"]) for r in rows) / steps / 1e6
    if io_mode == "async":
        carga = mb_tok / rate * 1e3
        tag = "estimada"
    else:
        carga = sum(int(r["load_us"]) for r in rows) / steps / 1e3
        tag = "medida"
    resto = ms_tok - carga - cb
    print(f"{d}\n  io={io_mode} tok/s={1e3/ms_tok:.2f} ms/token={ms_tok:.0f} | experts/token(geração)={mb_tok:.1f} MB "
          f"cargas/token={len(rows)/steps:.1f} | MB/s carga={rate:.0f} | despejos={desp}\n"
          f"  tempo/token: carga={carga:.0f} ms ({tag}, {carga/ms_tok:.0%}) callback={cb:.1f} ms ({cb/ms_tok:.0%}) "
          f"resto/cálculo={resto:.0f} ms ({resto/ms_tok:.0%})")
    if rt:
        print("  " + rt.group(0))
