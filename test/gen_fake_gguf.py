#!/usr/bin/env python3
"""Gera um GGUF v3 falso com tensores de experts 3D (fatias NÃO alinhadas a página)
e um arquivo com o layout dos experts, pra testar o libpager sem o modelo real."""
import struct, sys

out = sys.argv[1] if len(sys.argv) > 1 else "fake.gguf"
LAYERS, EXPERTS, NE0, NE1 = 4, 64, 50, 30          # fatia = 50*30*4 = 6000 bytes
ALIGN = 32

def s(x):
    b = x.encode(); return struct.pack("<Q", len(b)) + b

tensors = [("token_embd.weight", [NE0, 1000])]
for l in range(LAYERS):
    tensors.append((f"blk.{l}.attn_q.weight", [NE0, NE0]))
    for p in ("gate", "up", "down"):
        tensors.append((f"blk.{l}.ffn_{p}_exps.weight", [NE0, NE1, EXPERTS]))
tensors.append(("output.weight", [NE0, 1000]))

kv = b""
kv += s("general.architecture") + struct.pack("<I", 8) + s("fake")
kv += s("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", ALIGN)
kv += s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 3) + s("a") + s("bb") + s("ccc")
kv += s("fake.scores") + struct.pack("<I", 9) + struct.pack("<IQ", 6, 4) + struct.pack("<4f", 1, 2, 3, 4)

infos, off, layout = b"", 0, []
for name, ne in tensors:
    n = 4
    for d in ne: n *= d
    infos += s(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", d) for d in ne)
    infos += struct.pack("<IQ", 0, off)                # tipo F32, offset
    layout.append((name, ne, off, n))
    off = (off + n + ALIGN - 1) // ALIGN * ALIGN

hdr = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 4) + kv + infos
data_start = (len(hdr) + ALIGN - 1) // ALIGN * ALIGN
total = data_start + off

def val(i): return ((i * 2654435761) >> 13) & 0xFF

buf = bytearray(total)
buf[:len(hdr)] = hdr
for i in range(data_start, total):
    buf[i] = val(i)
open(out, "wb").write(buf)

with open(out + ".layout", "w") as f:
    for name, ne, o, n in layout:
        if len(ne) == 3:
            l = int(name.split(".")[1]); sl = n // ne[2]
            for e in range(ne[2]):
                f.write(f"{l} {e} {data_start + o + e * sl} {sl}\n")
print(f"{out}: {total} bytes, dados em {data_start}")
