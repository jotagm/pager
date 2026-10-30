#!/usr/bin/env python3
"""Resume um JSON do fio (randread): MB/s, IOPS e latência p50/p99."""
import json, sys

r = json.load(open(sys.argv[1]))["jobs"][0]["read"]
lat = r["clat_ns"]["percentile"]
print(f"{sys.argv[2]}: {r['bw_bytes']/1e6:.0f} MB/s, {r['iops']:.0f} IOPS, "
      f"lat p50={lat['50.000000']/1e6:.2f} ms p99={lat['99.000000']/1e6:.2f} ms")
