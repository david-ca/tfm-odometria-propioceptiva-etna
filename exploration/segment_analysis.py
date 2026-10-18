"""
segment_analysis.py — Analiza como queda segmentada la trayectoria de ground
truth bajo varios umbrales de hueco. Para cada umbral reporta: nº de tramos
continuos resultantes, y la distribucion de sus longitudes (en segundos y en
nº de muestras). No decide el umbral: da la tabla para decidirlo con criterio.
Uso:  python segment_analysis.py Run_A
      python segment_analysis.py Run_B
"""
import sys
import csv
from pathlib import Path

if len(sys.argv) < 2:
    print("Uso: python segment_analysis.py <NOMBRE_DEL_RUN>")
    sys.exit(1)

RUN = sys.argv[1]
GT = Path(r"C:\Users\Usuario\Desktop\TFM") / "extracted" / RUN / "ground_truth.csv"

# Umbrales candidatos a evaluar (segundos)
THRESHOLDS = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]

# Cargar ground truth (timestamp en segundos)
rows = []
with open(GT) as f:
    r = csv.reader(f)
    next(r)  # cabecera
    for row in r:
        rows.append([float(x) for x in row])

n = len(rows)
t = [row[0] for row in rows]
dts = [t[i]-t[i-1] for i in range(1, n)]

print(f"=== RUN: {RUN} — Segmentacion por umbral ===")
print(f"Ground truth: {n} muestras, {t[-1]-t[0]:.1f} s totales, ~10 Hz\n")

def segment(threshold):
    """Devuelve lista de tramos como (idx_ini, idx_fin) [fin exclusivo].
    Corta donde dt > threshold."""
    segs = []
    start = 0
    for i in range(1, n):
        if dts[i-1] > threshold:
            segs.append((start, i))
            start = i
    segs.append((start, n))
    return segs

print(f"{'umbral(s)':>9} | {'nº tramos':>9} | {'muestras útiles':>15} | "
      f"{'tramo max(s)':>12} | {'tramos >60s':>11} | {'tramos >300s':>12}")
print("-" * 90)

for th in THRESHOLDS:
    segs = segment(th)
    lengths_s = [t[e-1]-t[s] for (s, e) in segs]
    lengths_n = [e-s for (s, e) in segs]
    # "muestras utiles" = suma de muestras en tramos de al menos 10 s
    # (tramos muy cortos no serviran para ventanas; 10 s es un proxy provisional)
    useful = sum(ln for ln, ls in zip(lengths_n, lengths_s) if ls >= 10.0)
    n_over_60 = sum(1 for ls in lengths_s if ls >= 60.0)
    n_over_300 = sum(1 for ls in lengths_s if ls >= 300.0)
    print(f"{th:>9.1f} | {len(segs):>9} | {useful:>15} | "
          f"{max(lengths_s):>12.1f} | {n_over_60:>11} | {n_over_300:>12}")

# Detalle para un umbral de referencia (2 s) — ver los tramos reales
print("\n--- Detalle con umbral = 2.0 s (referencia) ---")
segs = segment(2.0)
lengths_s = sorted(((t[e-1]-t[s], e-s) for (s, e) in segs), reverse=True)
print(f"  Nº de tramos: {len(segs)}")
print(f"  Los 10 tramos más largos (segundos / muestras):")
for ls, ln in lengths_s[:10]:
    print(f"    {ls:8.1f} s   {ln:7d} muestras")
if len(lengths_s) > 10:
    tiny = sum(1 for ls, _ in lengths_s if ls < 10.0)
    print(f"  Tramos < 10 s (probablemente inservibles): {tiny}")

print("\nHecho.")