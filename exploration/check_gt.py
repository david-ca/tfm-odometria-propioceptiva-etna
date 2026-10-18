"""
check_gt.py — Caracteriza el ground truth DGPS de un run y localiza sus huecos.
Uso:  python check_gt.py Run_A
      python check_gt.py Run_B
No asume nº de columnas ni formato mas alla de: col 1 = timestamp epoch (s).
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Uso: python check_gt.py <NOMBRE_DEL_RUN>   (p.ej. Run_A o Run_B)")
    sys.exit(1)

RUN = sys.argv[1]
BASE = Path(r"C:\Users\Usuario\Desktop\TFM") / RUN / "HiDrive-dgps" / "processed"
GT = BASE / "dgps_aligned_nav.txt"
GAP_THRESHOLD = 1.0  # s

print(f"=== RUN: {RUN} ===")
print(f"Fichero: {GT}")
if not GT.exists():
    print("  NO EXISTE. Revisa la ruta/estructura.")
    sys.exit(1)

rows = []
ncols_set = set()
with open(GT) as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vals = [float(p) for p in s.split()]
        ncols_set.add(len(vals))
        rows.append(vals)

print(f"\n--- Caracterización general ---")
print(f"  Filas de datos:       {len(rows)}")
print(f"  Nº de columnas visto: {sorted(ncols_set)}")
if not rows:
    sys.exit(0)

t0, t1 = rows[0][0], rows[-1][0]
dur = t1 - t0
print(f"  Primer timestamp:     {t0:.6f}")
print(f"  Último timestamp:     {t1:.6f}")
print(f"  Duración total:       {dur:.2f} s  ({dur/60:.2f} min)")

# dt mediano (cadencia real, robusta a huecos)
dts = sorted(rows[i][0] - rows[i-1][0] for i in range(1, len(rows)))
median_dt = dts[len(dts)//2]
print(f"  dt mediano:           {median_dt:.4f} s  (~{1/median_dt:.2f} Hz)")

# Rango espacial
ncols = min(ncols_set)
if ncols >= 4:
    for ci, label in [(1, "x"), (2, "y"), (3, "z")]:
        col = [r[ci] for r in rows]
        print(f"  {label}: min={min(col):8.3f}  max={max(col):8.3f}  rango={max(col)-min(col):8.3f} m")

# Longitud de trayectoria (distancia acumulada punto a punto, saltando huecos grandes)
if ncols >= 4:
    total_len = 0.0
    for i in range(1, len(rows)):
        dt = rows[i][0] - rows[i-1][0]
        if dt > GAP_THRESHOLD:
            continue  # no acumular a traves de un hueco
        dx = rows[i][1]-rows[i-1][1]
        dy = rows[i][2]-rows[i-1][2]
        dz = rows[i][3]-rows[i-1][3]
        total_len += (dx*dx+dy*dy+dz*dz)**0.5
    print(f"  Longitud de trayectoria (sin cruzar huecos): {total_len:.1f} m")

# ¿Columnas 5-8 idénticamente cero?
if ncols >= 8:
    allzero = all(r[k] == 0.0 for r in rows for k in range(4, 8))
    print(f"  Columnas 5-8 idénticamente cero: {allzero}")

print(f"\n--- Huecos temporales (> {GAP_THRESHOLD} s) ---")
gaps = []
for i in range(1, len(rows)):
    dt = rows[i][0] - rows[i-1][0]
    if dt > GAP_THRESHOLD:
        dx = rows[i][1]-rows[i-1][1]
        dy = rows[i][2]-rows[i-1][2]
        dz = rows[i][3]-rows[i-1][3]
        dist = (dx*dx+dy*dy+dz*dz)**0.5
        gaps.append((i, dt, dist))

print(f"  Nº de huecos > {GAP_THRESHOLD} s: {len(gaps)}")
if gaps:
    print(f"  {'dur(s)':>8}  {'salto(m)':>9}  {'t_inicio':>16}")
    for i, dt, dist in sorted(gaps, key=lambda g: -g[1]):
        print(f"  {dt:8.2f}  {dist:9.3f}  {rows[i-1][0]:16.3f}")
    # tramos que definiria el hueco mayor
    bi = max(gaps, key=lambda g: g[1])[0]
    print(f"\n  Hueco mayor parte la trayectoria en:")
    print(f"    Tramo 1: {bi} puntos, {rows[bi-1][0]-rows[0][0]:.1f} s")
    print(f"    Tramo 2: {len(rows)-bi} puntos, {rows[-1][0]-rows[bi][0]:.1f} s")

print("\nHecho.")