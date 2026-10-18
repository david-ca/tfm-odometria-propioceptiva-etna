"""
inspect_gap.py — Localiza los huecos temporales en el ground truth DGPS.
Reporta todos los huecos por encima de un umbral, con la posición
antes y despues de cada uno, para entender si el hueco es una parada,
una perdida de senal, o una separacion entre travesias.
"""
from pathlib import Path

GT = Path(r"C:\Users\Usuario\Desktop\TFM\HiDrive-dgps\processed\dgps_aligned_nav.txt")
GAP_THRESHOLD = 1.0  # s; un hueco "normal" a 8 Hz es ~0.125 s. Reportamos > 1 s.

rows = []
with open(GT) as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vals = [float(p) for p in s.split()]
        rows.append(vals)

print(f"Filas: {len(rows)}")
print(f"Umbral de hueco reportado: > {GAP_THRESHOLD} s\n")

# dt normal (mediana de los intervalos) como referencia
dts = [rows[i][0] - rows[i-1][0] for i in range(1, len(rows))]
dts_sorted = sorted(dts)
median_dt = dts_sorted[len(dts_sorted)//2]
print(f"dt mediano entre filas: {median_dt:.4f} s  (~{1/median_dt:.2f} Hz)\n")

# Localiza y describe cada hueco por encima del umbral
gaps = []
for i in range(1, len(rows)):
    dt = rows[i][0] - rows[i-1][0]
    if dt > GAP_THRESHOLD:
        gaps.append((i, dt))

print(f"Nº de huecos > {GAP_THRESHOLD} s: {len(gaps)}\n")
print("=" * 70)
for rank, (i, dt) in enumerate(sorted(gaps, key=lambda g: -g[1]), 1):
    before = rows[i-1]
    after = rows[i]
    # distancia euclidea entre el punto antes y despues del hueco
    dx = after[1] - before[1]
    dy = after[2] - before[2]
    dz = after[3] - before[3]
    dist = (dx*dx + dy*dy + dz*dz) ** 0.5
    print(f"HUECO #{rank}  (índice de fila {i})")
    print(f"  Duración del hueco:  {dt:.3f} s  ({dt/60:.2f} min)")
    print(f"  t antes:  {before[0]:.3f}")
    print(f"  t después:{after[0]:.3f}")
    print(f"  pos antes:  x={before[1]:8.3f}  y={before[2]:8.3f}  z={before[3]:8.3f}")
    print(f"  pos después:x={after[1]:8.3f}  y={after[2]:8.3f}  z={after[3]:8.3f}")
    print(f"  Salto de posición a través del hueco: {dist:.3f} m")
    print("-" * 70)

# Contexto: cuántos puntos hay a cada lado del hueco mayor
if gaps:
    biggest_i = max(gaps, key=lambda g: g[1])[0]
    print(f"\nHueco mayor en índice {biggest_i}:")
    print(f"  Puntos ANTES del hueco: {biggest_i}")
    print(f"  Puntos DESPUÉS del hueco: {len(rows) - biggest_i}")
    t_span_before = rows[biggest_i-1][0] - rows[0][0]
    t_span_after = rows[-1][0] - rows[biggest_i][0]
    print(f"  Duración tramo 1: {t_span_before:.1f} s ({t_span_before/60:.1f} min)")
    print(f"  Duración tramo 2: {t_span_after:.1f} s ({t_span_after/60:.1f} min)")

print("=" * 70)
print("Hecho.")