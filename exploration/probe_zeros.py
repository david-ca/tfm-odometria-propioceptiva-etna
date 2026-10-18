"""
probe_zeros.py -- Diagnostico: localiza filas/bloques todo-cero en imu.csv y
wheel_odometry.csv, y comprueba si algun bloque de ceros del wheel solapa con
un intervalo de TARGET valido (target_<RUN>.csv). Puro diagnostico, no escribe
ningun dato del dataset.

Interpretacion:
  - IMU todo-cero: ANOMALO (el acelerometro siempre mide gravedad ~9.81 en z).
    Si aparece, es dato invalido, no reposo.
  - WHEEL todo-cero: reposo legitimo (ruedas quietas). Lo relevante es DONDE:
    solo en el arranque (esperado) o bloques dispersos (posible hueco de sensor).
  - Cruce con target: si un bloque de ceros del wheel cae DENTRO de un intervalo
    de target usado, una ventana IMU+wheel ahi tendria el canal wheel muerto.

Uso:  python probe_zeros.py Run_A
"""
import sys, csv
from pathlib import Path
import numpy as np

if len(sys.argv) < 2:
    print("Uso: python probe_zeros.py <RUN>"); sys.exit(1)
RUN = sys.argv[1]
BASE = Path(r"C:\Users\Usuario\Desktop\TFM") / "extracted" / RUN


def load(p):
    with open(p) as f:
        r = csv.reader(f); header = next(r)
        data = np.array([[float(x) for x in row] for row in r])
    return header, data


def zero_blocks(t_s, allzero):
    """Devuelve lista de bloques contiguos de ceros como (t_ini, t_fin, n)."""
    blocks = []
    i = 0
    n = len(allzero)
    while i < n:
        if allzero[i]:
            j = i
            while j < n and allzero[j]:
                j += 1
            blocks.append((t_s[i], t_s[j - 1], j - i))
            i = j
        else:
            i += 1
    return blocks


print(f"=== PROBE ZEROS -- {RUN} ===\n")

# --- IMU ---
h_imu, imu = load(BASE / "imu.csv")
imu_t = imu[:, 0] / 1e9
imu_vals = imu[:, 1:]                      # gyro(3) + acc(3)
imu_allzero = np.all(imu_vals == 0.0, axis=1)
print(f"IMU: {len(imu)} filas, {imu_allzero.sum()} todo-cero "
      f"({imu_allzero.mean()*100:.2f}%)")
if imu_allzero.any():
    print("  *** ANOMALO: hay filas IMU todo-cero (acc_z deberia ser ~9.81). ***")
    for (a, b, n) in zero_blocks(imu_t, imu_allzero):
        print(f"    bloque: {a:.3f}..{b:.3f}  ({b-a:.1f} s, {n} filas)")
else:
    print("  OK: ninguna fila IMU todo-cero (sensor valido desde el inicio).")

# --- WHEEL ---
h_wo, wo = load(BASE / "wheel_odometry.csv")
wo_t = wo[:, 0] / 1e9
wo_vals = wo[:, 1:]
wo_allzero = np.all(wo_vals == 0.0, axis=1)
print(f"\nWHEEL: {len(wo)} filas, {wo_allzero.sum()} todo-cero "
      f"({wo_allzero.mean()*100:.2f}%)")
wo_blocks = zero_blocks(wo_t, wo_allzero)
print(f"  bloques de ceros contiguos: {len(wo_blocks)}")
for (a, b, n) in wo_blocks:
    pos = "ARRANQUE" if a <= wo_t[0] + 1e-6 else ("FINAL" if b >= wo_t[-1]-1e-6 else "INTERIOR")
    print(f"    {pos}: {a:.3f}..{b:.3f}  ({b-a:.1f} s, {n} filas)")

# --- Cruce de bloques INTERIORES del wheel con targets validos ---
tgt_path = BASE / f"target_{RUN}.csv"
if not tgt_path.exists():
    print(f"\n(no existe {tgt_path.name}: omito el cruce con target)")
else:
    _, tgt = load(tgt_path)
    tgt_start, tgt_end = tgt[:, 0], tgt[:, 1]
    n_tgt = len(tgt_start)
    interior = [(a, b, n) for (a, b, n) in wo_blocks
                if a > wo_t[0] + 1e-6 and b < wo_t[-1] - 1e-6]

    # umbral de duracion: bloques cortos = reposo/micro-parada (dato legitimo);
    # bloques largos = sensor probablemente caido (hueco real a tratar).
    DUR_THR = 1.0   # s

    short_blocks = [(a, b, n) for (a, b, n) in interior if (b - a) < DUR_THR]
    long_blocks  = [(a, b, n) for (a, b, n) in interior if (b - a) >= DUR_THR]

    # Para cada intervalo de target, fraccion de su 1 s cubierta por ceros wheel
    # (solapando TODOS los bloques interiores). Un intervalo esta "danado" si una
    # parte apreciable de su duracion cae sobre ceros.
    cover = np.zeros(n_tgt)
    cover_long = np.zeros(n_tgt)
    for (a, b, n) in interior:
        ov = np.clip(np.minimum(tgt_end, b) - np.maximum(tgt_start, a), 0, None)
        cover += ov
        if (b - a) >= DUR_THR:
            cover_long += ov
    dur = tgt_end - tgt_start
    frac = cover / dur
    frac_long = cover_long / dur

    print(f"\nCruce con target (umbral bloque largo = {DUR_THR}s):")
    print(f"  bloques interiores totales : {len(interior)}")
    print(f"    cortos (< {DUR_THR}s, reposo/micro-parada): {len(short_blocks)}")
    print(f"    largos (>= {DUR_THR}s, posible sensor caido): {len(long_blocks)}")
    print(f"  intervalos de target: {n_tgt}")
    for thr in (0.10, 0.25, 0.50, 0.90):
        print(f"    con >{thr*100:.0f}% de su duracion sobre ceros wheel: "
              f"{int(np.sum(frac > thr))}  (solo bloques largos: {int(np.sum(frac_long > thr))})")
    print(f"  intervalos TOTALMENTE limpios (0% ceros wheel): "
          f"{int(np.sum(frac == 0))} / {n_tgt} ({np.mean(frac==0)*100:.1f}%)")

    # --- Caracterizacion de la franja dudosa (frac > 25%): que estariamos ---
    # --- descartando? Columnas target: 0 t_start,1 t_end,2 seg,3 fwd,4 lat,5 up ---
    d_fwd_all, d_lat_all, d_up_all = tgt[:, 3], tgt[:, 4], tgt[:, 5]
    lat_ref = np.abs(d_lat_all).mean()   # d_lat medio de TODO el target (referencia)
    print(f"\n  Referencia: |d_lat| medio de todo el target = {lat_ref*100:.2f} cm")
    print(f"  Intervalos con frac>25% (franja a decidir 25-50-90):")
    idx = np.where(frac > 0.25)[0]
    if len(idx) == 0:
        print("    (ninguno)")
    else:
        print(f"    {'t_start':>14}  {'frac%':>6}  {'d_fwd':>7}  {'d_lat':>7}  {'d_up':>7}  slip?")
        for i in idx:
            slip = "SLIP-ALTO" if abs(d_lat_all[i]) > 2 * lat_ref else ""
            print(f"    {tgt[i,0]:14.3f}  {frac[i]*100:6.0f}  "
                  f"{d_fwd_all[i]*100:6.1f}c {d_lat_all[i]*100:6.1f}c {d_up_all[i]*100:6.1f}c  {slip}")
        n_slip = int(np.sum(np.abs(d_lat_all[idx]) > 2 * lat_ref))
        print(f"    -> de {len(idx)} intervalos dañados, {n_slip} con |d_lat| > 2x la media (slip alto)")
    if long_blocks:
        print(f"\n  bloques largos (>= {DUR_THR}s):")
        for (a, b, n) in long_blocks:
            hits = int(np.sum((tgt_start < b) & (tgt_end > a)))
            print(f"    {a:.3f}..{b:.3f}  ({b-a:.1f}s, {n} filas)  "
                  f"solapa {hits} intervalos de target")

print("\nHecho.")