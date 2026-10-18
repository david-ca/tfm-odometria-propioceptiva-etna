"""
probe_wheel_channels.py -- Verifica que canales de wheel_odometry.csv tienen
SENAL REAL y cuales son constantes (idealmente cero), en cada run. Decide cuantos
canales entran en la config IMU+wheel: incluir canales constantes solo anade
dimensiones muertas.

Mira cada canal (lin_x, lin_y, lin_z, ang_x, ang_y, ang_z) EXCLUYENDO las filas
todo-cero (reposo), para no confundir "sensor en reposo" con "canal muerto".

Uso:  python probe_wheel_channels.py Run_A
"""
import sys, csv
from pathlib import Path
import numpy as np

if len(sys.argv) < 2:
    print("Uso: python probe_wheel_channels.py <RUN>"); sys.exit(1)
RUN = sys.argv[1]
BASE = Path(r"C:\Users\Usuario\Desktop\TFM") / "extracted" / RUN

with open(BASE / "wheel_odometry.csv") as f:
    r = csv.reader(f); header = next(r)
    data = np.array([[float(x) for x in row] for row in r])

names = header[1:]           # lin_x, lin_y, lin_z, ang_x, ang_y, ang_z
vals = data[:, 1:]

# excluir filas todo-cero (reposo) para no sesgar
allzero = np.all(vals == 0.0, axis=1)
active = vals[~allzero]

print(f"=== WHEEL CHANNELS -- {RUN} ===")
print(f"  filas totales: {len(vals)}   activas (no todo-cero): {len(active)}\n")
print(f"  {'canal':>6}  {'min':>12}  {'max':>12}  {'std':>12}  {'n!=0':>8}  veredicto")
for i, nm in enumerate(names):
    col = vals[:, i]
    col_active = active[:, i]
    nnz = int(np.sum(col != 0.0))
    std = col_active.std() if len(col_active) else 0.0
    mn, mx = col.min(), col.max()
    if nnz == 0:
        verdict = "CONSTANTE CERO (excluir)"
    elif std < 1e-9:
        verdict = "CONSTANTE (excluir)"
    else:
        verdict = "SENAL REAL (incluir)"
    print(f"  {nm:>6}  {mn:12.6f}  {mx:12.6f}  {std:12.6f}  {nnz:8d}  {verdict}")

print("\nHecho.")