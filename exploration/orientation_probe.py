"""
orientation_probe.py — Verifica el filtro Madgwick (libreria ahrs) sobre un
tramo continuo, comprobando que la orientacion resultante tiene sentido fisico
ANTES de usarla para construir el target.

Comprobaciones:
  1. Pitch/roll en reposo ~ 0 y estables (rover mayormente horizontal).
  2. Pitch coherente con la pendiente real: al bajar en z (segun GPS), el rover
     deberia cabecear; se correlaciona pitch integrado vs pendiente del GPS.
  3. Yaw evoluciona de forma suave (sin saltos).
  4. Exporta la orientacion (roll,pitch,yaw) del tramo a CSV para graficar.

Uso:  python orientation_probe.py Run_A
"""
import sys, csv
from pathlib import Path
import numpy as np

try:
    from ahrs.filters import Madgwick
except ImportError:
    print("ERROR: falta 'ahrs'.  Instala con:  pip install ahrs")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Uso: python orientation_probe.py <RUN>"); sys.exit(1)
RUN = sys.argv[1]
BASE = Path(r"C:\Users\Usuario\Desktop\TFM") / "extracted" / RUN

def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])

imu = load(BASE / "imu.csv")             # ts_ns, gyro_x,y,z, acc_x,y,z
gt  = load(BASE / "ground_truth.csv")    # ts_s, x,y,z
imu_t = imu[:,0] / 1e9
gt_t  = gt[:,0]

# --- Elegir un tramo continuo del GT (primero >= 60 s) ---
def find_segment(gt, min_dur=60.0, max_gap=2.0):
    start = 0
    for i in range(1, len(gt)):
        if gt[i,0]-gt[i-1,0] > max_gap:
            if gt[i-1,0]-gt[start,0] >= min_dur:
                return start, i
            start = i
    return start, len(gt)

s, e = find_segment(gt)
t0, t1 = gt_t[s], gt_t[e-1]
print(f"=== RUN: {RUN} — Verificacion del filtro Madgwick ===")
print(f"Tramo: [{t0:.1f}, {t1:.1f}]  ({t1-t0:.1f} s)\n")

# IMU dentro del tramo
mask = (imu_t >= t0) & (imu_t <= t1)
gyr = imu[mask, 1:4]      # rad/s (ya en rad/s, sin conversion)
acc = imu[mask, 4:7]      # m/s2
imu_seg_t = imu_t[mask]
n = len(gyr)
print(f"Muestras IMU en el tramo: {n}")

# Frecuencia real (para Dt del filtro)
dt_med = np.median(np.diff(imu_seg_t))
freq = 1.0/dt_med
print(f"Frecuencia IMU: {freq:.2f} Hz (Dt={dt_med*1000:.2f} ms)\n")

# --- Correr Madgwick (IMU: giro + acel, sin magnetometro) ---
print("Corriendo Madgwick (updateIMU)... esto puede tardar (Python puro)")
madgwick = Madgwick(frequency=freq)   # gain por defecto (0.033 IMU)
Q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
for t in range(1, n):
    Q[t] = madgwick.updateIMU(Q[t-1], gyr=gyr[t], acc=acc[t])

# --- Convertir cuaterniones [w,x,y,z] a roll/pitch/yaw (rad) ---
def quat_to_euler(Q):
    w, x, y, z = Q[:,0], Q[:,1], Q[:,2], Q[:,3]
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

roll, pitch, yaw = quat_to_euler(Q)

# --- Comprobacion 1: pitch/roll al inicio (primeros ~5s, rover suele arrancar quieto) ---
n5 = int(5*freq)
print("--- 1. Pitch/roll al inicio del tramo (primeros ~5 s) ---")
print(f"  roll:  media={roll[:n5].mean():+.2f} deg  std={roll[:n5].std():.2f}")
print(f"  pitch: media={pitch[:n5].mean():+.2f} deg  std={pitch[:n5].std():.2f}")

# --- Comprobacion 2: pitch vs pendiente real del GPS ---
# Pendiente del GPS: dz/distancia_horizontal a lo largo del tramo
gps = gt[s:e, 1:4]
gps_ts = gt_t[s:e]
# remuestrear pitch a los instantes del GPS
pitch_at_gps = np.interp(gps_ts, imu_seg_t, pitch)
dz = np.gradient(gps[:,2])
dh = np.gradient(np.sqrt(gps[:,0]**2 + gps[:,1]**2))  # aprox
slope_deg = np.degrees(np.arctan2(dz, np.abs(dh)+1e-9))
# correlacion (donde hay movimiento)
valid = np.abs(dh) > 1e-3
if valid.sum() > 10:
    corr = np.corrcoef(pitch_at_gps[valid], slope_deg[valid])[0,1]
else:
    corr = float('nan')
print("\n--- 2. Pitch (filtro) vs pendiente real (GPS) ---")
print(f"  rango pitch filtro: [{pitch.min():.1f}, {pitch.max():.1f}] deg")
print(f"  rango pendiente GPS: [{slope_deg[valid].min():.1f}, {slope_deg[valid].max():.1f}] deg")
print(f"  correlacion pitch<->pendiente: {corr:.3f}")
print("  (correlacion notable => el filtro capta las cuestas reales)")

# --- Comprobacion 3: suavidad del yaw (sin saltos bruscos) ---
yaw_unwrap = np.degrees(np.unwrap(np.radians(yaw)))
dyaw = np.abs(np.diff(yaw_unwrap))
print("\n--- 3. Suavidad del yaw ---")
print(f"  cambio de yaw entre muestras: media={dyaw.mean():.4f} deg  max={dyaw.max():.2f} deg")
print(f"  yaw total recorrido: {yaw_unwrap[-1]-yaw_unwrap[0]:+.1f} deg")

# --- Exportar para graficar ---
out = BASE / "orientation_probe.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t","roll_deg","pitch_deg","yaw_deg"])
    for i in range(n):
        w.writerow([imu_seg_t[i], roll[i], pitch[i], yaw_unwrap[i]])
print(f"\nOrientacion exportada a: {out.name}")

print("\n=== INTERPRETACION ===")
print("  - roll/pitch iniciales pequenos y estables => referencia de gravedad OK")
print("  - correlacion pitch<->pendiente alta => el filtro capta el relieve real")
print("  - yaw suave => integracion del giro coherente (deriva lenta esperada)")
print("\nHecho.")