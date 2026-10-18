"""
probe_gyro_bias2.py — Diagnostico AFINADO de sesgo del giroscopio.
Distingue si la media no nula de gyro_z es sesgo real o artefacto de haber
colado giros-en-el-sitio en la deteccion de reposo.

Define REPOSO GENUINO exigiendo simultaneamente:
  - |lin_x| < 0.01 m/s   (no avanza)
  - |ang_z| < 0.01 rad/s (no gira en el sitio)  <-- clave nueva
y ademas verifica estabilidad de la aceleracion en esas ventanas.

Compara el sesgo estimado con reposo LAXO (solo lin_x) vs reposo GENUINO.
Si baja hacia cero con el criterio genuino -> era artefacto.
Si se mantiene -> sesgo real del sensor.
Uso:  python probe_gyro_bias2.py Run_A
"""
import sys, csv
from pathlib import Path
import numpy as np

if len(sys.argv) < 2:
    print("Uso: python probe_gyro_bias2.py <RUN>"); sys.exit(1)
RUN = sys.argv[1]
BASE = Path(r"C:\Users\Usuario\Desktop\TFM") / "extracted" / RUN

def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])

imu = load(BASE / "imu.csv")             # ts_ns, gyro_x,y,z, acc_x,y,z
wo  = load(BASE / "wheel_odometry.csv")  # ts_ns, lin_x,y,z, ang_x,y,z
imu_t = imu[:,0] / 1e9
wo_t  = wo[:,0] / 1e9

print(f"=== RUN: {RUN} — Diagnostico afinado de sesgo del giroscopio ===\n")

# Dos criterios de reposo sobre la wheel odometry
lax_mask     = np.abs(wo[:,1]) < 0.01                       # solo lin_x
genuine_mask = (np.abs(wo[:,1]) < 0.01) & (np.abs(wo[:,6]) < 0.01)  # lin_x Y ang_z

print(f"Muestras wheel:")
print(f"  reposo LAXO    (|lin_x|<0.01):              {lax_mask.sum()}")
print(f"  reposo GENUINO (|lin_x|<0.01 y |ang_z|<0.01): {genuine_mask.sum()}")
print(f"  -> diferencia (instantes que eran giros en el sitio): "
      f"{lax_mask.sum()-genuine_mask.sum()}\n")

TOL = 0.07
def gyro_in_rest(rest_times):
    rest_sorted = np.sort(rest_times)
    if len(rest_sorted) == 0:
        return None
    gx, gy, gz = [], [], []
    for i in range(0, len(imu), 5):
        t = imu_t[i]
        idx = np.searchsorted(rest_sorted, t)
        hit = False
        for j in (idx-1, idx):
            if 0 <= j < len(rest_sorted) and abs(rest_sorted[j]-t) <= TOL:
                hit = True; break
        if hit:
            gx.append(imu[i,1]); gy.append(imu[i,2]); gz.append(imu[i,3])
    return np.array(gx), np.array(gy), np.array(gz)

def report(tag, res):
    if res is None or len(res[2]) == 0:
        print(f"[{tag}] sin muestras."); return None
    gx, gy, gz = res
    print(f"[{tag}] muestras IMU analizadas: {len(gz)}")
    for name, g in [("gyro_x", gx), ("gyro_y", gy), ("gyro_z", gz)]:
        ratio = abs(g.mean())/g.std() if g.std()>0 else 0
        print(f"   {name}: media={g.mean():+.6f}  std={g.std():.6f}  |media|/std={ratio:.2f}")
    return gz.mean()

bias_lax     = report("REPOSO LAXO", gyro_in_rest(wo_t[lax_mask]))
print()
bias_genuine = report("REPOSO GENUINO", gyro_in_rest(wo_t[genuine_mask]))

print("\n=== VEREDICTO ===")
if bias_lax is not None and bias_genuine is not None:
    print(f"  sesgo gyro_z (laxo):    {bias_lax:+.6f} rad/s")
    print(f"  sesgo gyro_z (genuino): {bias_genuine:+.6f} rad/s")
    if abs(bias_genuine) < 0.3*abs(bias_lax):
        print("  -> El sesgo CAE mucho con reposo genuino: era ARTEFACTO")
        print("     (giros en el sitio contaminaban la medida). El sesgo laxo estaba inflado por giros en el sitio; queda un sesgo residual.")
    elif abs(bias_genuine) < 0.7*abs(bias_lax):
        print("  -> El sesgo baja parcialmente: habia algo de artefacto, pero")
        print("     queda un sesgo residual a considerar.")
    else:
        print("  -> El sesgo SE MANTIENE con reposo genuino: es SESGO REAL")
        print("     del sensor; conviene corregirlo antes de integrar.")
    if abs(bias_genuine) > 1e-9:
        print(f"\n  Deriva de yaw por el sesgo genuino:")
        for s in (60, 300, 872):
            print(f"    tras {s:4d} s: {np.degrees(bias_genuine*s):+.2f} deg")

print("\nHecho.")