"""
analyze_slip.py -- Analiza si el MODELO captura el deslizamiento lateral (d_lat)
frente al BASELINE, que estructuralmente predice ~0. El MAE medio no distingue
'predecir 0 siempre' de 'capturar la senal'; la CORRELACION si.

Compara, sobre el test mixto (porcion de Run B):
  - modelo:  d_lat predicho (de test_pred_<cfg>.npz) vs d_lat real
  - baseline: d_lat de la odometria de ruedas (lin_y integrado) vs d_lat real
Metricas: correlacion de Pearson, y fraccion de varianza del d_lat real explicada.

Uso:  python analyze_slip.py --cfg imu
"""
import argparse, csv
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", default="imu")
args = ap.parse_args()

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
MODELS = Path(r"C:\Users\Usuario\Desktop\TFM\models")
DT = 1.0; SEG = 2.0; MOVE_EPS = 0.05


def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])


# --- predicciones del modelo (test mixto) ---
pred_npz = np.load(MODELS / f"test_pred_gru_{args.cfg}.npz")
mdl_lat_pred = pred_npz["pred"][:, 1]   # d_lat predicho
lat_real_mdl = pred_npz["true"][:, 1]   # d_lat real (del target)
print(f"Modelo: {len(mdl_lat_pred)} muestras de test\n")

# --- baseline: recomputar d_lat de la odometria sobre el test mixto ---
d = np.load(BASE / "Run_B" / "dataset_imu_mixed_test.npz", allow_pickle=True) \
    if (BASE / "Run_B" / "dataset_imu_mixed_test.npz").exists() \
    else np.load(BASE / "dataset_imu_mixed_test.npz", allow_pickle=True)
test_segs = set(int(s)-2000 for s in np.unique(d["seg_id"]))

gt = load(BASE / "Run_B" / "ground_truth.csv")
wo = load(BASE / "Run_B" / "wheel_odometry.csv")
gt_t = gt[:,0]; wo_t = wo[:,0]/1e9

def pos(ti):
    return np.array([np.interp(ti, gt_t, gt[:,1]), np.interp(ti, gt_t, gt[:,2])])
def wheel_disp(t0,t1):
    m=(wo_t>=t0)&(wo_t<=t1)
    if m.sum()<2: return None
    tt=wo_t[m]; vx=wo[m,1]; vy=wo[m,2]; dts=np.diff(tt)
    return (float(np.sum((vx[:-1]+vx[1:])/2*dts)), float(np.sum((vy[:-1]+vy[1:])/2*dts)))

segs,start=[],0
for i in range(1,len(gt)):
    if gt_t[i]-gt_t[i-1]>SEG: segs.append((start,i)); start=i
segs.append((start,len(gt)))

lat_real_bas, bas_lat = [], []
seg_id=0
for (s,e) in segs:
    t0,t1=gt_t[s],gt_t[e-1]
    if t1-t0<2*DT: continue
    seg_id+=1
    if seg_id not in test_segs: continue
    t=t0+DT
    while t+DT<=t1:
        d_prev=pos(t)-pos(t-DT); d_now=pos(t+DT)-pos(t)
        if np.hypot(*d_prev)>=MOVE_EPS:
            heading=np.arctan2(d_prev[1],d_prev[0]); c,sr=np.cos(heading),np.sin(heading)
            tl=-sr*d_now[0]+c*d_now[1]
            wd=wheel_disp(t,t+DT)
            if wd is not None:
                lat_real_bas.append(tl); bas_lat.append(wd[1])
        t+=DT
lat_real_bas=np.array(lat_real_bas); bas_lat=np.array(bas_lat)

def stats(name, pred, real):
    pred=np.asarray(pred); real=np.asarray(real)
    if pred.std() < 1e-9:
        corr = 0.0
        note = " (prediccion CONSTANTE -> corr indefinida, se toma 0)"
    else:
        corr = np.corrcoef(pred, real)[0,1]; note=""
    # fraccion de varianza explicada (R2 respecto a predecir la media)
    ss_res = np.sum((real-pred)**2); ss_tot = np.sum((real-real.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0
    print(f"  {name}:")
    print(f"    corr(pred, real) = {corr:+.3f}{note}")
    print(f"    std(pred) = {pred.std()*100:.2f} cm   std(real) = {real.std()*100:.2f} cm")
    print(f"    R2 (var. explicada) = {r2:+.3f}")

print("=== Captura del deslizamiento lateral (d_lat) en el test mixto ===\n")
stats("MODELO", mdl_lat_pred, lat_real_mdl)
print()
stats("BASELINE (odometria)", bas_lat, lat_real_bas)
print("\n=== LECTURA ===")
print("- corr alta y positiva en el modelo = capta la senal de deslizamiento.")
print("- baseline con prediccion casi constante (~0) = corr ~0 = NO capta el")
print("  deslizamiento (es estructuralmente ciego), aunque su MAE medio parezca")
print("  bajo solo porque el deslizamiento suele ser pequeno.")
print("- Si el modelo tiene corr baja tambien, ni el modelo capta bien el slip:")
print("  habria que reencuadrar el argumento (resultado honesto a conocer).")