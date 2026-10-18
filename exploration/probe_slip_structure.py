"""
probe_slip_structure.py -- Investiga si el deslizamiento lateral (d_lat) es
genuinamente impredecible (ruido) o si tiene estructura fisica que el modelo no
esta captando. Tres comprobaciones sobre Run_B (donde esta el test mixto):

  1) AUTOCORRELACION del d_lat real entre intervalos consecutivos: si el slip
     persiste (corr alta), es senal fisica; si no, es mas bien ruido.
  2) CORRELACION de |d_lat| con causas fisicas: velocidad de giro (ang_z) e
     inclinacion (pitch), integradas en el intervalo. Si correlaciona, el slip
     tiene causa observable.
  3) Relacion senal/ruido: compara la magnitud del d_lat con el suelo de ruido
     del GPS (~0.7 cm) para ver cuanto del d_lat podria ser ruido.

Uso:  python probe_slip_structure.py
"""
import csv
from pathlib import Path
import numpy as np

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
RUN = "Run_B"
DT = 1.0; SEG = 2.0; MOVE_EPS = 0.05
GPS_NOISE = 0.007  # m, suelo de ruido del GPS (del Bloque 5)


def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])


gt = load(BASE / RUN / "ground_truth.csv")
wo = load(BASE / RUN / "wheel_odometry.csv")
imu = load(BASE / RUN / "imu.csv")
gt_t = gt[:,0]; wo_t = wo[:,0]/1e9; imu_t = imu[:,0]/1e9


def pos(ti):
    return np.array([np.interp(ti, gt_t, gt[:,1]), np.interp(ti, gt_t, gt[:,2])])

segs,start=[],0
for i in range(1,len(gt)):
    if gt_t[i]-gt_t[i-1]>SEG: segs.append((start,i)); start=i
segs.append((start,len(gt)))

# recolectar por tramo: d_lat, giro (|ang_z| medio wheel), inclinacion (|gyro_y| como proxy de cambio de pitch)
dlat_all=[]; turn_all=[]; incl_all=[]; dlat_by_seg=[]
for (s,e) in segs:
    t0,t1=gt_t[s],gt_t[e-1]
    if t1-t0<2*DT: continue
    seg_dlat=[]
    t=t0+DT
    while t+DT<=t1:
        d_prev=pos(t)-pos(t-DT); d_now=pos(t+DT)-pos(t)
        if np.hypot(*d_prev)>=MOVE_EPS:
            heading=np.arctan2(d_prev[1],d_prev[0]); c,sr=np.cos(heading),np.sin(heading)
            dlat=-sr*d_now[0]+c*d_now[1]
            # causa fisica: giro medio del wheel (ang_z) y actividad de pitch (gyro_y)
            mw=(wo_t>=t)&(wo_t<t+DT); mi=(imu_t>=t)&(imu_t<t+DT)
            turn = np.abs(wo[mw,6]).mean() if mw.sum()>0 else np.nan
            incl = np.abs(imu[mi,2]).mean() if mi.sum()>0 else np.nan  # gyro_y
            dlat_all.append(dlat); turn_all.append(turn); incl_all.append(incl)
            seg_dlat.append(dlat)
        t+=DT
    if len(seg_dlat)>3: dlat_by_seg.append(np.array(seg_dlat))

dlat=np.array(dlat_all); turn=np.array(turn_all); incl=np.array(incl_all)
ok=~np.isnan(turn)&~np.isnan(incl)
dlat_c=dlat[ok]; turn=turn[ok]; incl=incl[ok]

print(f"=== Estructura del deslizamiento lateral (Run_B, {len(dlat)} intervalos) ===\n")

# 1) autocorrelacion lag-1 dentro de cada tramo
acs=[]
for sd in dlat_by_seg:
    if len(sd)>3 and sd[:-1].std()>1e-9 and sd[1:].std()>1e-9:
        acs.append(np.corrcoef(sd[:-1],sd[1:])[0,1])
print(f"1) AUTOCORRELACION lag-1 del d_lat (media sobre tramos): {np.mean(acs):+.3f}")
print(f"   (alta -> el deslizamiento persiste = senal fisica; ~0 -> ruido)\n")

# 2) correlacion con causas fisicas
print(f"2) |d_lat| vs causas fisicas:")
print(f"   corr(|d_lat|, |giro ang_z|)   = {np.corrcoef(np.abs(dlat_c),turn)[0,1]:+.3f}")
print(f"   corr(|d_lat|, |pitch gyro_y|) = {np.corrcoef(np.abs(dlat_c),incl)[0,1]:+.3f}")
print(f"   (positiva -> el slip aumenta al girar/inclinarse = causa observable)\n")

# 3) senal / ruido
print(f"3) Magnitud del d_lat vs ruido GPS:")
print(f"   std(d_lat) = {dlat.std()*100:.2f} cm   |d_lat| medio = {np.abs(dlat).mean()*100:.2f} cm")
print(f"   suelo de ruido GPS ~ {GPS_NOISE*100:.1f} cm")
print(f"   SNR aprox (std_dlat / ruido) = {dlat.std()/GPS_NOISE:.1f}")
frac_below = np.mean(np.abs(dlat) < GPS_NOISE)*100
print(f"   fraccion de intervalos con |d_lat| < ruido GPS: {frac_below:.0f}%")
print(f"   (si gran parte del d_lat esta bajo el ruido, es intrinsecamente dificil)\n")

print("=== LECTURA ===")
print("- Autocorr alta + corr con giro/pitch = slip fisico predecible en principio.")
print("- Autocorr ~0 + sin corr + mucha fraccion bajo ruido = slip dominado por")
print("  ruido/impredecible: honesto documentar que no es recuperable propioceptiv.")