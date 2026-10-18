"""
probe_slip_artifact.py -- Comprueba si la autocorrelacion negativa del d_lat
(-0.45) es un ARTEFACTO de como calculamos el target o refleja el deslizamiento
real. Calcula d_lat de varias formas y compara su autocorrelacion lag-1:

  A) referencia = movimiento del intervalo ANTERIOR (como el target actual).
  B) referencia = direccion de movimiento SUAVIZADA (ventana de +/-3 s).
  C) referencia = direccion del PROPIO intervalo (auto-referencia).
  D) a distintas escalas de intervalo (1, 2, 3 s) con referencia suavizada.

Si la autocorr negativa DESAPARECE con referencia suavizada o a mayor escala,
era artefacto del ruido en la referencia. Si PERSISTE, el d_lat a 1s esta
genuinamente dominado por ruido.

Uso:  python probe_slip_artifact.py
"""
import csv
from pathlib import Path
import numpy as np

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
RUN = "Run_B"
SEG = 2.0; MOVE_EPS = 0.05


def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])


gt = load(BASE / RUN / "ground_truth.csv")
gt_t = gt[:,0]


def pos(ti):
    return np.array([np.interp(ti, gt_t, gt[:,1]), np.interp(ti, gt_t, gt[:,2])])


def segments():
    segs,start=[],0
    for i in range(1,len(gt)):
        if gt_t[i]-gt_t[i-1]>SEG: segs.append((start,i)); start=i
    segs.append((start,len(gt)))
    return [(s,e) for (s,e) in segs if gt_t[e-1]-gt_t[s] >= 4.0]


def autocorr_lag1(seqs):
    acs=[]
    for sd in seqs:
        sd=np.array(sd)
        if len(sd)>3 and sd[:-1].std()>1e-9 and sd[1:].std()>1e-9:
            acs.append(np.corrcoef(sd[:-1],sd[1:])[0,1])
    return np.mean(acs) if acs else np.nan


def compute(dt, ref_mode, smooth_s=3.0):
    """Devuelve lista de secuencias de d_lat por tramo, con la referencia indicada."""
    out=[]
    for (s,e) in segments():
        t0,t1=gt_t[s],gt_t[e-1]
        seq=[]
        t=t0+dt
        while t+dt<=t1:
            d_now=pos(t+dt)-pos(t)
            if np.hypot(*d_now)>=MOVE_EPS:
                if ref_mode=="prev":
                    ref=pos(t)-pos(t-dt)
                elif ref_mode=="self":
                    ref=d_now
                elif ref_mode=="smooth":
                    # direccion de movimiento suavizada en +/- smooth_s
                    ref=pos(min(t+smooth_s,t1))-pos(max(t-smooth_s,t0))
                if np.hypot(*ref)>=MOVE_EPS:
                    h=np.arctan2(ref[1],ref[0]); c,sr=np.cos(h),np.sin(h)
                    dlat=-sr*d_now[0]+c*d_now[1]
                    seq.append(dlat)
            t+=dt
        if len(seq)>3: out.append(seq)
    return out


print(f"=== Autocorrelacion lag-1 del d_lat segun como se calcula (Run_B) ===\n")
print(f"  A) ref = intervalo anterior (target actual), dt=1s:   "
      f"{autocorr_lag1(compute(1.0,'prev')):+.3f}")
print(f"  B) ref = movimiento suavizado (+/-3s),        dt=1s:   "
      f"{autocorr_lag1(compute(1.0,'smooth')):+.3f}")
print(f"  C) ref = propio intervalo (auto-ref),         dt=1s:   "
      f"{autocorr_lag1(compute(1.0,'self')):+.3f}")
print()
for dt in (1.0, 2.0, 3.0):
    print(f"  D) ref suavizada, dt={dt:.0f}s:   autocorr={autocorr_lag1(compute(dt,'smooth')):+.3f}")
print()
# magnitud del d_lat con referencia suavizada a 1s (mas limpia)
seqs=compute(1.0,'smooth')
allv=np.concatenate([np.array(s) for s in seqs])
print(f"  Magnitud d_lat (ref suavizada, 1s): std={allv.std()*100:.2f} cm  "
      f"|medio|={np.abs(allv).mean()*100:.2f} cm")
print("\n=== LECTURA ===")
print("- Si A es muy negativa pero B/C/D se acercan a 0 o positivo: la oscilacion")
print("  era ARTEFACTO de la referencia ruidosa del intervalo anterior.")
print("- Si sigue negativa en todas: el d_lat a 1s esta dominado por ruido real.")