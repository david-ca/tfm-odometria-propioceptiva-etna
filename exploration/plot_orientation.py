"""
plot_orientation.py — Grafica la orientacion del filtro y la compara con la
altura z del GPS para confirmar visualmente que el pitch capta el relieve.
Uso:  python plot_orientation.py Run_A
"""
import sys, csv
from pathlib import Path
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("falta matplotlib"); sys.exit(1)

RUN = sys.argv[1] if len(sys.argv) > 1 else "Run_A"
BASE = Path(r"C:\Users\Usuario\Desktop\TFM") / "extracted" / RUN

def load(p, skip=True):
    with open(p) as f:
        r = csv.reader(f)
        if skip: next(r)
        return np.array([[float(x) for x in row] for row in r])

ori = load(BASE / "orientation_probe.csv")   # t, roll, pitch, yaw
gt  = load(BASE / "ground_truth.csv")         # t, x, y, z

# recorte del GPS al rango temporal de la orientacion
t0, t1 = ori[0,0], ori[-1,0]
m = (gt[:,0] >= t0) & (gt[:,0] <= t1)
gt_seg = gt[m]

fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel 1: roll, pitch, yaw
ax[0].plot(ori[:,0]-t0, ori[:,1], label='roll', linewidth=0.8)
ax[0].plot(ori[:,0]-t0, ori[:,2], label='pitch', linewidth=0.8)
ax[0].set_ylabel('grados'); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_title(f'{RUN}: roll y pitch (filtro Madgwick)')

# Panel 2: yaw
ax[1].plot(ori[:,0]-t0, ori[:,3], 'purple', linewidth=0.8, label='yaw')
ax[1].set_ylabel('yaw (grados)'); ax[1].legend(); ax[1].grid(alpha=0.3)

# Panel 3: pitch del filtro vs altura z del GPS (dos ejes)
ax2 = ax[2]
ax2.plot(ori[:,0]-t0, ori[:,2], 'b-', linewidth=0.8, label='pitch (filtro)')
ax2.set_ylabel('pitch (grados)', color='b'); ax2.grid(alpha=0.3)
ax3 = ax2.twinx()
ax3.plot(gt_seg[:,0]-t0, gt_seg[:,3], 'r-', linewidth=1.2, label='z GPS')
ax3.set_ylabel('z GPS (m)', color='r')
ax2.set_xlabel('tiempo (s)')
ax2.set_title('pitch (filtro) vs altura z (GPS): ¿coinciden subidas/bajadas?')

plt.tight_layout()
plt.savefig(BASE / f"orientation_{RUN}.png", dpi=130, bbox_inches='tight')
print(f"Guardado: orientation_{RUN}.png")