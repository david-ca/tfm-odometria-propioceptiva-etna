"""
baseline_odometry.py — Baseline clasica de odometria (Opcion A: wheel odometry).
v2: dos variantes de alineamiento.
  - GLOBAL (Umeyama): mejor encaje global de la forma. Estandar academico del ATE.
  - ANCLAJE: fija el inicio en el punto GPS real y ajusta solo la orientacion
    inicial optima. Mide la deriva desde un inicio conocido (uso real).

Reconstruye por dead reckoning 2D integrando velocidades del cuerpo
(lin_x, lin_y, ang_z) por TRAMO continuo (umbral 2 s), calcula ATE y RTE
en metros a varias longitudes, exporta trayectorias y guarda graficas PNG.

Convenciones verificadas (baseline_diagnose.py): 2D, ang_z signo +1, lin_y incluido.

Uso:  python baseline_odometry.py Run_A
      python baseline_odometry.py Run_B
"""
import sys
import csv
import math
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: falta numpy.  pip install numpy"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False

if len(sys.argv) < 2:
    print("Uso: python baseline_odometry.py <NOMBRE_DEL_RUN>"); sys.exit(1)

RUN = sys.argv[1]
BASE = Path(r"C:\Users\Usuario\Desktop\TFM") / "extracted" / RUN

SEG_THRESHOLD = 2.0
RTE_LENGTHS = [1.0, 5.0, 10.0, 25.0, 50.0]
SIGN_WZ = +1
MIN_SEG_DUR = 5.0

def load(path):
    with open(path) as f:
        r = csv.reader(f); header = next(r)
        return header, [[float(x) for x in row] for row in r]

_, wo = load(BASE / "wheel_odometry.csv")
_, gt = load(BASE / "ground_truth.csv")
gt = np.array(gt); wo = np.array(wo)
wo_t = wo[:, 0] / 1e9

def segment_gt(gt, threshold):
    segs, start = [], 0
    for i in range(1, len(gt)):
        if gt[i, 0] - gt[i-1, 0] > threshold:
            segs.append((start, i)); start = i
    segs.append((start, len(gt)))
    return segs

def dead_reckon(wo_slice):
    x = y = th = 0.0
    traj = [(x, y)]; times = [wo_slice[0, 0] / 1e9]
    for i in range(1, len(wo_slice)):
        dt = (wo_slice[i, 0] - wo_slice[i-1, 0]) / 1e9
        if dt <= 0 or dt > SEG_THRESHOLD:
            traj.append((x, y)); times.append(wo_slice[i, 0]/1e9); continue
        vx = wo_slice[i, 1]; vy = wo_slice[i, 2]; wz = SIGN_WZ * wo_slice[i, 6]
        th += wz * dt
        x += (vx*math.cos(th) - vy*math.sin(th)) * dt
        y += (vx*math.sin(th) + vy*math.cos(th)) * dt
        traj.append((x, y)); times.append(wo_slice[i, 0]/1e9)
    return np.array(traj), np.array(times)

def umeyama_2d(src, dst):
    mu_s = src.mean(axis=0); mu_d = dst.mean(axis=0)
    H = (src-mu_s).T @ (dst-mu_d)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1; R = Vt.T @ U.T
    return R, mu_d - R @ mu_s

def align_global(src, dst):
    R, t = umeyama_2d(src, dst)
    return (R @ src.T).T + t

def align_anchored(src, dst):
    s0 = src[0]; d0 = dst[0]
    S = src - s0; D = dst - d0
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1; R = Vt.T @ U.T
    return (R @ S.T).T + d0

def resample_to_gps(dr_traj, dr_times, gps_times):
    out = []
    for tg in gps_times:
        idx = np.searchsorted(dr_times, tg)
        out.append(dr_traj[min(max(idx, 0), len(dr_traj)-1)])
    return np.array(out)

def ate_dists(est, gps):
    return np.linalg.norm(est - gps, axis=1)

def cumulative_arclength(traj):
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])

def rte_at_length(est, gps, length):
    arc = cumulative_arclength(gps)
    errs = []; n = len(gps)
    for i in range(n):
        j = np.searchsorted(arc, arc[i] + length)
        if j >= n: break
        e_sub, g_sub = est[i:j+1], gps[i:j+1]
        if len(e_sub) < 2: continue
        e_al = align_global(e_sub, g_sub)
        errs.append(np.linalg.norm(e_al[-1] - g_sub[-1]))
    if not errs: return None, 0
    errs = np.array(errs)
    return math.sqrt(np.mean(errs**2)), len(errs)

print(f"=== RUN: {RUN} — Baseline odometria clasica (wheel, 2D) ===")
segs = segment_gt(gt, SEG_THRESHOLD)
usable = [(s, e) for (s, e) in segs if gt[e-1, 0]-gt[s, 0] >= MIN_SEG_DUR]
print(f"Tramos usables (>= {MIN_SEG_DUR}s): {len(usable)}\n")

est_global, est_anchor, gps_all = [], [], []
d_global, d_anchor = [], []
seg_report = []

for (s, e) in usable:
    t0, t1 = gt[s, 0], gt[e-1, 0]
    gps = gt[s:e, 1:3]; gps_times = gt[s:e, 0]
    mask = (wo_t >= t0) & (wo_t <= t1)
    wo_slice = wo[mask]
    if len(wo_slice) < 2: continue
    dr, drt = dead_reckon(wo_slice)
    dr_rs = resample_to_gps(dr, drt, gps_times)
    g_al = align_global(dr_rs, gps)
    a_al = align_anchored(dr_rs, gps)
    est_global.append(g_al); est_anchor.append(a_al); gps_all.append(gps)
    dg = ate_dists(g_al, gps); da = ate_dists(a_al, gps)
    d_global.append(dg); d_anchor.append(da)
    seg_report.append((t1-t0, cumulative_arclength(gps)[-1],
                       math.sqrt(np.mean(dg**2)), math.sqrt(np.mean(da**2)), len(gps)))

def rmse(dlist):
    allc = np.concatenate(dlist)
    return math.sqrt(np.mean(allc**2))

print("--- ATE (Absolute Trajectory Error), RMSE global ---")
print(f"  Alineamiento GLOBAL (Umeyama):   {rmse(d_global):.3f} m")
print(f"  Alineamiento ANCLAJE en inicio:  {rmse(d_anchor):.3f} m")
print(f"  Distancia total evaluada: {sum(s[1] for s in seg_report):.1f} m")
print(f"  Tramos: {len(seg_report)}\n")

print("--- RTE (Relative Trajectory Error) por longitud [alineamiento GLOBAL] ---")
print(f"  {'long(m)':>8} | {'RTE(m)':>8} | {'%':>6} | {'n':>8}")
for L in RTE_LENGTHS:
    parts = []
    for est, gps in zip(est_global, gps_all):
        r, n = rte_at_length(est, gps, L)
        if r is not None: parts.append((r, n))
    if parts:
        tot = sum(n for _, n in parts)
        agg = math.sqrt(sum((r**2)*n for r, n in parts)/tot)
        print(f"  {L:>8.1f} | {agg:>8.3f} | {agg/L*100:>5.1f}% | {tot:>8}")

out_csv = BASE / "baseline_traj.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seg_id","gps_x","gps_y","global_x","global_y","anchor_x","anchor_y"])
    for sid, (g, eg, ea) in enumerate(zip(gps_all, est_global, est_anchor)):
        for i in range(len(g)):
            w.writerow([sid, g[i,0], g[i,1], eg[i,0], eg[i,1], ea[i,0], ea[i,1]])
print(f"\nTrayectorias exportadas a: {out_csv.name}")

if HAVE_PLT:
    for variant, est_list, tag in [("Umeyama global", est_global, "global"),
                                    ("anclaje en inicio", est_anchor, "anchor")]:
        plt.figure(figsize=(9, 9))
        for i, (g, est) in enumerate(zip(gps_all, est_list)):
            lg = 'GPS (ground truth)' if i == 0 else None
            le = 'Odometria clasica' if i == 0 else None
            plt.plot(g[:,0], g[:,1], 'g-', linewidth=2, label=lg)
            plt.plot(est[:,0], est[:,1], 'r--', linewidth=1.2, label=le)
        plt.axis('equal'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.title(f'{RUN}: real vs odometria clasica ({variant})')
        png = BASE / f"baseline_{RUN}_{tag}.png"
        plt.savefig(png, dpi=150, bbox_inches='tight'); plt.close()
        print(f"Grafica guardada: {png.name}")
else:
    print("(matplotlib no disponible: no se guardaron PNG)")

print("\n--- Resumen por tramo (8 mas largos) ---")
print(f"  {'dur(s)':>8} | {'long(m)':>8} | {'ATE glob':>8} | {'ATE ancl':>8} | {'n':>7}")
for dur, ln, ag, aa, npt in sorted(seg_report, key=lambda r: -r[1])[:8]:
    print(f"  {dur:>8.1f} | {ln:>8.1f} | {ag:>8.3f} | {aa:>8.3f} | {npt:>7}")

print("\nHecho.")