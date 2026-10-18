"""
reconstruct.py -- Reconstruye la trayectoria de Run_A acumulando los incrementos
relativos (d_fwd, d_lat, d_up) predichos por el modelo solo-IMU de la particion
entre recorridos (entrenado con Run_B) y calcula el ATE frente al DGPS.

Dos versiones de rumbo, con los MISMOS incrementos:
  ideal         : direccion real de movimiento del DGPS (no valido en inferencia).
  propioceptivo : yaw del filtro Madgwick (orientation_full.csv).

Reconstruccion por tramos (seg_id), cada tramo anclado en su punto DGPS inicial.
ATE con alineacion rigida (Umeyama sin escala) en XY sobre el recorrido completo
y ajuste de offset en z. Se calcula tambien el ATE trivial (estimar siempre el
centro del recorrido) como referencia de saturacion.

El ATE de la odometria clasica (baseline_odometry.py) usa otro protocolo
(dead reckoning desde cero por tramo, alineacion rigida por tramo) y no es
comparable con este; no se muestra aqui.

Uso:  python reconstruct.py
"""
from pathlib import Path
import csv
import numpy as np

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
MODELS = Path(r"C:\Users\Usuario\Desktop\TFM\models")
PRED_FILE = "test_pred_gru_imu.npz"
DT = 1.0


def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])


def align(est, gt):
    """Alineacion rigida (Umeyama sin escala) en XY; offset en z."""
    mu_s, mu_d = est[:, :2].mean(0), gt[:, :2].mean(0)
    s0, d0 = est[:, :2] - mu_s, gt[:, :2] - mu_d
    U, _, Vt = np.linalg.svd(d0.T @ s0)
    sg = np.ones(2)
    if np.linalg.det(U @ Vt) < 0:
        sg[-1] = -1
    R = U @ np.diag(sg) @ Vt
    out = est.copy()
    out[:, :2] = (R @ s0.T).T + mu_d
    out[:, 2] = est[:, 2] - est[:, 2].mean() + gt[:, 2].mean()
    return out


def rmse(a, b):
    return np.sqrt((np.linalg.norm(a - b, axis=1) ** 2).mean())


def ate(est, gt):
    return rmse(align(est, gt), gt)


def ate_trivial(gt):
    return rmse(np.tile(gt.mean(0), (len(gt), 1)), gt)


_pd = np.load(MODELS / PRED_FILE)
PRED, SEG = _pd["pred"], _pd["seg_id"]
_ds = np.load(BASE / "Run_A" / "dataset_imu_Run_A.npz", allow_pickle=True)
if not np.allclose(_ds["y"], _pd["true"], atol=1e-5):
    raise RuntimeError("El orden de las predicciones no coincide con el dataset.")
T_START = _ds["t_start"]

_gt = load(BASE / "Run_A" / "ground_truth.csv")
_ori = load(BASE / "Run_A" / "orientation_full.csv")
_ori_t = _ori[:, 0] / 1e9


def gt_pos(t):
    return np.array([np.interp(t, _gt[:, 0], _gt[:, c]) for c in (1, 2, 3)])


def heading(t, mode):
    if mode == "ideal":
        d = gt_pos(t + DT) - gt_pos(t)
        return np.arctan2(d[1], d[0])
    return np.deg2rad(np.interp(t, _ori_t, _ori[:, 7]))


def reconstruct(mode):
    """Devuelve listas de tramos (estimado, referencia), cada uno Nx3."""
    est_segs, gt_segs = [], []
    for s in np.unique(SEG):
        idx = np.where(SEG == s)[0]
        idx = idx[np.argsort(T_START[idx])]
        p = gt_pos(T_START[idx[0]])
        est, ref = [p.copy()], [p.copy()]
        for k in idx:
            h = heading(T_START[k], mode)
            c, sn = np.cos(h), np.sin(h)
            df, dl, du = PRED[k]
            p = p + np.array([c * df - sn * dl, sn * df + c * dl, du])
            est.append(p.copy()); ref.append(gt_pos(T_START[k] + DT))
        est_segs.append(np.array(est)); gt_segs.append(np.array(ref))
    return est_segs, gt_segs


if __name__ == "__main__":
    print(f"=== Reconstruccion Run_A -- {PRED_FILE} ===")
    for mode, label in [("ideal", "rumbo ideal (DGPS)"), ("prop", "rumbo propioceptivo")]:
        E, G = reconstruct(mode)
        est, ref = np.vstack(E), np.vstack(G)
        print(f"  ATE {label:22s}: {ate(est, ref):6.2f} m")
    print(f"  ATE trivial (centro del recorrido): {ate_trivial(ref):6.2f} m")
    print(f"  Tramos: {len(np.unique(SEG))}")