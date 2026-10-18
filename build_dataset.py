"""
build_dataset.py -- Ensamblado final del dataset del Bloque 5.

Junta todas las decisiones ya cerradas y verificadas empiricamente:

  B1 base comun 100 Hz : IMU tal cual (100 Hz exacto, verificado sin saltos);
                         wheel interpolado linealmente a los timestamps del IMU.
  B2 ventana 1 s       : 100 muestras por ventana, emparejada 1:1 con el target
                         del intervalo [t, t+1s].
  B3 solape            : train (Run_B) paso 0.5 s (target regenerado a ese paso);
                         test (Run_A) sin solape (paso 1.0 s).
  B4 segmentacion      : ninguna ventana ni target cruza un hueco de GT > 2 s
                         (umbral SEG). Se respeta via seg_id.
  Canales              : IMU-only  = 6  (gyro x3, acc x3)
                         IMU+wheel = 10 (+ lin_x, lin_y, ang_y, ang_z)
                         (ang_x y lin_z excluidos: constantes, verificado).
  Filtro wheel         : en IMU+wheel se excluyen intervalos con > 50 % de su
                         duracion sobre ceros del wheel (sensor caido).
  C normalizacion      : z-score por canal; media/std de la SENAL CONTINUA de
                         Run_B (tramos validos); aplicada identica a train y test.
                         Guardadas en norm_stats.json.

Target (relativo al propio movimiento, ya validado en build_target.py):
  d_fwd (avance), d_lat (deslizamiento), d_up (vertical). forward = direccion de
  movimiento GPS del intervalo anterior. Primer intervalo de cada tramo descartado.

Salida (.npz por config y run) con arrays:
  X       (n, 100, n_canales)  ventanas de entrada NORMALIZADAS
  y       (n, 3)               targets [d_fwd, d_lat, d_up]
  seg_id  (n,)                 id de tramo continuo de cada ventana
  t_start (n,)                 timestamp de inicio de cada ventana
  channels                     nombres de canal (para trazabilidad)
Y norm_stats.json con media/std por canal de cada config.

Uso:
  # 1) calcular stats desde Run_B y construir train:
  python build_dataset.py Run_B --role train
  # 2) construir test con las stats ya guardadas:
  python build_dataset.py Run_A --role test

NOTA: ejecutar primero Run_B (train) para generar norm_stats.json, luego Run_A.
"""
import sys, csv, json, argparse
from pathlib import Path
import numpy as np

# ---- constantes de diseno (cerradas y verificadas) ----
FS = 100.0            # Hz, base comun
WIN = 100             # muestras por ventana (1 s)
DT = 1.0              # s, duracion del intervalo objetivo
SEG = 2.0             # s, umbral de segmentacion
MOVE_EPS = 0.05       # m, movimiento previo minimo para 'forward' fiable
STEP_TRAIN = 0.5      # s, paso entre ventanas en train
STEP_TEST = 1.0       # s, paso entre ventanas en test (sin solape)
ZERO_COVER_MAX = 0.50 # excluir intervalo de IMU+wheel si > 50 % sobre ceros wheel

IMU_CH = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
WHEEL_CH = ["lin_x", "lin_y", "ang_y", "ang_z"]   # con senal real (verificado)
# indices de esos canales dentro de wheel_odometry.csv:
# header: timestamp_ns, lin_x, lin_y, lin_z, ang_x, ang_y, ang_z
WHEEL_COL = {"lin_x": 1, "lin_y": 2, "lin_z": 3, "ang_x": 4, "ang_y": 5, "ang_z": 6}


def load_csv(p, expect_first=None):
    with open(p) as f:
        r = csv.reader(f); header = next(r)
        if expect_first and header[0] != expect_first:
            raise ValueError(f"{p}: cabecera inesperada {header[0]!r}")
        data = np.array([[float(x) for x in row] for row in r])
    return header, data


def segment(t, seg_thr):
    segs, start = [], 0
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > seg_thr:
            segs.append((start, i)); start = i
    segs.append((start, len(t)))
    return segs


def load_run(base, run):
    _, gt = load_csv(base / run / "ground_truth.csv", "timestamp_s")
    gt = gt[np.argsort(gt[:, 0])]
    _, imu = load_csv(base / run / "imu.csv", "timestamp_ns")
    _, wo = load_csv(base / run / "wheel_odometry.csv", "timestamp_ns")
    imu = imu[np.argsort(imu[:, 0])]
    wo = wo[np.argsort(wo[:, 0])]
    return gt, imu, wo


def wheel_zero_blocks(wo):
    """Bloques [t_ini, t_fin] donde el wheel es todo-cero (para el filtro de cobertura)."""
    t = wo[:, 0] / 1e9
    allzero = np.all(wo[:, 1:] == 0.0, axis=1)
    blocks = []
    i = 0
    while i < len(allzero):
        if allzero[i]:
            j = i
            while j < len(allzero) and allzero[j]:
                j += 1
            blocks.append((t[i], t[j - 1]))
            i = j
        else:
            i += 1
    return blocks


def zero_cover_fraction(t0, t1, zblocks):
    """Fraccion del intervalo [t0,t1] cubierta por bloques de ceros wheel."""
    cov = 0.0
    for (a, b) in zblocks:
        cov += max(0.0, min(t1, b) - max(t0, a))
    return cov / (t1 - t0)


def build_windows(base, run, step, want_wheel, zblocks=None):
    """
    Genera ventanas e y-targets. Devuelve:
      X (n, WIN, nch)  SIN normalizar,  y (n,3),  seg_id (n,),  t_start (n,)
    Reconstruye el target con la logica validada (forward = mov. previo GPS),
    con el paso 'step' indicado (1.0 test / 0.5 train).
    """
    gt, imu, wo = load_run(base, run)
    gt_t = gt[:, 0]
    imu_t = imu[:, 0] / 1e9
    wo_t = wo[:, 0] / 1e9

    # canales de entrada e interpolacion del wheel a los timestamps del IMU
    imu_sig = imu[:, 1:7]                      # 6 canales IMU en su malla nativa
    if want_wheel:
        wheel_interp = np.column_stack([
            np.interp(imu_t, wo_t, wo[:, WHEEL_COL[c]]) for c in WHEEL_CH
        ])
        channels = IMU_CH + WHEEL_CH
    else:
        wheel_interp = None
        channels = list(IMU_CH)

    def pos(ti):
        return np.array([np.interp(ti, gt_t, gt[:, 1]),
                         np.interp(ti, gt_t, gt[:, 2]),
                         np.interp(ti, gt_t, gt[:, 3])])

    def window_at(t_start):
        """Ventana de WIN muestras del IMU cuyo primer instante es >= t_start."""
        i0 = np.searchsorted(imu_t, t_start)
        if i0 + WIN > len(imu_t):
            return None
        sl = slice(i0, i0 + WIN)
        if want_wheel:
            return np.hstack([imu_sig[sl], wheel_interp[sl]])
        return imu_sig[sl]

    segs = segment(gt_t, SEG)
    X, Y, SID, TS = [], [], [], []
    n_excluded_wheel = 0
    seg_id = 0
    for (s, e) in segs:
        t0, t1 = gt_t[s], gt_t[e - 1]
        if t1 - t0 < 2 * DT:
            continue
        seg_id += 1
        # bordes con el paso pedido; el primer intervalo del tramo se descarta
        edges = np.arange(t0, t1 - DT + 1e-9, step)  # inicio de cada intervalo
        p_prev_ref = t0
        for ts in edges:
            # necesitamos un intervalo previo completo [ts-DT, ts] dentro del tramo
            if ts - DT < t0:
                continue
            if ts + DT > t1:
                break
            d_prev = pos(ts) - pos(ts - DT)
            d_now = pos(ts + DT) - pos(ts)
            if np.hypot(d_prev[0], d_prev[1]) < MOVE_EPS:
                continue
            # filtro de cobertura de ceros wheel (solo config IMU+wheel)
            if want_wheel and zblocks is not None:
                if zero_cover_fraction(ts, ts + DT, zblocks) > ZERO_COVER_MAX:
                    n_excluded_wheel += 1
                    continue
            w = window_at(ts)
            if w is None:
                continue
            heading = np.arctan2(d_prev[1], d_prev[0])
            c, sr = np.cos(heading), np.sin(heading)
            d_fwd = c * d_now[0] + sr * d_now[1]
            d_lat = -sr * d_now[0] + c * d_now[1]
            d_up = d_now[2]
            X.append(w); Y.append([d_fwd, d_lat, d_up]); SID.append(seg_id); TS.append(ts)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    SID = np.array(SID, dtype=np.int32)
    TS = np.array(TS, dtype=np.float64)
    return X, Y, SID, TS, channels, n_excluded_wheel


def continuous_signal_stats(base, run, want_wheel):
    """Media/std por canal sobre la SENAL CONTINUA de tramos validos (no ventanas)."""
    gt, imu, wo = load_run(base, run)
    gt_t = gt[:, 0]
    imu_t = imu[:, 0] / 1e9
    wo_t = wo[:, 0] / 1e9
    imu_sig = imu[:, 1:7]
    if want_wheel:
        wheel_interp = np.column_stack([
            np.interp(imu_t, wo_t, wo[:, WHEEL_COL[c]]) for c in WHEEL_CH
        ])
        sig = np.hstack([imu_sig, wheel_interp])
    else:
        sig = imu_sig
    # mascara: instantes IMU dentro de un tramo valido de GT
    segs = segment(gt_t, SEG)
    mask = np.zeros(len(imu_t), dtype=bool)
    for (s, e) in segs:
        t0, t1 = gt_t[s], gt_t[e - 1]
        if t1 - t0 < 2 * DT:
            continue
        mask |= (imu_t >= t0) & (imu_t <= t1)
    sig = sig[mask]
    mean = sig.mean(axis=0)
    std = sig.std(axis=0)
    std[std < 1e-8] = 1.0   # evitar division por cero
    return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--role", choices=["train", "test"], required=True)
    ap.add_argument("--base", default=r"C:\Users\Usuario\Desktop\TFM\extracted")
    ap.add_argument("--stats", default=r"C:\Users\Usuario\Desktop\TFM\extracted\norm_stats.json")
    args = ap.parse_args()

    base = Path(args.base)
    step = STEP_TRAIN if args.role == "train" else STEP_TEST
    stats_path = Path(args.stats)

    # zblocks para el filtro de la config IMU+wheel
    _, _, wo = load_run(base, args.run)
    zblocks = wheel_zero_blocks(wo)

    # --- stats: si es train, se calculan y guardan; si es test, se cargan ---
    if args.role == "train":
        stats = {}
        for cfg, want_wheel in [("imu", False), ("imuwheel", True)]:
            mean, std = continuous_signal_stats(base, args.run, want_wheel)
            stats[cfg] = {"mean": mean.tolist(), "std": std.tolist()}
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"norm_stats guardadas en {stats_path}")
    else:
        if not stats_path.exists():
            print(f"ERROR: no existe {stats_path}. Ejecuta primero Run_B --role train.")
            sys.exit(1)
        stats = json.loads(stats_path.read_text())

    print(f"\n=== BUILD DATASET -- {args.run} ({args.role}, paso {step}s) ===")
    for cfg, want_wheel in [("imu", False), ("imuwheel", True)]:
        X, Y, SID, TS, channels, n_excl = build_windows(
            base, args.run, step, want_wheel, zblocks if want_wheel else None)
        mean = np.array(stats[cfg]["mean"], dtype=np.float32)
        std = np.array(stats[cfg]["std"], dtype=np.float32)
        Xn = (X - mean) / std     # normalizacion z-score por canal
        out = base / args.run / f"dataset_{cfg}_{args.run}.npz"
        np.savez_compressed(out, X=Xn, y=Y, seg_id=SID, t_start=TS,
                            channels=np.array(channels))
        print(f"  [{cfg:9s}] ventanas={len(X):6d}  canales={len(channels)}  "
              f"tramos={len(np.unique(SID))}  "
              + (f"excl_wheel={n_excl}  " if want_wheel else "")
              + f"-> {out.name}")
        # firma rapida del target para verificar coherencia
        mv = np.hypot(Y[:, 0], Y[:, 1]) > MOVE_EPS
        af, al = np.abs(Y[mv, 0]).mean(), np.abs(Y[mv, 1]).mean()
        print(f"              reparto fwd/lat: {af/(af+al)*100:.0f}/{al/(af+al)*100:.0f}  "
              f"d_fwd={af*100:.1f}cm d_lat={al*100:.1f}cm")

    print("\nHecho.")


if __name__ == "__main__":
    main()