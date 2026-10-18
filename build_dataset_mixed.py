"""
build_dataset_mixed.py -- Particion MIXTA propuesta por el tutor.

Entrenamiento = Run A completo + 75% de los tramos de Run B (con solapamiento).
Prueba        = 25% restante de los tramos de Run B (sin solapamiento).
Separacion por TRAMOS (seg_id) para que no haya fuga: ningun tramo aparece a la
vez en train y test. Objetivo: dar al modelo ambos perfiles de velocidad (el
estrecho de B y el amplio de A) para mejorar la generalizacion del avance.

Reutiliza la maquinaria validada de build_dataset.py (build_windows, load_run,
wheel_zero_blocks, continuous_signal_stats). Normalizacion: stats z-score sobre
el conjunto de ENTRENAMIENTO combinado (A completo + porcion B), no solo B, ya
que el train ahora incluye A. Se guardan en norm_stats_mixed.json.

Salida: dataset_{imu,imuwheel}_mixed_train.npz y _mixed_test.npz.

Decision honesta a declarar: el test (porcion de B) comparte recorrido/dia/terreno
con parte del train (B), por lo que es una evaluacion MENOS severa que la Hold-Out
original (A independiente). Se reporta junto a la original, no en su lugar.

Uso:  python build_dataset_mixed.py

CAMBIOS v2 (09/10/2026):
  1) Se guarda t_start en los .npz de train y test (necesario para emparejar la
     linea base ventana a ventana y para reconstruir los tramos de prueba de B).
  2) Correccion en _stats_run: la exclusion de los tramos de test de B comparaba
     (seg_id + 2000) con los ids de test SIN el prefijo 2000, por lo que (si
     build_windows numera los tramos 1..N) nunca excluia nada y los tramos de
     test entraban en la media/desv de normalizacion. Ahora la exclusion se hace
     por TIEMPO (un tramo DGPS se excluye si contiene alguna ventana de test),
     sin depender de la numeracion. Se imprime cuantos tramos se excluyen: debe
     coincidir con el numero de tramos de test.
"""
import json
from pathlib import Path
import numpy as np

from build_dataset import (build_windows, load_run, wheel_zero_blocks,
                           continuous_signal_stats, WHEEL_CH, WHEEL_COL, IMU_CH,
                           STEP_TRAIN, STEP_TEST, SEG, DT)

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
TEST_FRAC_B = 0.25     # fraccion de VENTANAS de B reservada a test
SEED = 42


def split_B_segments(seg_ids_B, counts_B, test_frac, seed):
    """Reparte tramos ENTEROS de B en train(75%)/test(25%) por numero de ventanas.
    Evita que un tramo gigante acapare el test, pero relaja el umbral si con el
    limite estricto no se alcanza la fraccion objetivo (robusto a distribuciones
    de tramos muy desiguales)."""
    rng = np.random.default_rng(seed)
    total = sum(counts_B.values())
    target = test_frac * total
    # probar umbrales crecientes hasta poder acumular la fraccion objetivo
    for cap in (0.15, 0.25, 0.40, 1.0):
        max_seg = cap * total
        eligible = [s for s in counts_B if counts_B[s] <= max_seg]
        if sum(counts_B[s] for s in eligible) >= target:
            break
    order = rng.permutation(eligible)
    test_segs, acc = set(), 0
    for s in order:
        if acc >= target:
            break
        test_segs.add(int(s)); acc += counts_B[int(s)]
    return test_segs


def main():
    print("=== PARTICION MIXTA (Run A completo + 75% Run B / 25% Run B test) ===\n")
    results = {}
    stats = {}
    for cfg, want_wheel in [("imu", False), ("imuwheel", True)]:
        # --- Run A completo (train), con solapamiento ---
        zblocks_A = wheel_zero_blocks(load_run(BASE, "Run_A")[2]) if want_wheel else None
        XA, yA, segA, tsA, ch, _ = build_windows(BASE, "Run_A", STEP_TRAIN, want_wheel, zblocks_A)
        segA = segA + 1000        # prefijo para unicidad

        # --- Run B: generar con solape (para train) y sin solape (para test) ---
        zblocks_B = wheel_zero_blocks(load_run(BASE, "Run_B")[2]) if want_wheel else None
        XB_ov, yB_ov, segB_ov, tsB_ov, _, _ = build_windows(BASE, "Run_B", STEP_TRAIN, want_wheel, zblocks_B)
        XB_no, yB_no, segB_no, tsB_no, _, _ = build_windows(BASE, "Run_B", STEP_TEST, want_wheel, zblocks_B)

        # reparto de tramos de B (por ventanas del conjunto SIN solape, que define los tramos)
        segs_B = np.unique(segB_no)
        counts_B = {int(s): int(np.sum(segB_no == s)) for s in segs_B}
        test_segs = split_B_segments(segs_B, counts_B, TEST_FRAC_B, SEED)

        # train B = tramos NO en test, tomados del conjunto SOLAPADO
        trainB_mask = ~np.isin(segB_ov, list(test_segs))
        XBt, yBt, segBt = XB_ov[trainB_mask], yB_ov[trainB_mask], segB_ov[trainB_mask] + 2000
        tsBt = tsB_ov[trainB_mask]
        # test B = tramos en test, del conjunto SIN solapar
        testB_mask = np.isin(segB_no, list(test_segs))
        XBte, yBte, segBte = XB_no[testB_mask], yB_no[testB_mask], segB_no[testB_mask] + 2000
        tsBte = tsB_no[testB_mask]

        # --- combinar train: A completo + B-train ---
        Xtr = np.concatenate([XA, XBt]); ytr = np.concatenate([yA, yBt])
        segtr = np.concatenate([segA, segBt])
        tstr = np.concatenate([tsA, tsBt])
        Xte, yte, segte, tste = XBte, yBte, segBte, tsBte

        # --- stats de normalizacion sobre el TRAIN combinado (A + porcion B) ---
        # usamos la senal continua: A completa + tramos de B que van a train
        mA, sA, nA, _ = _stats_run(BASE, "Run_A", want_wheel, None)
        mB, sB, nB, n_excl = _stats_run(BASE, "Run_B", want_wheel, tsBte)  # excluye tramos de test
        print(f"[{cfg:9s}] normalizacion: excluidos {n_excl} tramos DGPS de B "
              f"(tramos de test: {len(test_segs)}) -> deben coincidir")
        mean = (mA * nA + mB * nB) / (nA + nB)
        # std combinada: recompute desde sumas de cuadrados
        # (aprox: recomputamos con la media combinada usando varianzas)
        var = ((sA**2 + mA**2) * nA + (sB**2 + mB**2) * nB) / (nA + nB) - mean**2
        std = np.sqrt(np.clip(var, 1e-16, None)); std[std < 1e-8] = 1.0

        Xtr = (Xtr - mean) / std
        Xte = (Xte - mean) / std
        stats[cfg] = {"mean": mean.tolist(), "std": std.tolist()}

        np.savez_compressed(BASE / f"dataset_{cfg}_mixed_train.npz",
                            X=Xtr, y=ytr, seg_id=segtr, t_start=tstr, channels=np.array(ch))
        np.savez_compressed(BASE / f"dataset_{cfg}_mixed_test.npz",
                            X=Xte, y=yte, seg_id=segte, t_start=tste, channels=np.array(ch))
        n_test_segs = len(test_segs)
        print(f"[{cfg:9s}] train={len(Xtr):6d} vent (A={len(XA)} + B={len(XBt)}) | "
              f"test={len(Xte):5d} vent ({n_test_segs} tramos de B) | canales={len(ch)}")
        # firma del target en test
        mv = np.hypot(yte[:,0], yte[:,1]) > 0.05
        if mv.sum() > 0:
            af, al = np.abs(yte[mv,0]).mean(), np.abs(yte[mv,1]).mean()
            print(f"            test fwd/lat: {af/(af+al)*100:.0f}/{al/(af+al)*100:.0f}  "
                  f"d_fwd={af*100:.1f}cm  d_lat={al*100:.1f}cm  (real, referencia)")
        results[cfg] = (len(Xtr), len(Xte))

    (BASE / "norm_stats_mixed.json").write_text(json.dumps(stats, indent=2))
    print(f"\nGuardado: dataset_*_mixed_{{train,test}}.npz + norm_stats_mixed.json")


def _stats_run(base, run, want_wheel, exclude_ts):
    """Media/std/N de la senal continua de un run, opcionalmente excluyendo tramos.
    Reimplementa continuous_signal_stats devolviendo tambien N para poder combinar.
    exclude_ts: instantes de inicio de las ventanas de test; se excluye todo tramo
    DGPS que contenga alguno. Devuelve (media, std, N, n_tramos_excluidos)."""
    gt, imu, wo = load_run(base, run)
    gt_t = gt[:, 0]; imu_t = imu[:, 0] / 1e9; wo_t = wo[:, 0] / 1e9
    imu_sig = imu[:, 1:7]
    if want_wheel:
        wheel = np.column_stack([np.interp(imu_t, wo_t, wo[:, WHEEL_COL[c]]) for c in WHEEL_CH])
        sig = np.hstack([imu_sig, wheel])
    else:
        sig = imu_sig
    # segmentacion identica a build_windows
    segs, start = [], 0
    for i in range(1, len(gt_t)):
        if gt_t[i]-gt_t[i-1] > SEG:
            segs.append((start,i)); start=i
    segs.append((start,len(gt_t)))
    mask = np.zeros(len(imu_t), dtype=bool)
    n_excl = 0
    for (s,e) in segs:
        t0,t1 = gt_t[s], gt_t[e-1]
        if t1-t0 < 2*DT: continue
        # excluir tramos de test: los que contienen alguna ventana de test
        if exclude_ts is not None and np.any((exclude_ts >= t0) & (exclude_ts <= t1)):
            n_excl += 1
            continue
        mask |= (imu_t >= t0) & (imu_t <= t1)
    sig = sig[mask]
    return sig.mean(axis=0), sig.std(axis=0), len(sig), n_excl


if __name__ == "__main__":
    main()