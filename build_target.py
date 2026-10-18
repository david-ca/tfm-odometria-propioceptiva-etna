"""
build_target.py -- Construccion del TARGET DE PRODUCCION del Bloque 5.

Para cada intervalo consecutivo (no solapado) de 1.0 s dentro de un tramo
continuo del ground truth, calcula el incremento de posicion 3D expresado
RELATIVO al propio movimiento del rover:

    d_fwd : avance a lo largo de la direccion de movimiento previa
    d_lat : deslizamiento lateral (perpendicular horizontal)  <- senal valiosa
    d_up  : componente vertical (eje z global, anclado a gravedad)

Base local (Opcion 2, ya validada en verify_option2.py):
    forward = desplazamiento GPS del intervalo ANTERIOR (course previo), normalizado.
    Legitimo SOLO para construir el target: el GPS es ground truth de entrenamiento,
    nunca entrada del modelo (coherente con el alcance, seccion 2.3 del TFM).

Decisiones de diseno (acordadas con David):
  - Intervalos de 1.0 s NO solapados (particion limpia del tramo).
  - t+1.0 s se fija por INTERPOLACION LINEAL de la posicion (intervalos de
    duracion exacta -> componentes comparables entre si).
  - Umbral de segmentacion 2.0 s: ningun intervalo cruza un hueco de GT > 2 s.
  - El PRIMER intervalo de cada tramo se descarta (no tiene 'forward' previo).
  - Tramos recomputados aqui desde el propio CSV (no se asumen).

Salida:  target_<RUN>.csv con columnas
    t_start, t_end, seg_id, d_fwd, d_lat, d_up, dist_2d, dt_real

Uso:  python build_target.py Run_A
      python build_target.py Run_A --base /ruta/a/extracted
"""
import sys, csv, argparse
from pathlib import Path
import numpy as np

DT = 1.0        # duracion del intervalo objetivo (s)
SEG = 2.0       # umbral de segmentacion: cortar en huecos de GT > SEG (s)
MOVE_EPS = 0.05 # m: movimiento previo minimo para definir un 'forward' fiable


def load_gt(path):
    """Carga ground_truth.csv (timestamp_s, x, y, z). Verifica cabecera."""
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        expected = ["timestamp_s", "x", "y", "z"]
        if header[:4] != expected:
            raise ValueError(
                f"Cabecera inesperada en {path}: {header[:4]} (esperaba {expected})"
            )
        data = np.array([[float(v) for v in row[:4]] for row in r])
    # ordenar por tiempo por seguridad
    data = data[np.argsort(data[:, 0])]
    return data


def segment(t, seg_thr):
    """Devuelve lista de (inicio, fin) de tramos continuos (fin exclusivo)."""
    segs, start = [], 0
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > seg_thr:
            segs.append((start, i))
            start = i
    segs.append((start, len(t)))
    return segs


def build(run, base):
    gt = load_gt(base / run / "ground_truth.csv")
    t = gt[:, 0]
    x, y, z = gt[:, 1], gt[:, 2], gt[:, 3]

    def pos(ti):
        # interpolacion lineal de la posicion en el instante ti
        return np.array([np.interp(ti, t, x),
                         np.interp(ti, t, y),
                         np.interp(ti, t, z)])

    segs = segment(t, SEG)

    rows = []
    n_seg_used = 0
    n_first_dropped = 0
    n_lowmove = 0
    # --- instrumentacion de distancia (no afecta al target) ---
    dist_lowmove = 0.0     # dist 2D de intervalos descartados por falta de forward
    dist_firstdrop = 0.0   # dist 2D de los primeros intervalos de tramo descartados
    dist_gt_full = 0.0     # dist 2D GT integrada a ~10 Hz en los tramos usables
    for (s, e) in segs:
        t0, t1 = t[s], t[e - 1]
        if t1 - t0 < 2 * DT:      # tramo demasiado corto para 2 intervalos
            continue
        n_seg_used += 1
        seg_id = n_seg_used
        # distancia GT integrada punto a punto (10 Hz) dentro del tramo usable
        dx = np.diff(x[s:e]); dy = np.diff(y[s:e])
        dist_gt_full += float(np.sum(np.hypot(dx, dy)))

        # bordes de intervalos consecutivos no solapados dentro del tramo
        edges = np.arange(t0, t1 + 1e-9, DT)
        # necesitamos al menos 3 bordes: [b0,b1,b2] -> intervalo previo [b0,b1]
        # y primer intervalo emitible [b1,b2]. El intervalo [b0,b1] es el que
        # se descarta (primer intervalo del tramo, sin 'forward' previo).
        if len(edges) < 3:
            continue

        p = [pos(b) for b in edges]  # posiciones interpoladas en cada borde

        # el primer intervalo del tramo [b0,b1] se descarta (sin forward previo)
        dist_firstdrop += float(np.hypot(*(p[1][:2] - p[0][:2])))

        first_emitted = True
        for k in range(1, len(edges) - 1):
            d_prev = p[k] - p[k - 1]          # desplazamiento intervalo anterior
            d_now = p[k + 1] - p[k]           # desplazamiento intervalo actual
            dprev_h = d_prev[:2]
            if np.hypot(*dprev_h) < MOVE_EPS:
                # sin forward fiable: no se puede definir la base local
                n_lowmove += 1
                dist_lowmove += float(np.hypot(*d_now[:2]))
                continue

            heading = np.arctan2(dprev_h[1], dprev_h[0])
            c, srot = np.cos(heading), np.sin(heading)
            # proyeccion horizontal de d_now sobre la base (forward, lateral)
            d_fwd = c * d_now[0] + srot * d_now[1]
            d_lat = -srot * d_now[0] + c * d_now[1]
            d_up = d_now[2]

            if first_emitted:
                n_first_dropped += 0  # el primer intervalo [b0,b1] ya no se emite
                first_emitted = False

            rows.append([edges[k], edges[k + 1], seg_id,
                         d_fwd, d_lat, d_up,
                         float(np.hypot(*d_now[:2])), DT])

    rows = np.array(rows) if rows else np.empty((0, 8))
    dist_info = {
        "lowmove": dist_lowmove,
        "firstdrop": dist_firstdrop,
        "gt_full": dist_gt_full,
    }
    return rows, n_seg_used, n_lowmove, len(segs), dist_info


def summarize(run, rows, n_seg, n_lowmove, n_segs_total, dist_info=None):
    print(f"=== TARGET DE PRODUCCION -- {run} ===")
    print(f"  tramos totales detectados : {n_segs_total}")
    print(f"  tramos usados (>= 2 s)     : {n_seg}")
    print(f"  intervalos emitidos        : {len(rows)}")
    print(f"  intervalos sin forward     : {n_lowmove} (movimiento previo < {MOVE_EPS*100:.0f} cm)")
    if len(rows) == 0:
        print("  (sin filas)"); return
    fwd, lat, up = rows[:, 3], rows[:, 4], rows[:, 5]
    mv = np.hypot(rows[:, 3], rows[:, 4]) > MOVE_EPS
    afwd, alat = np.abs(fwd[mv]).mean(), np.abs(lat[mv]).mean()
    tot = afwd + alat
    print(f"  reparto medio: fwd={afwd/tot*100:.0f}%  lat={alat/tot*100:.0f}%")
    print(f"  magnitudes medias (en movimiento): "
          f"d_fwd={afwd*100:.1f}cm  d_lat={alat*100:.1f}cm  |d_up|={np.abs(up[mv]).mean()*100:.1f}cm")
    print(f"  distancia 2D total (sum |d_now|): {rows[:,6].sum():.1f} m")
    # consistencia temporal 1a vs 2a mitad (misma prueba que la validacion)
    half = len(rows) // 2
    f1 = np.abs(fwd[:half][mv[:half]]).mean()
    f2 = np.abs(fwd[half:][mv[half:]]).mean()
    dev = abs(f1 - f2) / max(f1, f2) * 100
    print(f"  consistencia temporal d_fwd: 1a={f1*100:.1f}cm 2a={f2*100:.1f}cm  "
          f"(desviacion {dev:.0f}%)")

    if dist_info is not None:
        d_emit = rows[:, 6].sum()
        d_low = dist_info["lowmove"]
        d_first = dist_info["firstdrop"]
        d_poly = d_emit + d_low + d_first   # toda la poligonal de saltos de 1 s
        d_gt = dist_info["gt_full"]         # GT integrada a ~10 Hz
        print("  --- balance de distancia 2D ---")
        print(f"    emitida (en target)        : {d_emit:.1f} m")
        print(f"    descartada sin forward     : {d_low:.1f} m")
        print(f"    descartada 1er intervalo   : {d_first:.1f} m")
        print(f"    total poligonal 1 s        : {d_poly:.1f} m")
        print(f"    GT integrada 10 Hz (usable): {d_gt:.1f} m")
        print(f"    corte de esquinas (10Hz-1s): {d_gt - d_poly:.1f} m "
              f"({(d_gt-d_poly)/d_gt*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--base", default=r"C:\Users\Usuario\Desktop\TFM\extracted")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = Path(args.base)
    rows, n_seg, n_lowmove, n_segs_total, dist_info = build(args.run, base)
    summarize(args.run, rows, n_seg, n_lowmove, n_segs_total, dist_info)

    out = Path(args.out) if args.out else base / args.run / f"target_{args.run}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start", "t_end", "seg_id", "d_fwd", "d_lat", "d_up",
                    "dist_2d", "dt_real"])
        for row in rows:
            w.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}", int(row[2]),
                        f"{row[3]:.6f}", f"{row[4]:.6f}", f"{row[5]:.6f}",
                        f"{row[6]:.6f}", f"{row[7]:.3f}"])
    print(f"\n  -> escrito: {out}")


if __name__ == "__main__":
    main()