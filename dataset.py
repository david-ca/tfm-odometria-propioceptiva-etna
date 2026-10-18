"""
dataset.py -- Carga de los .npz del Bloque 5 y particion train/val por TRAMOS.

- Carga dataset_<cfg>_<run>.npz (X ya normalizado, y, seg_id, t_start, channels).
- Split train/val SOLO sobre Run_B (entrenamiento), reservando tramos ENTEROS
  hasta acumular ~VAL_FRAC de las VENTANAS (no de los tramos, porque los tramos
  tienen tamanos muy distintos). Semilla fija para reproducibilidad.
- Run_A (test) se carga entero, nunca se parte.

No hay fuga: ningun tramo aparece a la vez en train y val (se separan por seg_id,
y las ventanas nunca cruzan tramos por construccion del Bloque 5).
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

VAL_FRAC = 0.18      # fraccion objetivo de VENTANAS para validacion
SEED = 42


def load_npz(base, cfg, run):
    p = Path(base) / run / f"dataset_{cfg}_{run}.npz"
    d = np.load(p, allow_pickle=True)
    return d["X"], d["y"], d["seg_id"], list(d["channels"])


def split_by_segments(seg_id, val_frac=VAL_FRAC, seed=SEED):
    """
    Reserva tramos ENTEROS para validacion buscando ~val_frac de las VENTANAS,
    pero garantizando VARIEDAD: los tramos muy grandes (> 5% del total de ventanas)
    se excluyen de la validacion y se quedan siempre en train, para que la val no
    quede dominada por uno o dos tramos gigantes. Entre los tramos elegibles se
    seleccionan al azar (semilla fija) hasta acumular la fraccion objetivo.
    """
    rng = np.random.default_rng(seed)
    segs = np.unique(seg_id)
    counts = {int(s): int(np.sum(seg_id == s)) for s in segs}
    total = len(seg_id)

    max_seg = 0.05 * total          # ningun tramo de val supera el 5% del total
    eligible = [s for s in counts if counts[s] <= max_seg]
    # salvaguarda: si el umbral estricto deja sin candidatos (pocos tramos o muy
    # desiguales), relajarlo progresivamente hasta tener con que formar validacion
    for cap in (0.05, 0.10, 0.20, 0.40, 1.0):
        eligible = [s for s in counts if counts[s] <= cap * total]
        if sum(counts[s] for s in eligible) >= val_frac * total:
            break

    order = rng.permutation(eligible)
    val_segs, acc = set(), 0
    for s in order:
        if acc >= val_frac * total:
            break
        val_segs.add(int(s)); acc += counts[int(s)]
    val_mask = np.isin(seg_id, list(val_segs))
    return ~val_mask, val_mask, sorted(val_segs)


def make_loaders(base, cfg, batch_size=128, val_frac=VAL_FRAC, seed=SEED):
    Xb, yb, segb, channels = load_npz(base, cfg, "Run_B")
    tr_mask, va_mask, val_segs = split_by_segments(segb, val_frac, seed)

    def to_ds(X, y):
        return TensorDataset(torch.tensor(X, dtype=torch.float32),
                             torch.tensor(y, dtype=torch.float32))

    tr = to_ds(Xb[tr_mask], yb[tr_mask])
    va = to_ds(Xb[va_mask], yb[va_mask])
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)

    info = {
        "channels": channels, "n_channels": Xb.shape[2],
        "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum()),
        "val_segments": val_segs, "n_train_segs": len(np.unique(segb[tr_mask])),
        "n_val_segs": len(val_segs),
    }
    return tr_loader, va_loader, info


def load_test(base, cfg, batch_size=256):
    Xa, ya, sega, channels = load_npz(base, cfg, "Run_A")
    ds = TensorDataset(torch.tensor(Xa, dtype=torch.float32),
                       torch.tensor(ya, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, {"n_test": len(Xa), "seg_id": sega, "channels": channels}


# ---------- Particion MIXTA (Run A completo + porcion B / porcion B test) ----------

def _load_mixed(base, cfg, split):
    """Carga dataset_<cfg>_mixed_<split>.npz  (split = 'train' o 'test')."""
    p = Path(base) / f"dataset_{cfg}_mixed_{split}.npz"
    d = np.load(p, allow_pickle=True)
    return d["X"], d["y"], d["seg_id"], list(d["channels"])


def make_loaders_mixed(base, cfg, batch_size=128, val_frac=VAL_FRAC, seed=SEED):
    """Como make_loaders pero sobre el train MIXTO; la validacion se separa por
    tramos del propio train mixto (sin tocar el test mixto)."""
    X, y, seg, channels = _load_mixed(base, cfg, "train")
    tr_mask, va_mask, val_segs = split_by_segments(seg, val_frac, seed)

    def to_ds(Xa, ya):
        return TensorDataset(torch.tensor(Xa, dtype=torch.float32),
                             torch.tensor(ya, dtype=torch.float32))

    tr = to_ds(X[tr_mask], y[tr_mask])
    va = to_ds(X[va_mask], y[va_mask])
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    info = {"channels": channels, "n_channels": X.shape[2],
            "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum()),
            "n_val_segs": len(val_segs)}
    return tr_loader, va_loader, info


def load_test_mixed(base, cfg, batch_size=256):
    X, y, seg, channels = _load_mixed(base, cfg, "test")
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, {"n_test": len(X), "seg_id": seg, "channels": channels}