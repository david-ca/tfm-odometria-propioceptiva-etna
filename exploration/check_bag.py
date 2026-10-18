"""
check_bag.py — Verifica integridad y lista topics del bag de un run.
Uso:  python check_bag.py Run_A
      python check_bag.py Run_B
Detecta dinamicamente el campo de conteo por connection y verifica su tipo
antes de formatear (no asume que se llame 'count').
"""
import sys
from pathlib import Path
from rosbags.highlevel import AnyReader

if len(sys.argv) < 2:
    print("Uso: python check_bag.py <NOMBRE_DEL_RUN>")
    sys.exit(1)

RUN = sys.argv[1]
RUN_DIR = Path(r"C:\Users\Usuario\Desktop\TFM") / RUN

bags = list(RUN_DIR.glob("*.bag"))
if not bags:
    print(f"No se encontró ningún .bag en {RUN_DIR}")
    sys.exit(1)
if len(bags) > 1:
    print(f"AVISO: hay varios .bag, uso el primero: {[b.name for b in bags]}")
BAG = bags[0]

print(f"=== RUN: {RUN} ===")
print(f"Bag: {BAG.name}  ({BAG.stat().st_size/1e9:.1f} GB)\n")

def get_count(conn):
    """Devuelve el conteo de mensajes de una connection como int, o None.
    Prueba varios nombres de atributo y verifica que sea un entero real."""
    for cand in ("msgcount", "count", "message_count"):
        val = getattr(conn, cand, None)
        if isinstance(val, int):
            return val
    return None

try:
    with AnyReader([BAG]) as reader:
        print(f"  message_count: {getattr(reader, 'message_count', 'n/d')}")
        print(f"  duration (ns): {getattr(reader, 'duration', 'n/d')}")
        print(f"  start_time:    {getattr(reader, 'start_time', 'n/d')}")
        print(f"  end_time:      {getattr(reader, 'end_time', 'n/d')}")
        conns = list(reader.connections)
        print(f"  connections:   {len(conns)}\n")

        print("=== TOPICS ===")
        rows = []
        for c in conns:
            rows.append((c.topic, c.msgtype, get_count(c)))
        rows.sort(key=lambda r: r[0])
        for topic, msgtype, n in rows:
            n_str = f"{n:>10}" if isinstance(n, int) else "     (n/d)"
            print(f"  {n_str}  {topic}   [{msgtype}]")

        print("\n=== INTEGRIDAD ===")
        print("  Índice leído sin errores -> bag NO truncado, compresión soportada.")
except Exception as e:
    print(f"\nFALLO AL LEER EL BAG: {type(e).__name__}: {e}")
    sys.exit(1)

print("\nHecho.")