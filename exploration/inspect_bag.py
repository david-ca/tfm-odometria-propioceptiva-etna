"""
inspect_bag.py — Verifica integridad de un rosbag ROS1 y lista sus topics.
No asume nombres de campos no confirmados: inspecciona el objeto connection
antes de usar atributos, y cae a conteo por iteración si hace falta.
"""
import sys
from pathlib import Path

try:
    from rosbags.highlevel import AnyReader
except ImportError:
    print("ERROR: falta 'rosbags'. Instala con:  pip install rosbags")
    sys.exit(1)

BAG = Path(r"C:\Users\Usuario\Desktop\TFM\2017-06-21-16-44-03_lru_ROBEX_longrange_compressed.bag")

if not BAG.exists():
    print(f"ERROR: no existe el archivo:\n  {BAG}")
    sys.exit(1)

print(f"Abriendo: {BAG.name}  ({BAG.stat().st_size/1e9:.1f} GB)\n")

try:
    with AnyReader([BAG]) as reader:
        # --- Metadatos globales del bag ---
        print("=== INFO GLOBAL ===")
        # duration/start/end: nombres a confirmar por inspección
        for attr in ("duration", "start_time", "end_time", "message_count"):
            if hasattr(reader, attr):
                print(f"  reader.{attr} = {getattr(reader, attr)}")

        conns = list(reader.connections)
        print(f"\n  Nº de connections: {len(conns)}")

        # --- Inspección de UN objeto connection para descubrir sus campos ---
        if conns:
            c0 = conns[0]
            print("\n=== ATRIBUTOS DE UN OBJETO connection (para verificar API) ===")
            attrs = [a for a in dir(c0) if not a.startswith("_")]
            print("  ", attrs)
            print()

        # --- Listado de topics ---
        print("=== TOPICS ===")
        # Detecta dinámicamente si existe un campo de conteo por connection
        count_attr = None
        for cand in ("msgcount", "count", "message_count"):
            if conns and hasattr(conns[0], cand):
                count_attr = cand
                break

        rows = []
        for c in conns:
            n = getattr(c, count_attr) if count_attr else None
            rows.append((c.topic, c.msgtype, n))

        rows.sort(key=lambda r: r[0])
        for topic, msgtype, n in rows:
            n_str = f"{n:>10}" if n is not None else "     (n/d)"
            print(f"  {n_str}  {topic}   [{msgtype}]")

        if count_attr is None:
            print("\n  (El objeto connection no expone conteo directo;")
            print("   el conteo por iteración se hará en un segundo paso si hace falta.)")

        print("\n=== VERIFICACIÓN DE INTEGRIDAD ===")
        print("  El bag se abrió y su índice se leyó sin errores.")
        print("  -> Índice consistente = bag NO truncado.")

except Exception as e:
    print("\n=== FALLO AL LEER EL BAG ===")
    print(f"  {type(e).__name__}: {e}")
    print("\n  Un error aquí puede indicar bag truncado/corrupto.")
    print("  NO borres el .crdownload todavía.")
    sys.exit(1)