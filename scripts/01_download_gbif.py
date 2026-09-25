#!/usr/bin/env python3
"""
01_download_gbif.py — Descarga registros de GBIF para Costa Rica por taxón.

Usa la Occurrence Download API (asincrona), NO el endpoint de busqueda:
  - /occurrence/search tiene tope de offset 100 000 registros. Insuficiente.
  - /occurrence/download genera un archivo completo Y un DOI citable,
    que es lo que la propuesta necesita para la seccion de datos.

Requiere una cuenta gratuita en https://www.gbif.org/user/profile
Exportar antes de correr:
    export GBIF_USER=tu_usuario
    export GBIF_PWD=tu_contrasena
    export GBIF_EMAIL=tu@correo.com

Uso:
    python 01_download_gbif.py --taxa aves plantae amphibia insecta
    python 01_download_gbif.py --status            # revisa descargas pendientes
    python 01_download_gbif.py --fetch 0012345-XXX # baja un zip ya listo

Las descargas tardan de minutos a horas segun el tamano. El script guarda
los keys en data/raw/download_keys.json para poder retomar despues.
"""
import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

API = "https://api.gbif.org/v1"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
KEYS_FILE = RAW / "download_keys.json"

# Costa Rica. Se usa el codigo ISO, no un poligono: el bounding box lo
# aplicamos despues en 02_prepare.py, ya con criterio unico para todos.
COUNTRY = "CR"

# taxonKey del backbone de GBIF. Estables y verificables en
# https://www.gbif.org/species/<key>
TAXA = {
    "aves":     {"key": 212,  "name": "Aves"},
    "plantae":  {"key": 6,    "name": "Plantae"},
    "amphibia": {"key": 131,  "name": "Amphibia"},
    "insecta":  {"key": 216,  "name": "Insecta"},
    "mammalia": {"key": 359,  "name": "Mammalia"},
    "reptilia": {"key": 358,  "name": "Reptilia"},
}


def creds():
    u, p, e = (os.environ.get("GBIF_USER"), os.environ.get("GBIF_PWD"),
               os.environ.get("GBIF_EMAIL"))
    if not all([u, p, e]):
        sys.exit("Faltan GBIF_USER, GBIF_PWD y/o GBIF_EMAIL en el entorno.\n"
                 "Crea la cuenta gratis en https://www.gbif.org/user/profile")
    return u, p, e


def predicate(taxon_key, email):
    """Filtros aplicados del lado del servidor.

    Cada filtro tiene una razon:
      - hasCoordinate / hasGeospatialIssue: sin coordenadas utiles no sirve.
      - occurrenceStatus PRESENT: los registros de ausencia no se filtran igual.
      - basisOfRecord: se EXCLUYEN los fosiles y los ejemplares vivos de
        zoologico/jardin botanico, cuyas coordenadas son las de la institucion
        y crearian cumulos artificiales enormes en un solo punto.
    """
    return {
        "creator": os.environ["GBIF_USER"],
        "notification_address": [email],
        "sendNotification": True,
        "format": "SIMPLE_CSV",
        "predicate": {
            "type": "and",
            "predicates": [
                {"type": "equals", "key": "COUNTRY", "value": COUNTRY},
                {"type": "equals", "key": "TAXON_KEY", "value": str(taxon_key)},
                {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
                {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
                {"type": "equals", "key": "OCCURRENCE_STATUS", "value": "PRESENT"},
                {"type": "not", "predicate": {
                    "type": "in", "key": "BASIS_OF_RECORD",
                    "values": ["FOSSIL_SPECIMEN", "LIVING_SPECIMEN"]}},
            ],
        },
    }


def count_first(taxon_key):
    """Cuenta antes de pedir. Evita encolar descargas gigantes por error."""
    r = requests.get(f"{API}/occurrence/search", timeout=60, params={
        "country": COUNTRY, "taxonKey": taxon_key, "hasCoordinate": "true",
        "hasGeospatialIssue": "false", "occurrenceStatus": "PRESENT", "limit": 0})
    r.raise_for_status()
    return r.json().get("count", -1)


def request_download(taxon, email, user, pwd):
    info = TAXA[taxon]
    n = count_first(info["key"])
    print(f"  {info['name']}: ~{n:,} registros con coordenadas")
    r = requests.post(f"{API}/occurrence/download/request",
                      auth=(user, pwd), timeout=120,
                      headers={"Content-Type": "application/json"},
                      json=predicate(info["key"], email))
    if r.status_code not in (200, 201):
        print(f"  ERROR {r.status_code}: {r.text[:300]}")
        return None
    key = r.text.strip()
    print(f"  encolada -> key {key}")
    return {"taxon": taxon, "key": key, "expected_count": n}


def poll(key):
    r = requests.get(f"{API}/occurrence/download/{key}", timeout=60)
    r.raise_for_status()
    return r.json()


def fetch(key, taxon="dataset"):
    meta = poll(key)
    if meta["status"] != "SUCCEEDED":
        print(f"  {key}: estado {meta['status']}, aun no descargable")
        return None
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / f"gbif_cr_{taxon}_{key}.zip"
    if not zip_path.exists():
        print(f"  bajando {meta['size']/1e6:.1f} MB ...")
        with requests.get(meta["downloadLink"], stream=True, timeout=1800) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    out_dir = RAW / f"gbif_cr_{taxon}"
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    # El DOI es obligatorio citarlo en el paper. Se guarda junto a los datos.
    (out_dir / "CITATION.txt").write_text(
        f"GBIF.org ({time.strftime('%d %B %Y')}) GBIF Occurrence Download\n"
        f"https://doi.org/{meta['doi']}\n"
        f"key: {key}\nregistros: {meta.get('totalRecords')}\n", encoding="utf-8")
    print(f"  OK -> {out_dir}  DOI: {meta['doi']}  "
          f"({meta.get('totalRecords'):,} registros)")
    return out_dir


def load_keys():
    return json.loads(KEYS_FILE.read_text()) if KEYS_FILE.exists() else []


def save_keys(keys):
    RAW.mkdir(parents=True, exist_ok=True)
    KEYS_FILE.write_text(json.dumps(keys, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxa", nargs="+", choices=sorted(TAXA), help="taxones a pedir")
    ap.add_argument("--status", action="store_true", help="estado de lo pedido")
    ap.add_argument("--fetch", metavar="KEY", help="bajar un zip ya listo")
    ap.add_argument("--wait", action="store_true",
                    help="esperar y bajar automaticamente al terminar")
    ap.add_argument("--count-only", action="store_true",
                    help="solo contar, sin encolar nada")
    args = ap.parse_args()

    if args.count_only:
        for t, info in TAXA.items():
            print(f"{info['name']:>10}: {count_first(info['key']):>12,}")
        return

    if args.fetch:
        fetch(args.fetch)
        return

    if args.status:
        keys = load_keys()
        if not keys:
            print("No hay descargas registradas.")
            return
        for k in keys:
            m = poll(k["key"])
            print(f"{k['taxon']:>10} {k['key']}  {m['status']:>12}  "
                  f"{m.get('totalRecords', '-')}")
            if m["status"] == "SUCCEEDED":
                fetch(k["key"], k["taxon"])
        return

    if not args.taxa:
        ap.error("indica --taxa, --status, --fetch o --count-only")

    user, pwd, email = creds()
    keys = load_keys()
    for t in args.taxa:
        print(f"[{t}]")
        res = request_download(t, email, user, pwd)
        if res:
            keys = [k for k in keys if k["taxon"] != t] + [res]
            save_keys(keys)
        time.sleep(2)  # cortesia con la API

    print(f"\nKeys guardados en {KEYS_FILE}")
    if args.wait:
        pend = {k["taxon"]: k["key"] for k in keys if k["taxon"] in args.taxa}
        while pend:
            time.sleep(60)
            for t, key in list(pend.items()):
                st = poll(key)["status"]
                print(f"  {t}: {st}")
                if st == "SUCCEEDED":
                    fetch(key, t)
                    del pend[t]
                elif st in ("KILLED", "FAILED", "CANCELLED"):
                    print(f"  {t} termino en {st}")
                    del pend[t]
    else:
        print("Revisa el avance con:  python 01_download_gbif.py --status")


if __name__ == "__main__":
    main()
