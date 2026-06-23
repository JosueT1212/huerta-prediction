"""
Replay Excel sensor data to /ingest/sensors for pipeline testing.

Usage:
    INGEST_TOKEN=your-token python3.11 scripts/mock_ingest.py --inv 3 --host http://localhost:8000
"""
import argparse
import os
import sys
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"

SENSOR_FILES = {
    3: DATA / "Variables internas Invernadero 3.xlsx",
    4: DATA / "Variables internas invernadero 4.xlsx",
}

COLUMN_MAP = {
    "Temperatura Promedio":       "temp_prom_int",
    "Temperatura Mínima":         "temp_min_int",
    "Temperatura Máxima":         "temp_max_int",
    "Humedad Relativa Promedio":  "hr_prom_int",
    "CO2":                        "co2_ppm",
}


def load_sensor_data(inv: int) -> pd.DataFrame:
    xl = pd.ExcelFile(SENSOR_FILES[inv])
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df.columns = [str(c).strip() for c in df.columns]
        if "Fecha" not in df.columns:
            continue
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        frames.append(df.dropna(subset=["Fecha"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inv", type=int, choices=[3, 4], required=True)
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("INGEST_TOKEN", "")
    if not token and not args.dry_run:
        print("ERROR: set INGEST_TOKEN env var", file=sys.stderr)
        sys.exit(1)

    df = load_sensor_data(args.inv)
    if df.empty:
        print(f"No data for inv {args.inv}")
        sys.exit(1)

    sent = 0
    for _, row in df.iterrows():
        ts = row["Fecha"].isoformat()
        for col, sensor_name in COLUMN_MAP.items():
            if col not in row or pd.isna(row[col]):
                continue
            if not args.dry_run:
                requests.post(
                    f"{args.host}/ingest/sensors",
                    json={"greenhouse_id": args.inv, "sensor_name": sensor_name,
                          "value": float(row[col]), "recorded_at": ts},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10,
                ).raise_for_status()
            sent += 1

    print(f"{'[DRY RUN] Would send' if args.dry_run else 'Sent'} {sent} readings for inv {args.inv}")


if __name__ == "__main__":
    main()
