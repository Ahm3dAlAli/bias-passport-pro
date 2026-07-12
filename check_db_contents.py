#!/usr/bin/env python3
"""Check database contents to verify image counts."""

import sqlite3
from pathlib import Path

results_dir = Path("results/single_runs_35k")

print("=" * 70)
print("Database Contents Check")
print("=" * 70)
print()

for db_path in sorted(results_dir.glob("*.db")):
    print(f"File: {db_path.name}")
    print(f"Size: {db_path.stat().st_size / (1024*1024):.1f} MB")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get table schema
        cursor.execute("PRAGMA table_info(probe_results)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        print(f"  Columns: {', '.join(column_names)}")

        # Get total rows
        cursor.execute("SELECT COUNT(*) FROM probe_results")
        total_rows = cursor.fetchone()[0]

        # Get unique images (check if image_id exists)
        if 'image_id' in column_names:
            cursor.execute("SELECT COUNT(DISTINCT image_id) FROM probe_results")
            unique_images = cursor.fetchone()[0]
        else:
            unique_images = "N/A"

        # Get unique probes
        if 'probe_id' in column_names:
            cursor.execute("SELECT COUNT(DISTINCT probe_id) FROM probe_results")
            unique_probes = cursor.fetchone()[0]
        else:
            unique_probes = "N/A"

        conn.close()

        print(f"  Total rows: {total_rows:,}")
        if unique_images != "N/A":
            print(f"  Unique images: {unique_images:,}")
        if unique_probes != "N/A":
            print(f"  Unique probes: {unique_probes}")
            if unique_images != "N/A":
                print(f"  Expected rows: {unique_images} × {unique_probes} = {unique_images * unique_probes:,}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

    print()

print("=" * 70)
