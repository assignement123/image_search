#!/usr/bin/env python3
"""
scripts/migrate_vein_17.py

Migration: vein_features vector(9) → vector(17)
  - Rotation-invariant: góc vein normalized về trục chính lá
  - 16-bin histogram (từ 8-bin)

Chạy BÊN TRONG container:
    docker exec leaf_app python scripts/migrate_vein_17.py

Ước tính: ~5-7 phút (vein extraction nặng hơn EFD)
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from pgvector.psycopg2 import register_vector

from src.core.preprocess import preprocess_leaf
from src.features.vein import extract_vein_features

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "leaf_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin")
DATA_DIR = Path(os.getenv("DATA_DIR", "./leaves_data")).resolve()


def migrate():
    print("=" * 55)
    print("  Vein Migration: vector(9) → vector(17)")
    print("  rotation-invariant + 16-bin histogram")
    print("=" * 55)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    register_vector(conn)
    cur = conn.cursor()

    cur.execute("SELECT value FROM meta WHERE key = 'dim_vein'")
    row = cur.fetchone()
    current_dim = int(row[0]) if row else 0
    if current_dim == 17:
        print("\n✅ DB đã ở dim_vein=17, không cần migrate.")
        cur.close(); conn.close(); return

    print(f"\n  dim_vein hiện tại: {current_dim} → migrate lên 17\n")

    # ── Step 1: Schema migration ─────────────────────────────────────
    print("[ Step 1 ] Schema migration...")
    cur.execute("DROP INDEX IF EXISTS idx_vein_hnsw")
    cur.execute("ALTER TABLE leaf_collection DROP COLUMN IF EXISTS vein_features_old")
    cur.execute("ALTER TABLE leaf_collection RENAME COLUMN vein_features TO vein_features_old")
    cur.execute("ALTER TABLE leaf_collection ADD COLUMN vein_features vector(17)")
    conn.commit()
    print("  ✓ Schema migrated\n")

    # ── Step 2: Re-extract vein only ─────────────────────────────────
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted(p for p in DATA_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS)
    total = len(all_images)
    print(f"[ Step 2 ] Re-extracting vein cho {total} ảnh...")

    BATCH = 50   # commit mỗi 50 ảnh
    ok = errors = 0
    for i, img_path in enumerate(all_images, 1):
        filename = img_path.name
        try:
            img, _, _, mask, leaf_area = preprocess_leaf(str(img_path))
            vein = extract_vein_features(img, mask, leaf_area)

            cur.execute(
                "UPDATE leaf_collection SET vein_features = %s WHERE filename = %s",
                (vein, filename)
            )
            ok += 1
        except Exception as e:
            errors += 1
            print(f"  ⚠  {filename}: {e}")

        if i % BATCH == 0:
            conn.commit()
        if i % 200 == 0 or i == total:
            conn.commit()   # đảm bảo flush
            print(f"  [{i:4d}/{total}] ✓ {ok}  ✗ {errors}")

    print(f"\n  Xong: {ok} updated, {errors} lỗi\n")

    # ── Step 3: Cleanup + recreate HNSW index ────────────────────────
    print("[ Step 3 ] Cleanup + recreate HNSW index...")
    cur.execute("SELECT COUNT(*) FROM leaf_collection WHERE vein_features IS NULL")
    nulls = cur.fetchone()[0]
    if nulls > 0:
        print(f"  ⚠  WARNING: còn {nulls} row NULL")

    cur.execute("ALTER TABLE leaf_collection DROP COLUMN IF EXISTS vein_features_old")
    cur.execute("""
        CREATE INDEX idx_vein_hnsw ON leaf_collection
        USING hnsw (vein_features vector_cosine_ops)
    """)
    cur.execute("UPDATE meta SET value = '17' WHERE key = 'dim_vein'")
    conn.commit()
    print("  ✓ Done\n")

    cur.close()
    conn.close()
    print("=" * 55)
    print("  ✅ Migration hoàn thành! dim_vein = 17")
    print("=" * 55)


if __name__ == "__main__":
    migrate()
