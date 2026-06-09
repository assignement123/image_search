#!/usr/bin/env python3
"""
scripts/migrate_efd_37.py

Migration: efd_coeffs → vector(57)
  HARMONICS 10→15, sign fix (A₁=+1), D₁ thay C₁ (eccentricity đúng).

Chạy BÊN TRONG container (khuyến nghị):
    docker exec leaf_app python scripts/migrate_efd_37.py

Chạy ngoài container (DB phải expose port 5433):
    DB_HOST=localhost DB_PORT=5433 python scripts/migrate_efd_37.py

Ước tính: ~3-5 phút (chỉ re-extract EFD, không chạy lại vein/LBP/GLCM/color)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from pgvector.psycopg2 import register_vector

from src.core.preprocess import preprocess_leaf
from src.features.contour import extract_efd

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "leaf_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin")
DATA_DIR = Path(os.getenv("DATA_DIR", "./leaves_data")).resolve()


def migrate():
    print("=" * 55)
    print("  EFD Migration: vector(36) → vector(37)")
    print("=" * 55)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    register_vector(conn)
    cur = conn.cursor()

    # ── Kiểm tra version hiện tại ────────────────────────────────────
    cur.execute("SELECT value FROM meta WHERE key = 'dim_efd'")
    row = cur.fetchone()
    current_dim = int(row[0]) if row else 0
    if current_dim == 57:
        print("\n✅ DB đã ở dim_efd=57, không cần migrate.")
        cur.close(); conn.close(); return

    print(f"\n  dim_efd hiện tại: {current_dim} → sẽ migrate lên 37\n")

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Schema migration
    # ─────────────────────────────────────────────────────────────────
    print("[ Step 1 ] Schema migration...")
    cur.execute("DROP INDEX IF EXISTS idx_efd_hnsw")
    cur.execute("ALTER TABLE leaf_collection DROP COLUMN IF EXISTS efd_coeffs_old")
    cur.execute("ALTER TABLE leaf_collection RENAME COLUMN efd_coeffs TO efd_coeffs_old")
    cur.execute("ALTER TABLE leaf_collection ADD COLUMN efd_coeffs vector(57)")
    conn.commit()
    print("  ✓ efd_coeffs (36) → efd_coeffs_old, thêm efd_coeffs vector(37)\n")

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Re-extract EFD only
    # ─────────────────────────────────────────────────────────────────
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted(p for p in DATA_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS)
    total = len(all_images)
    print(f"[ Step 2 ] Re-extracting EFD cho {total} ảnh...")
    print(f"  DATA_DIR = {DATA_DIR}\n")

    ok = errors = 0
    for i, img_path in enumerate(all_images, 1):
        filename = img_path.name
        try:
            _, contour, _, _, _ = preprocess_leaf(str(img_path))
            efd = extract_efd(contour)          # 37 dims (C₁ + harmonics 2..10)

            cur.execute(
                "UPDATE leaf_collection SET efd_coeffs = %s WHERE filename = %s",
                (efd, filename)
            )
            conn.commit()
            ok += 1
        except Exception as e:
            conn.rollback()
            errors += 1
            print(f"  ⚠  {filename}: {e}")

        if i % 200 == 0 or i == total:
            print(f"  [{i:4d}/{total}] ✓ {ok}  ✗ {errors}")

    print(f"\n  Xong: {ok} updated, {errors} lỗi\n")

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Cleanup + recreate HNSW index
    # ─────────────────────────────────────────────────────────────────
    print("[ Step 3 ] Cleanup + recreate HNSW index...")

    cur.execute("SELECT COUNT(*) FROM leaf_collection WHERE efd_coeffs IS NULL")
    nulls = cur.fetchone()[0]
    if nulls > 0:
        print(f"  ⚠  WARNING: còn {nulls} row NULL — kiểm tra lỗi ở Step 2")

    cur.execute("ALTER TABLE leaf_collection DROP COLUMN efd_coeffs_old")
    cur.execute("""
        CREATE INDEX idx_efd_hnsw ON leaf_collection
        USING hnsw (efd_coeffs vector_cosine_ops)
    """)
    cur.execute("UPDATE meta SET value = '57' WHERE key = 'dim_efd'")
    conn.commit()
    print("  ✓ Old column dropped, HNSW index recreated\n")

    cur.close()
    conn.close()
    print("=" * 55)
    print("  ✅ Migration hoàn thành! dim_efd = 37")
    print("=" * 55)


if __name__ == "__main__":
    migrate()
