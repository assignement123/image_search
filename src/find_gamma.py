import psycopg2
import psycopg2.extras
import numpy as np
import math
import random
from collections import defaultdict
import json  # 👈 1. THÊM THƯ VIỆN NÀY ĐỂ CHUYỂN ĐỔI CHUỖI THÀNH MẢNG

# ==========================================
# CẤU HÌNH THÔNG SỐ TOÁN HỌC
# ==========================================
TARGET_SCORE = 0.2
NUM_SAMPLE_PAIRS = 5000 

def connect_db():
    return psycopg2.connect(host="localhost", port=5432, dbname="leaf_db", user="admin", password="admin")

def chi_square_distance(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return 0.5 * np.sum(((v1 - v2) ** 2) / (v1 + v2 + 1e-10))

def euclidean_distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))

def main():
    print("⏳ Đang kết nối Database và tải dữ liệu...")
    conn = connect_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT species, efd_coeffs, morphology_stats, lbp_hist, glcm_stats, color_moments, vein_features 
        FROM leaf_collection
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("❌ Database rỗng. Vui lòng trích xuất đặc trưng trước!")
        return

    # =========================================================
    # 👇 2. ĐOẠN ĐƯỢC SỬA LỖI NẰM Ở ĐÂY 👇
    # =========================================================
    species_dict = defaultdict(list)
    
    # Danh sách các cột chứa mảng (vector) cần chuyển đổi
    vector_columns = [
        "efd_coeffs", "morphology_stats", "lbp_hist", 
        "glcm_stats", "color_moments", "vein_features"
    ]
    
    for row in rows:
        for col in vector_columns:
            # Nếu dữ liệu lấy ra từ DB đang bị hiểu là chuỗi (string)
            if isinstance(row[col], str):
                try:
                    # Chuyển chuỗi "[0.1, 0.2]" thành mảng thực thụ
                    row[col] = json.loads(row[col])
                except json.JSONDecodeError:
                    # Đề phòng chuỗi lưu trong DB không chuẩn JSON (ví dụ dùng nháy đơn)
                    import ast
                    row[col] = ast.literal_eval(row[col])
        
        # Nhóm dữ liệu đã sạch vào dictionary
        species_dict[row["species"]].append(row)
    # =========================================================
    # 👆 KẾT THÚC ĐOẠN SỬA LỖI 👆
    # =========================================================
    
    species_list = list(species_dict.keys())
    if len(species_list) < 2:
        print("❌ Cần ít nhất 2 loài lá khác nhau trong Database để tính toán phân tách!")
        return

    print(f"✅ Đã tải {len(rows)} chiếc lá thuộc {len(species_list)} loài.")
    print(f"⏳ Đang tạo ngẫu nhiên {NUM_SAMPLE_PAIRS} cặp lá KHÁC LOÀI để đo khoảng cách...")

    distances = {
        "efd": [], "morph": [], "lbp": [], 
        "glcm": [], "color": [], "vein": []
    }

    for _ in range(NUM_SAMPLE_PAIRS):
        sp1, sp2 = random.sample(species_list, 2)
        leaf1 = random.choice(species_dict[sp1])
        leaf2 = random.choice(species_dict[sp2])

        distances["efd"].append(euclidean_distance(leaf1["efd_coeffs"], leaf2["efd_coeffs"]))
        distances["morph"].append(euclidean_distance(leaf1["morphology_stats"], leaf2["morphology_stats"]))
        distances["glcm"].append(euclidean_distance(leaf1["glcm_stats"], leaf2["glcm_stats"]))
        distances["color"].append(euclidean_distance(leaf1["color_moments"], leaf2["color_moments"]))
        distances["vein"].append(euclidean_distance(leaf1["vein_features"], leaf2["vein_features"]))
        distances["lbp"].append(chi_square_distance(leaf1["lbp_hist"], leaf2["lbp_hist"]))

    print("-" * 50)
    print(f"🎯 KẾT QUẢ TÍNH TOÁN HỆ SỐ GAMMA (Target Score = {TARGET_SCORE})")
    print("-" * 50)
    print(f"{'Đặc trưng':<15} | {'K/cách TB (d)':<15} | {'Hệ số Gamma (γ)':<15}")
    print("-" * 50)

    final_gammas = {}

    for feature_name, dist_list in distances.items():
        avg_d = np.mean(dist_list)
        if avg_d > 1e-6:
            gamma = -math.log(TARGET_SCORE) / avg_d
        else:
            gamma = 1.0 

        final_gammas[feature_name] = round(gamma, 4)
        print(f"{feature_name.upper():<15} | {avg_d:<15.4f} | {gamma:<15.4f}")

    print("-" * 50)
    print("\n💡 HÃY COPY ĐOẠN CODE DƯỚI ĐÂY DÁN VÀO API SEARCH CỦA BẠN:\n")
    print(f"""
            s_efd   = distance_to_score(d_efd, gamma={final_gammas['efd']})
            s_morph = distance_to_score(d_morph, gamma={final_gammas['morph']})
            s_glcm  = distance_to_score(d_glcm, gamma={final_gammas['glcm']})
            s_color = distance_to_score(d_color, gamma={final_gammas['color']})
            s_vein  = distance_to_score(d_vein, gamma={final_gammas['vein']})
            s_lbp   = distance_to_score(d_lbp, gamma={final_gammas['lbp']})
    """)

if __name__ == "__main__":
    main()