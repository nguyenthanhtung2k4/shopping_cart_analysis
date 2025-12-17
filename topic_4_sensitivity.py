# -*- coding: utf-8 -*-
"""
Script chạy Chủ đề 4: Phân tích độ nhạy tham số (Parameter Sensitivity Analysis)
Tự động thử nghiệm các mức min_support khác nhau để tìm ngưỡng tối ưu.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Đảm bảo Python nhìn thấy thư mục src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import các class từ thư viện dự án (theo cấu trúc trong Lab)
    from src.apriori_library import DataCleaner, BasketPreparer, AssociationRulesMiner
except ImportError:
    # Fallback nếu cấu trúc thư mục khác (ví dụ file thư viện để cùng cấp)
    try:
        from apriori_library import DataCleaner, BasketPreparer, AssociationRulesMiner
    except ImportError:
        print("LỖI: Không tìm thấy file 'apriori_library.py'.")
        print("Hãy đảm bảo bạn đã tạo file này trong thư mục 'src' hoặc cùng thư mục chạy code.")
        sys.exit(1)

# --- CẤU HÌNH THÍ NGHIỆM ---
DATA_PATH = "data/online_retail.csv"  # Đường dẫn file dữ liệu
# Các ngưỡng support cần thử nghiệm (đây là nội dung chính của Chủ đề 4)
SUPPORT_THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]

def main():
    print("="*50)
    print("BẮT ĐẦU CHỦ ĐỀ 4: PHÂN TÍCH ĐỘ NHẠY THAM SỐ")
    print("="*50)

    # 1. CHUẨN BỊ DỮ LIỆU (Chạy 1 lần duy nhất)
    print(f"\n[Bước 1] Đang đọc và làm sạch dữ liệu từ: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"LỖI: Không tìm thấy file '{DATA_PATH}'. Hãy kiểm tra lại thư mục 'data'.")
        return

    cleaner = DataCleaner(DATA_PATH)
    cleaner.load_data()
    df_clean = cleaner.clean_data()
    
    print("[Bước 2] Đang tạo giỏ hàng (Basket)...")
    preparer = BasketPreparer(df_clean)
    preparer.create_basket()
    basket_bool = preparer.encode_basket()
    print(f"-> Kích thước ma trận giỏ hàng: {basket_bool.shape}")

    # 2. CHẠY VÒNG LẶP THÍ NGHIỆM
    print(f"\n[Bước 3] Bắt đầu chạy vòng lặp thử nghiệm với các ngưỡng: {SUPPORT_THRESHOLDS}")
    
    results = []
    miner = AssociationRulesMiner(basket_bool)

    for support in SUPPORT_THRESHOLDS:
        print(f"\n--- Đang chạy với Min Support = {support} ---")
        start_time = time.time()
        
        # a. Tìm tập phổ biến (Frequent Itemsets)
        # Support càng thấp -> Càng nhiều tập phổ biến -> Chạy càng lâu
        frequent_itemsets = miner.mine_frequent_itemsets(min_support=support)
        num_itemsets = len(frequent_itemsets)
        
        # b. Sinh luật (Generate Rules)
        # Giữ cố định Lift > 1 để so sánh công bằng về chất lượng
        try:
            rules = miner.generate_rules(metric="lift", min_threshold=1.0)
            num_rules = len(rules)
            avg_lift = rules["lift"].mean() if num_rules > 0 else 0
            avg_conf = rules["confidence"].mean() if num_rules > 0 else 0
        except ValueError:
            # Trường hợp không tìm ra luật nào
            num_rules = 0
            avg_lift = 0
            avg_conf = 0
            
        duration = time.time() - start_time
        
        print(f"   + Tìm thấy: {num_rules} luật")
        print(f"   + Lift trung bình: {avg_lift:.2f}")
        print(f"   + Thời gian chạy: {duration:.2f} giây")
        
        results.append({
            "min_support": support,
            "num_rules": num_rules,
            "avg_lift": avg_lift,
            "avg_confidence": avg_conf,
            "execution_time": duration
        })

    # 3. TỔNG HỢP VÀ VẼ BIỂU ĐỒ BÁO CÁO
    print("\n" + "="*50)
    print("TỔNG HỢP KẾT QUẢ (Dùng số liệu này cho báo cáo)")
    df_results = pd.DataFrame(results)
    print(df_results)
    print("="*50)

    # Vẽ biểu đồ 2 trục (Dual Axis Chart)
    # Trục trái: Số lượng luật (Cột hoặc Đường)
    # Trục phải: Chất lượng luật - Lift (Đường)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Vẽ đường số lượng luật (Màu xanh)
    color = 'tab:blue'
    ax1.set_xlabel('Min Support Threshold')
    ax1.set_ylabel('Số lượng luật (Number of Rules)', color=color, fontweight='bold')
    ax1.plot(df_results['min_support'], df_results['num_rules'], color=color, marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Tạo trục thứ 2 dùng chung trục X
    ax2 = ax1.twinx()  
    
    # Vẽ đường chất lượng Lift (Màu đỏ)
    color = 'tab:red'
    ax2.set_ylabel('Lift Trung bình (Average Lift)', color=color, fontweight='bold')
    ax2.plot(df_results['min_support'], df_results['avg_lift'], color=color, marker='x', linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('PHÂN TÍCH ĐỘ NHẠY (SENSITIVITY ANALYSIS)\nTrade-off giữa Số lượng luật và Chất lượng (Lift)')
    fig.tight_layout()
    
    # Lưu biểu đồ ra file ảnh để chèn vào báo cáo
    output_img = "sensitivity_analysis_chart.png"
    plt.savefig(output_img)
    print(f"\nĐã lưu biểu đồ vào file: {output_img}")
    
    plt.show()

if __name__ == "__main__":
    main()