# %% [markdown]
# # Báo cáo Đánh giá & Khuyến nghị (Evaluation Report)
#
# **Bối cảnh (Context):**
# Notebook này đóng vai trò như bản báo cáo cuối kỳ hoàn chỉnh trình lên Ban Giám Hiệu/Khách hàng. Nó tổng hợp kết quả từ 4 Notebook trước, thực hiện Phân tích lỗi (Error Analysis) ở mức Chiến lược, và đưa ra 5 Insight hành động (Actionable Insights) dựa trên tư duy Bloom's Level 5 (Đánh giá & Sáng tạo).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="darkgrid", rc={"axes.facecolor":"#121212", "figure.facecolor":"#121212", 
                                    "axes.edgecolor":"#333333", "grid.color":"#333333",
                                    "text.color":"white", "axes.labelcolor":"white", 
                                    "xtick.color":"white", "ytick.color":"white"})
plt.rcParams["font.family"] = "sans-serif"

# %% [markdown]
# ## 1. Tổng quát Toàn cục (Holistic Pipeline Review)
# Chúng ta đã đi qua quy trình chuẩn từ Data Raw -> EDA -> Discretization -> Association Rules -> Clustering -> Classification -> Regression (Time Series Evaluation).
# Sự thú vị nhất nằm ở lúc phát hiện ra **CV Illusion (Ảo giác Đánh giá do Leakage)** trong dữ liệu Nông nghiệp.

# %%
# Load bảng kết quả từ NB04
try:
    df_reg = pd.read_csv("../outputs/tables/regression_splits_comparison.csv")
    display(df_reg)
except:
    print("Vui lòng chạy NB04 trước để sinh bảng kết quả.")

# %% [markdown]
# ## 2. Phân tích Dạng Lỗi (Error Analysis)
# Mô hình dự đoán sai nhiều nhất ở đâu?
# 1. **Under-predicting Spike Years (Đoán trượt mũi nhọn năng suất)**: Random Forest/XGBoost có xu hướng dự đoán mức "trung bình an toàn" đối với các ngoại lai (Outliers). Khi thời tiết thuận lợi bất thường hoặc công nghệ sinh học bùng nổ (Giống lúa mới), mô hình sẽ đoán thấp hơn thực tế (Under-predict). Lý do: MAE penalty không đủ cao cho những lỗi quá xa (không giống MSE), và các cây quyết định trung bình hóa output lá cuối.
# 2. **Hiệu ứng Nhầm Nhãn Phân loại (False Negative Risk)**: Ở bài toán Phân lớp NB03, mô hình đôi khi dự đoán Medium khi năng suất thực chất là Low (Crop Failure). Điều này đặc biệt độc hại nếu dùng Deploy Real-time vì Bộ Nông nghiệp sẽ không phát được cảnh báo thiên tai.

# %% [markdown]
# ## 3. Đánh giá Hệ thống & Đề xuất Nâng cấp Kỹ thuật (Technical Roadmaps)
# Nếu phải đưa hệ thống này vào Production cho Hợp tác xã/Chính phủ:
# - **Feature Engineering Mới**: Cần thêm `Yield_Lag_1` (Năng suất năm ngoái), `Yield_Lag_2` và `Pesticide_Cumulative` (Độ tích tụ hóa chất trong đất) thay vì chỉ nhìn vào một lát cắt năm hiện tại.
# - **Data Enrichment**: Yếu tố cốt lõi của năng suất không chỉ là Thời tiết mà còn là *Loại đất (Soil Type)* và *Giống cây (Seed Variety)*. Việc thiếu 2 thuộc tính này khiến trần hiệu năng (Performance Ceiling) của ML bay lượn ở mức 70%.
# - **Custom Loss Function**: Viết hàm Loss riêng trong XGBoost (Tương tự Focal Loss) để trừng phạt thật nặng những case đoán "Yield_High" nhưng sự thật là "Yield_Low".

# %% [markdown]
# ## 4. 🚀 TOP 5 Actionable Insights (Dành cho Quản trị & Nông dân)
# Xuyên suốt quá trình Data Mining, đây là 5 tài sản trí tuệ đắt giá nhất rút ra được (Bloom's Evaluate Level):
# 
# 1. **Dừng Việc Tối Ưu Hóa Tối Đa Thuốc Trừ Sâu (Pesticides Plateau):**
#    Luật Kết Hợp (Association Rules) ở NB02 chỉ ra rằng việc gia tăng thuốc trừ sâu (Pesticides High) trong điều kiện nhiệt độ không ủng hộ (Temp_Hot) KHÔNG dẫn đến luật năng suất cao bền vững, ngược lại tạo Lift cực đại cho các luật năng suất thấp/trung bình. Nông dân cần được khuyến nghị về "Điểm bão hoà hóa chất".
# 
# 2. **Chấp Nhận False Alarms Để Cứu Vãn Nền Nông Nghiệp:**
#    Dựa trên Trade-off NB03, hãy hạ Threshold của class "Yield_Low" (Thất bát). Thà chịu chi phí báo động giả 20% (False Positives) để kích hoạt quỹ bảo vệ còn hơn bỏ sót 10% vụ mùa mất trắng (False Negatives - Recall Drop). Nông nghiệp không phải sân chơi cần Precision tuyệt đối.
# 
# 3. **Phân Bổ Trợ Cấp Dựa Trên Cụm Sinh Thái (Clustering-Based Subsidies):**
#    Thay vì phân bổ chính sách cào bằng qua từng Tỉnh/Quốc gia (Area), K-Means Profiling (NB02) cho thấy Chính phủ nên chia ngân sách theo "Cụm thời tiết hiện tại". Vùng đang rơi vào Cụm Nhiệt độ Nóng Cực Đoan mặc định phải được trợ giá Nước tưới khẩn cấp.
# 
# 4. **Ngừng Sử Dụng Random Split Trong Tuyển Sinh/Tuyển Dụng Dự Án AgTech:**
#    Màn trình diễn tệ hại của mô hình khi test trên Time-Series Split (NB04) là tiếng chuông cảnh tỉnh đỏ. Bất kỳ Data Scientist nào dùng Cross-Validation thông thường trên tập dữ liệu này đang "bán" một chỉ số MAE giả mạo do học lỏm xu hướng tương lai. Data Drift là kẻ thù số 1.
# 
# 5. **Chuyển Dịch Lợi Thế Thời Tiết:**
#    Từ EDA NB01 (Top 10 Countries Boxplot), các quốc gia có năng suất cao kỉ lục nằm trong vùng dao động nhiệt độ rất cô đặc. Nông dân cần đẩy mạnh nhà kính (Greenhouse automation) để giả lập dải nhiệt chuẩn xác, vì chỉ vài độ C thay đổi thiên nhiên cũng bẻ cong hoàn toàn đường cung năng suất.

# %%
print("KẾT THÚC PIPELINE DỰ ÁN. CẢM ƠN BẠN ĐÃ THEO DÕI!")
