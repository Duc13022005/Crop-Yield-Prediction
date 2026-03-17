# %% [markdown]
# # Mô hình hoá: Phân Lớp (Classification)
#
# **Bối cảnh phân tích (Analytical Context):**
# Notebook này biến đổi biến mục tiêu Năng suất (Yield) từ liên tục (Regression) sang rời rạc (Classification) thành 3 nhãn: Thấp (Low), Trung bình (Medium), và Cao (High) để giải quyết yêu cầu Phân lớp.
# 
# Tư duy mức cao (Evaluate/Judgement):
# - **Xác định Baseline**: Ta sử dụng một mô hình cơ bản (Logistic Regression) làm điểm chuẩn (Baseline) vì nó tuyến tính, dễ giải thích ranh giới quyết định. Mô hình học sâu/vượt trội (Strong Model) là Random Forest Classifier.
# - **Trade-off Analysis (Đánh đổi Precision/Recall)**:
#   - Trong bài toán Nông nghiệp, dự đoán "Năng suất cực thấp" (Tương đương Vụ mùa thất bát) là trọng tâm quản trị rủi ro. 
#   - Nếu Precision(Thấp) cao nhưng Recall(Thấp) thấp: Nghĩa là hệ thống rất chắc chắn khi báo động, nhưng lại bỏ sót cực kỳ nhiều năm thất thu thật sự. (Nguy hiểm!).
#   - Nếu Recall(Thấp) cao nhưng Precision(Thấp) thấp: Hệ thống hay báo động nhầm (False Alarms) khiến nông dân tốn tiền đề phòng, nhưng bù lại ít khi "chết bất đắc kỳ tử". Sự đánh đổi này thiên về Recall nếu ưu tiên an toàn lương thực. Ta sẽ Evaluate trên **F1-macro** để có cái nhìn tổng quát giữa các class (do có thể bị imbalanced).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay

sns.set_theme(style="darkgrid", rc={"axes.facecolor":"#121212", "figure.facecolor":"#121212", 
                                    "axes.edgecolor":"#333333", "grid.color":"#333333",
                                    "text.color":"white", "axes.labelcolor":"white", 
                                    "xtick.color":"white", "ytick.color":"white"})
plt.rcParams["font.family"] = "sans-serif"

with open("../configs/params.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load data đã chuẩn hoá cho Mô hình
df_model = pd.read_csv(os.path.join("../" + config['data']['processed_path'], "scaled_data.csv"))

# %% [markdown]
# ## 1. Rời rạc hoá biến Mục tiêu (Target Discretization)
# Biến đổi biến Yield liên tục thành class bằng Quantile (Mỗi class 33% Data). Bằng cách này ta tránh được Class Imbalance cực độ.
# Lưu ý: Các Features input (Nhiệt độ, Mưa, Thuốc, Area) đã được Scaling Z-score từ NB01. Ta tập trung vào 3 features số đầu vào: Rainfall, Pesticides, Avg_Temperature vì Area đang bị mã hóa ở One-hot quá lãng phí. Để có base tốt, ta sẽ chỉ dùng yếu tố khí hậu + hóa chất tĩnh.

# %%
# Tạo thêm Đặc trưng mới (Feature Extraction) dựa trên Insight khai phá
df_model['Climate_Stress'] = df_model['Avg_Temperature'] - df_model['Rainfall']
df_model['Tropical_Index'] = df_model['Avg_Temperature'] * df_model['Rainfall']

from sklearn.preprocessing import LabelEncoder
df_model['Area_Encoded'] = LabelEncoder().fit_transform(df_model['Area'])
df_model['Item_Encoded'] = LabelEncoder().fit_transform(df_model['Item'])

features = ['Rainfall', 'Avg_Temperature', 'Pesticides', 'Climate_Stress', 'Tropical_Index', 'Area_Encoded', 'Item_Encoded']
X = df_model[features]

# Định nghĩa target (Yield -> Low, Medium, High) 
# Chú ý Yield ở đây là df_model['Yield'] đã bị Z-transformed. 
df_model['Yield_Class'] = pd.qcut(df_model['Yield'], q=3, labels=['Low', 'Medium', 'High'])
y = df_model['Yield_Class']

print("Phân phối (Class Distribution):")
print(y.value_counts(normalize=True))

# Chia tập Test/Train (Tỉ lệ lấy từ configs)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=config['data']['split_ratio'], 
    random_state=config['project']['random_seed'],
    stratify=y
)

# %% [markdown]
# ## 2. Huấn luyện Mô hình: Baseline vs. Strong Model

# %%
# 1. BASELINE: Logistic Regression
baseline_model = LogisticRegression(random_state=config['project']['random_seed'])
baseline_model.fit(X_train, y_train)
y_pred_base = baseline_model.predict(X_test)
base_f1_macro = f1_score(y_test, y_pred_base, average='macro')

# 2. STRONG MODEL: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=config['project']['random_seed'], max_depth=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_f1_macro = f1_score(y_test, y_pred_rf, average='macro')

# 3. ADVANCED MODEL: XGBoost (Phase 2 Enhancement)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

xgb_model = XGBClassifier(n_estimators=300, random_state=config['project']['random_seed'], max_depth=6)
xgb_model.fit(X_train, y_train_enc)
y_pred_xgb_enc = xgb_model.predict(X_test)
y_pred_xgb = le.inverse_transform(y_pred_xgb_enc)
xgb_f1_macro = f1_score(y_test, y_pred_xgb, average='macro')

print("=== SO SÁNH F1-MACRO ===")
print(f"Baseline (Logistic Regression): {base_f1_macro:.4f}")
print(f"Strong (Random Forest)        : {rf_f1_macro:.4f}")
print(f"Advanced (XGBoost 300 estimators): {xgb_f1_macro:.4f}")

# %% [markdown]
# **Đánh giá Hiệu năng Cơ bản:** RF Model nhỉnh hơn Baseline nhưng vì ta chỉ cung cấp 3 biến đầu vào thời tiết không kết hợp Không-thời-gian, F1 là cực điểm. Quan trọng hơn, ta cần xem sự đánh đổi bên trong "Low Yield" class.

# %% [markdown]
# ## 3. Error Analysis & Trade-off: Precision vs Recall
# Xem xét thật kĩ Report của RF Model, tập trung vào dòng `Low`.

# %%
print("\n=== CLASSIFICATION REPORT (XGBOOST) ===")
print(classification_report(y_test, y_pred_xgb))

# Vẽ Confusion Matrix đẹp
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_xgb, labels=['Low', 'Medium', 'High'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title("Confusion Matrix (XGBoost Phase 2)", fontsize=14, fontweight='bold')
plt.savefig("../outputs/figures/classification_xgb_cm.png", dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Bàn luận phân tích (Hành vi & Đánh đổi - Bloom's Evaluate)
# 
# Bạn có thể thấy Confusion Matrix cho hay số lượng lớp bị nhầm từ Medium sang Low và vân vân.
# Xét riêng Class "Low" (Thất thu năng suất):
# - **Recall Class Low:** Trong thực tế CÓ 100 vụ mùa hỏng, thuật toán cảnh báo đúng bao nhiêu %? (Tức là True Positive / Bờ rìa ngang thực tế).
# - **Precision Class Low:** Trong 100 lần thuật toán rú còi cảnh báo hỏng mùa, chỉ có thực tế bao nhiêu % thật? (Tránh Boy Who Cried Wolf).
# 
# **Insight đúc kết cho Nông Lâm Nghiệp:** Nếu mô hình đang bị False Positives quá lớn ở Class Low (Dự đoán Low nhưng thật ra Mid), Nông dân sẽ vội lãng phí Phân bón hoặc Nước để cứu chữa. Ngược lại, False Negatives ở Class Low (Dự đoán Cao/Mid nhưng thật ra đứt mùa) sẽ gây khủng hoảng chuỗi cung ứng quốc gia vì không ngưng kịp dự báo mua gom từ nước ngoài. Ta nên hạ ngưỡng tự tin (Threshold) của thuật toán đối với Class "Low" nếu chính phủ muốn thiên vị "Thà báo nhầm hơn bỏ sót".
