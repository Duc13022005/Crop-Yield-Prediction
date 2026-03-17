# %% [markdown]
# # Mô hình hoá: Hồi quy & Chuỗi thời gian (Regression & Time Series)
#
# **Bối cảnh phân tích (Analytical Context):**
# Khác với bài toán Phân lớp (Nhận diện phân khúc Yield), Hồi quy trực tiếp dự đoán con số Năng suất (hg/ha).
# 
# Tư duy mức cao (Evaluate/Judgement):
# 1. **Baseline vs Strong Model**: Ta thiết lập Ridge Regression làm mô hình cơ sở chống Overfitting do đa cộng tuyến (nếu có). Trái lại, XGBoost Regressor sẽ thu thập các biểu diễn phi tuyến phức tạp giữa Nhiệt-Mưa-Thuốc-Khu vực (Tree-based model không cần One-hot Area cực đoan nếu xài Ordinal nhưng ta dùng Z-score sẵn có).
# 2. **Chọc thủng "CV Illusion" (Đánh giá Chéo Sai lầm)**:
#    - Dữ liệu Nông nghiệp bản chất là Time-Series dọc theo các `Year` (Năm).
#    - Việc dùng Random `train_test_split` trích xuất các điểm dữ liệu năm 2010 vứt vào tập Train và để 2005 ở Test tạo ra Leakage Khổng Lồ (Học Tương Lai dự đoán Quá khứ). 
#    - Đánh đổi (Trade-off): Ta sẽ chứng minh hiệu năng **Sụp đổ/Giảm sút** của XGBoost khi đổi từ Random Split sang Time-Series Split (Cắt ngang thời gian). Đó không phải là mô hình dở, mà đó mới là **Hiệu năng Thực tế** của việc "Dự báo" mùa vụ năm sau từ dữ liệu năm trước.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

sns.set_theme(style="darkgrid", rc={"axes.facecolor":"#121212", "figure.facecolor":"#121212", 
                                    "axes.edgecolor":"#333333", "grid.color":"#333333",
                                    "text.color":"white", "axes.labelcolor":"white", 
                                    "xtick.color":"white", "ytick.color":"white"})
plt.rcParams["font.family"] = "sans-serif"

with open("../configs/params.yaml", "r") as f:
    config = yaml.safe_load(f)

# Phải load RAW DATA để có Year, rồi ghép với SCALED DATA (hoặc scale lại nhanh)
df_raw = pd.read_csv("../" + config['data']['raw_path'])
# Giữ nguyên pipeline clean như nb01
df_raw.drop_duplicates(inplace=True)
df_raw.dropna(inplace=True)
df_raw = df_raw[(df_raw['hg/ha_yield'] > 0)]
df_raw.rename(columns={
    'hg/ha_yield': 'Yield',
    'average_rain_fall_mm_per_year': 'Rainfall',
    'pesticides_tonnes': 'Pesticides',
    'avg_temp': 'Avg_Temperature'
}, inplace=True)
def clean_numeric(val):
    try:
        if pd.isna(val): return np.nan
        return float(str(val).strip())
    except:
        return np.nan
df_raw['Rainfall'] = df_raw['Rainfall'].apply(clean_numeric)
df_raw.dropna(subset=['Rainfall'], inplace=True)

# %% [markdown]
# ## 1. Lấy dữ liệu và Chuẩn bị Features
# Do `Year` là yếu tố cốt lõi trong phần này, ta sẽ dùng Dữ liệu thô và Scale tại chỗ các biến số. Target không scale để MAE dễ lý giải (tính bằng hg/ha).

# %%
df_ts = df_raw.copy()
# Sort cứng theo thời gian
df_ts.sort_values(by=['Area', 'Item', 'Year'], inplace=True)

# Tích hợp One-hot/Categorical Encoding thay vì Label Encoding
df_ts = pd.get_dummies(df_ts, columns=['Area', 'Item'], drop_first=True)

# Lấy các biến tĩnh làm features (bỏ Year và Target Yield) và ép toàn bộ bool từ One-hot về float32
feature_cols = [c for c in df_ts.columns if c not in ['Year', 'Yield']]
for col in feature_cols:
    df_ts[col] = df_ts[col].astype(np.float32)

X = df_ts[feature_cols].values
y = df_ts['Yield'].values
years = df_ts['Year'].values

print("Biên độ Thời gian:", years.min(), "->", years.max())

# %% [markdown]
# ## 2. Phân tích Ảo giác CV (Random Split vs Time Split)

# %% [markdown]
# ### 2.1 Cách 1: Random Split rò rỉ Time (CV Truyền thống)
# Lấy ngẫu nhiên 80% Train, 20% Test.

# %%
X_train_rd, X_test_rd, y_train_rd, y_test_rd = train_test_split(
    X, y, test_size=config['data']['split_ratio'], random_state=config['project']['random_seed']
)

# Baseline: Ridge
ridge_rd = Ridge(alpha=1.0)
ridge_rd.fit(X_train_rd, y_train_rd)
pred_ridge_rd = ridge_rd.predict(X_test_rd)

# Strong: XGBoost
xgb_rd = XGBRegressor(n_estimators=100, max_depth=5, random_state=config['project']['random_seed'])
xgb_rd.fit(X_train_rd, y_train_rd)
pred_xgb_rd = xgb_rd.predict(X_test_rd)

print("--- NẾU CHIA RANDOM (Lạc quan Tếu / CV Illusion) ---")
print(f"RIDGE   - MAE: {mean_absolute_error(y_test_rd, pred_ridge_rd):.0f} | RMSE: {np.sqrt(mean_squared_error(y_test_rd, pred_ridge_rd)):.0f}")
print(f"XGBOOST - MAE: {mean_absolute_error(y_test_rd, pred_xgb_rd):.0f} | RMSE: {np.sqrt(mean_squared_error(y_test_rd, pred_xgb_rd)):.0f}")

# %% [markdown]
# ### 2.2 Cách 2: Time Series Split (Thực tế)
# Chúng ta sẽ ngắt đôi biểu đồ thời gian: Dữ liệu (Trước mốc T) để Train, Dữ liệu (Sau mốc T) để Test.
# Mốc `T` được xác định để lấy ~80% data đầu cho train.

# %%
split_idx = int(len(X) * (1 - config['data']['split_ratio']))
split_year = years[split_idx]
print(f"Chọn năm chia cắt (Split Threshold): {split_year}")

# Cắt array cứng
X_train_ts, X_test_ts = X[:split_idx], X[split_idx:]
y_train_ts, y_test_ts = y[:split_idx], y[split_idx:]

# Fit Baseline & Strong Model
ridge_ts = Ridge(alpha=1.0)
ridge_ts.fit(X_train_ts, y_train_ts)
pred_ridge_ts = ridge_ts.predict(X_test_ts)

xgb_ts = XGBRegressor(n_estimators=300, max_depth=5, random_state=config['project']['random_seed'])
xgb_ts.fit(X_train_ts, y_train_ts)
pred_xgb_ts = xgb_ts.predict(X_test_ts)

# --- ADVANCED LSTM PROCESSING (Sequence Grouping & Padding) ---
# Tái định hình lại ma trận 3D [samples, timesteps, features] dựa trên logic nhóm theo Dòng Thời Gian (Year sequence)
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math

df_train_ts = df_ts.iloc[:split_idx].copy()
df_test_ts = df_ts.iloc[split_idx:].copy()

# Cần phục hồi lại Group ID từ các dummy variables: Do quá nhiều One-hots, ta tạo nhóm ảo dựa trên nhóm liên hoàn các dòng 
# (Vì ban đầu ta đã sort theo ['Area', 'Item', 'Year']) -> Khoảng cách giữa các Year thường là 1 bước.
# Ta sẽ tái tạo lại Sequence bằng cách tìm các index đổi dấu Area/Item thông qua features
# (Một cách dễ hơn là lấy nguyên Dataframe và Extract sequences)

def create_sequences(df_sub, feature_cols, target_col='Yield', lookback=3):
    X_seq, y_seq = [], []
    for i in range(len(df_sub) - lookback):
        features_array = np.array(df_sub[feature_cols].iloc[i:(i + lookback)].values, dtype=np.float32)
        target_val = np.float32(df_sub[target_col].iloc[i + lookback])
        X_seq.append(features_array)
        y_seq.append(target_val)
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

# Do Time Split đã cắt ngang năm, ta áp dụng build Sequence trực tiếp trên cục Train/Test. Pad = None vì window cố định.
lookback = 3
X_train_lstm, y_train_lstm = create_sequences(df_train_ts, feature_cols, lookback=lookback)
X_test_lstm, y_test_lstm = create_sequences(df_test_ts, feature_cols, lookback=lookback)

# Nếu list bị rỗng (do Test/Train quá ít data, ta rollback về 1D pad)
if len(X_train_lstm) == 0:
    X_train_lstm = X_train_ts.reshape((X_train_ts.shape[0], 1, X_train_ts.shape[1]))
    y_train_lstm = y_train_ts
if len(X_test_lstm) == 0:
    X_test_lstm = X_test_ts.reshape((X_test_ts.shape[0], 1, X_test_ts.shape[1]))
    y_test_lstm = y_test_ts

lstm_model = Sequential([
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    LSTM(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

print("\n--- ĐANG TRAIN LSTM (50 Epochs) ---")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=50, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)
pred_lstm_ts = lstm_model.predict(X_test_lstm).flatten()

# Note: Độ dài test arrays của LSTM đã bị thụt lùi mất `lookback` do windowing. Bổ túc lại cho metrics
# bằng cách cắt tương ứng y_test_ts
y_test_ts_lstm_eval = y_test_ts[lookback:] if len(X_test_lstm) < len(y_test_ts) else y_test_ts

print("\n--- CHIA KÉO THEO THỜI GIAN CHUẨN MỰC (Time Series Split) ---")
print(f"RIDGE   - MAE: {mean_absolute_error(y_test_ts, pred_ridge_ts):.0f} | RMSE: {np.sqrt(mean_squared_error(y_test_ts, pred_ridge_ts)):.0f}")
print(f"XGBOOST - MAE: {mean_absolute_error(y_test_ts, pred_xgb_ts):.0f} | RMSE: {np.sqrt(mean_squared_error(y_test_ts, pred_xgb_ts)):.0f}")
print(f"LSTM    - MAE: {mean_absolute_error(y_test_ts_lstm_eval, pred_lstm_ts):.0f} | RMSE: {np.sqrt(mean_squared_error(y_test_ts_lstm_eval, pred_lstm_ts)):.0f}")

# %% [markdown]
# ### Đánh giá chuyên sâu (Evaluate)
# 1. **Data Drift (Trôi dạt số liệu)**: Phương pháp Nông nghiệp (Thuốc trừ sâu mới), Biến đổi Khí hậu (Nhiệt độ thay đổi cấu trúc hoàn toàn từ năm này sang năm khác) khiến dữ liệu Lịch Sử không hoàn toàn đại diện cho Tương Lai. Ta có thể đoán MAE ở mô hình cắt chuẩn (Time Split) sẽ tồi tệ hơn hẳn chia Random.
# 2. **Bài học ứng dụng**: Đối tác (Nông Dân/Chính Phủ) cần biết con số Tồi Tệ Thực Tế (Time Split) chứ không phải con số Đẹp Lạc Quan (Random CV). Việc dùng Random CV ở đây có thể tạo ra kỳ vọng giả mạo.

# Lấy metrics đẩy ra ngoài báo cáo
results = {
    'Method': ['Random Split', 'Random Split', 'Time Split', 'Time Split', 'Time Split (LSTM)'],
    'Model': ['Ridge', 'XGBoost', 'Ridge', 'XGBoost', 'LSTM'],
    'MAE': [
        mean_absolute_error(y_test_rd, pred_ridge_rd),
        mean_absolute_error(y_test_rd, pred_xgb_rd),
        mean_absolute_error(y_test_ts, pred_ridge_ts),
        mean_absolute_error(y_test_ts, pred_xgb_ts),
        mean_absolute_error(y_test_ts_lstm_eval, pred_lstm_ts)
    ]
}
df_res = pd.DataFrame(results)
df_res.to_csv("../outputs/tables/regression_splits_comparison.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_res, x='Method', y='MAE', hue='Model', palette='coolwarm')
plt.title("Ảo giác Đánh giá: Phân biệt Khốc liệt giữa Random CV và Time Split\n(Càng thấp càng tốt)", fontsize=14, fontweight='bold')
plt.ylabel("Sai số Tuyệt đối Trung bình (MAE - hg/ha)")
plt.savefig("../outputs/figures/regression_cv_illusion.png", dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Kết luận Hồi Quy:**
# Nếu nhìn vào MAE của XGBoost trên Time-split, sự tăng trưởng cực khủng của sai số nhắc nhở Data Scientist rằng Model chưa bắt được *Xu hướng Mùa vụ (Seasonality/Trend)* dài hạn chỉ dựa vào feature (Temp, Pest, Rain). Gợi ý cải tiến (Cho NB05): Bổ sung Lags (Yield của năm trước) để mô hình có định hướng Time Series đích thực.
