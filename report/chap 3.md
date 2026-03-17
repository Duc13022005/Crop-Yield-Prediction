# Chương 3: Phân tích Mã nguồn và Chức năng

## 3.1. Phân bổ Cấu trúc Kho chứa (Repo Architecture)
Dự án được kết cấu đúng chuẩn theo mô hình Data Science / MLOps hạng nhất, đề cao tính Tái sử dụng (Reproducibility) và Minh bạch (Clean Code). Các Module hoạt động chéo qua nhau:

- **`configs/params.yaml`**: Trái tim não bộ. Lưu vết hạt giống `random_seed=42`, `split_ratio=0.2`. Không bao giờ Hard-code tham số lạc loài trong script. 
- **`data/`**: Cấm commit dữ liệu lên Git (`.gitignore`). Bóp chia thành `/raw` (Dữ liệu gốc), và `/processed` (Thành phẩm sau Scaled/Discretized).
- **`outputs/`**: Nơi sản sinh Báo cáo. Chia ngạch `figures/` cho Đồ thị biểu cảm và `tables/` cho dữ liệu csv thô đánh giá MAE/RMSE.
- **`notebooks/`**: Trung tâm R&D phân tích mã. Được nén tự động (Sync) file Python Scripts (`.py`) để kiểm định qua Jupytext. Mọi giải thích (Markdown Bloom 4, 5) đều ghi tại đây.
- **`scripts/run_papermill.py`**: Khối Module cốt lõi thao túng tự động hoá dây chuyền.

## 3.2. Chức năng từng Module Lõi Notebook (Jupytext Scripts)
Kiến trúc luân chuyển (Pipeline Flow) không gọi Class cục bộ, mà được phân tách làm các File kịch bản chạy song song đồng thời từ đầu đến cuối:

1. **`01_eda_preprocessing.py` (DataCleaner & Transformer)**: 
   Sơ chế mảng bụi (Z-score + Quantiles). Vẽ Scatter, Boxplot. Cứu vãn định dạng `str` lẫn trong Numeric. Trả file về `/processed`.
2. **`02_mining_clustering.py` (Miner)**: 
   Nhập dữ liệu Scaled. Chạy K-Means Elbow / Silhouette. FP-Growth nắn luật Support/Confidence/Lift. Lôi ngược Insight ra đời thực (Actionable insights).
3. **`03_modeling_classification.py` (Categorical Classifier)**: 
   Feature Builder nội sinh (Cấu thành `Climate_Stress`). Dựng Baseline (Logistic) và Triển khai ngách Strong Ensemble (Random Forest, XGBoost) vào việc phân tách Nguy cơ đói kém. Vẽ Confusion Matrix xuất File hình.
4. **`04_modeling_regression_timeseries.py` (Sequence Evaluator)**: 
   Chọc khoá ảo giác Time-Series. Gọi lệnh ép `get_dummies` cho toàn bộ Quốc gia và Chi sinh học (Area/Item). Kéo chuỗi mảng 3D cho Tensor LSTM, Padding Datarrames và đối đầu cực hạn với Baseline. 
5. **`05_evaluation_report.py` (Metric Generator)**:
   Cuốn trôi bảng metrics từ Output, đọc nội dung lỗi tàn dư (Residuals/False alarms) để thiết kế kiến nghị cuối cùng.

## 3.3. Tính khả thi tái chạy lại (Reproducibility & Automation)
Dự án đáp ứng 100% tiêu chí "Chạy lại tạo một cục kết quả y nguyên" thông qua:
1. Giới hạn tĩnh (Seed): Mọi class sklearn hay TF (Random State) đều đóng kín bằng `yaml` config của `params_seed`.
2. Trình mô phỏng Pipeline: Không cần mở Jupyter bấm từng nút `Shift+Enter`. Lệnh `python scripts/run_papermill.py` sẽ tự động đóng vai con người, truy xuất thư viện `Jupytext -> Notebook`, ép chạy Execution tuần tự, tự động bỏ qua nếu báo lõi, và save lại Execution.ipynb mới cóng kèm Output.
3. Kịch bản gói gọn trong Virtual Env: `requirements.txt` ghim sẵn các version (Pandas, Numpy, Keras, xgboost) chống Drift Versions.
