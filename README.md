# VIBE CODING PROJECT DATA MINING: CROP YIELD PREDICTION 🌾

> **Dự án cuối kỳ Môn Khai phá Dữ liệu (Data Mining)**  
> **Sinh viên thực hiện:** Lê Thị Thùy Trang

---

## 📌 1. Bài toán (Problem Statement)
**Bối cảnh:** Biến đổi khí hậu đang làm suy giảm nghiêm trọng sản lượng nông nghiệp toàn cầu. Việc dự báo chính xác năng suất cây trồng giữa hằng hà sa số các yếu tố thời tiết (nhiệt độ, lượng mưa) và sự tác động của con người (thuốc trừ sâu) là một thách thức lớn.

**Mục tiêu:** Xây dựng một đường ống (pipeline) Khai phá Dữ liệu toàn diện từ đầu tới cuối (End-to-End) dựa trên kho dữ liệu của Tổ chức Lương thực và Nông nghiệp Liên Hợp Quốc (FAO) từ năm 1990 đến 2013. Sản phẩm cốt lõi là các mô hình dự báo nhằm cung cấp hệ thống cảnh báo sớm rủi ro mất mùa và hỗ trợ điều phối chính sách nông nghiệp vĩ mô.

**Ba trụ cột khai phá (Mining Pillars):**
1. **Unsupervised Learning (Mining & Clustering):** Dùng FP-Growth và K-Means để bóc tách quy luật thời tiết và phân cụm không gian sinh thái nông nghiệp.
2. **Supervised Classification:** Mô hình học máy rời rạc (Random Forest, XGBoost) chẩn đoán tình trạng mùa vụ (Thấp/Trung Bình/Cao) ưu tiên bắt chính xác rủi ro thất thu.
3. **Time-Series Regression & Deep Learning:** Khởi chạy XGBoost và Mạng trí tuệ nhân tạo (LSTM 3D Tensor) để rà soát sự lệch pha của độ trôi dạt dữ liệu (Data Drift) theo dòng thời gian thực tế.

---

## 🏗️ 2. Kiến trúc Hệ thống (Pipeline Architecture)
Dự án được cấu trúc theo chuẩn Data Science Workflow tự động hóa bằng `papermill`. 

**Dòng chảy dữ liệu (Data Flow):**
`Nguồn Data` $\Rightarrow$ `EDA & Preprocessing` $\Rightarrow$ `Khai phá sinh thái` $\Rightarrow$ `Mô hình Phân lớp` $\Rightarrow$ `Mô hình Time-Series` $\Rightarrow$ `Report/Báo cáo`

**Các Module (Notebooks) Cốt lõi:**
- `01_eda_preprocessing.ipynb`: Module làm sạch (Data Cleaner) & Kỹ nghệ Đặc trưng (Feature Builder). Sinh ra tương tác `Climate_Stress`, mã hóa Z-Score và băm Categorical.
- `02_mining_clustering.ipynb`: Module Khai phá (Miner). Tìm luật liên kết FP-Growth và Gom cụm K-Means.
- `03_modeling_classification.ipynb`: Module Phân loại. Huấn luyện Logistic Regression, Random Forest và XGBoost.
- `04_modeling_regression_timeseries.ipynb`: Module Định hướng thời gian (Trainer & Evaluator). Xử lý Sequence Padding cho LSTM và bóc trần ảo giác Leakage thông qua TimeSeriesSplit.
- `05_evaluation_report.ipynb`: Module Tổng hợp hệ thống hóa quy luật phân tích. (Đã được xuất ra thư mục `/report/` thành các định dạng .md).

---

## 📂 3. Cấu trúc Thư mục (Directory Structure)
Sổ lệnh dự án được chia tách chuẩn mực để tái lập (Reproducibility):

```bash
Crop-Yield-Prediction/
├── configs/
│   └── params.yaml                 # Chứa seed cố định và đường dẫn siêu tham số (Hyperparams)
├── data/
│   ├── raw/                        # Chứa file yield_df.csv gốc
│   └── processed/                  # Dữ liệu đã Scale, Discretized sẵn sàng Train
├── notebooks/                      # Các kịch bản lõi phân tích (.ipynb)
├── outputs/
│   ├── figures/                    # Hình ảnh biểu đồ, Plot v.v
│   └── tables/                     # Các file CSV kết quả dự báo/so sánh
├── report/                         # Báo cáo bản thu thập (.md) theo cấu trúc chương
├── scripts/
│   └── run_papermill.py            # Công cụ kích hoạt tự động Pipeline tuần tự
├── .gitignore
├── requirements.txt                # Danh sách thư viện môi trường
└── README.md                       # Tài liệu hướng dẫn (File này)
```

---

## 🚀 4. Hướng dẫn Sử dụng (Usage Instructions)

### Bước 1: Thiết lập môi trường ảo
Nên sử dụng `conda` hoặc `venv` đễ cô lập môi trường của dự án.
```bash
# Tạo môi trường với Python 3.12 (Ví dụ venv)
python -m venv venv_crop
# Kích hoạt môi trường (Windows)
.\venv_crop\Scripts\activate
```

### Bước 2: Cài đặt thư viện phụ thuộc
Đảm bảo bạn đang ở thư mục gốc của dự án `Crop-Yield-Prediction`.
```bash
pip install -r requirements.txt
```

### Bước 3: Đảm bảo Dữ liệu thô (Raw Data)
Kiểm tra xem tập tin `yield_df.csv` đã có tại thư mục con `/data/raw/` chưa. Nếu chưa hãy tải nó từ Kaggle bỏ vào.

### Bước 4: Kích hoạt Hệ thống (Run the Pipeline)
Bạn không cần mở từng Notebook lên chạy tay. Toàn bộ dây chuyền khai phá đã được tự động hóa.
Di chuyển vào nhánh thư mục hệ thống:
```bash
cd scripts
```
Chạy công lệnh kích hoạt Papermill (Sẽ mất khoảng thời gian từ 3 - 5 phút tùy năng lực CPU của máy tính):
```bash
python run_papermill.py
```

### Bước 5: Xem lại thành quả
Sau khi `run_papermill.py` báo cáo chạy vòng tròn xanh thành công:
1. Mở thư mục `notebooks/` để xem tất cả các file có hậu tố `_executed.ipynb` - Đây là các file vừa chạy tự động, in sẵn toàn kết quả.
2. Tại thư mục `outputs/figures/`, mở ngay các hình ảnh vẽ đồ thị Confusion Matrix, Feature Importance, hay Ảo giác Đánh giá Time Series...
3. Thư mục `report/` dùng làm tư liệu chính thức viết báo cáo đồ án của môn học.

---

> **Bản quyền Code Framework:** Xây dựng bởi Lê Thị Thùy Trang - DNU 2026.
