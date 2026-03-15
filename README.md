# Crop Yield Prediction - Data Mining Project

Dự án Data Mining cuối kỳ môn học: Dự báo năng suất cây trồng (Crop Yield Prediction).
Toàn bộ logic xử lý, EDA, Data Mining, Modeling và Report được trình bày trực tiếp trong 5 notebooks theo luồng chuẩn. Khuyến nghị chạy dự án bằng script tự động.

## Cấu trúc thư mục

```text
DATA_MINING_PROJECT/
├── README.md
├── requirements.txt
├── configs/
│   └── params.yaml          # Chứa tham số cấu hình: seed, split ratio, đường dẫn data
├── data/
│   ├── raw/                 # Nơi chứa `yield_df.csv`
│   └── processed/           # Dữ liệu trung gian sau tiền xử lý
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_mining_clustering.ipynb
│   ├── 03_modeling_classification.ipynb
│   ├── 04_modeling_regression_timeseries.ipynb
│   └── 05_evaluation_report.ipynb
├── scripts/
│   └── run_papermill.py     # Script chạy tự động theo tuần tự
└── outputs/
    ├── figures/
    ├── tables/
    └── models/
```

## Hướng dẫn tái lập (Reproducible Guide)

1. Cài đặt môi trường và các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Đảm bảo bạn đã đặt file `yield_df.csv` vào thư mục `data/raw/`.
3. Cập nhật các tuỳ chỉnh trong `configs/params.yaml` nếu cần.
4. Chạy toàn bộ Jupyter Notebooks theo thứ tự bằng Papermill:
   ```bash
   python scripts/run_papermill.py
   ```
5. Đợi script hoàn tất. Các biểu đồ, bảng và mô hình sẽ được lưu tự động vào thư mục `outputs/`.
