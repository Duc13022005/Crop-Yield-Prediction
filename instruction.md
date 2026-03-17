# Hướng dẫn & Giải thích Cấu trúc Dự án

Dưới đây là lời giải thích chi tiết cho các câu hỏi của bạn về cấu trúc của thư mục `notebooks/` và luồng chạy dữ liệu của dự án.

## 1. Tại sao trong folder `notebooks/` lại chứa 3 file cho mỗi loại? Khác gì nhau?
Trong mô hình Vibe Coding (Sử dụng Papermill và Jupytext), mã nguồn được sinh ra thành 3 định dạng cho các mục đích khác nhau. Ví dụ với `01_eda_preprocessing`:
- `01_eda_preprocessing.py`: Đây là file source **Python script** (định dạng Jupytext). Ưu điểm của file này là gọn nhẹ, dễ dàng đưa lên Git và tiện cho việc Code Review (vì file `.ipynb` gốc chứa metadata bằng JSON rất khó đọc diff).
- `01_eda_preprocessing.ipynb`: Đây là **Notebook gốc** được tạo tự động từ file `.py`. File này chứa toàn bộ code và Markdown nhưng **chưa được chạy** (không có output ô cell).
- `01_eda_preprocessing_executed.ipynb`: Đây là **Notebook đã có Output**, sinh ra sau khi chạy automation bằng hệ thống `run_papermill.py`. File này lưu trữ lại toàn bộ trạng thái hệ thống, biểu đồ sinh ra ở thời điểm thực thi để làm bằng chứng (Artifact log).

## 2. Thứ tự chạy và Có thể chạy riêng từng file không?
**Thứ tự chạy bắt buộc:**
Bạn phải chạy theo thứ tự từ `01` đến `05` ở lần chạy đầu tiên.

**Chạy riêng lẻ:**
**CÓ THỂ**. Sau khi bạn chạy **file `01_eda_preprocessing` ÍT NHẤT MỘT LẦN**, dữ liệu đã được làm sạch và lưu vào thư mục `data/processed/`. Từ lúc này trở đi, bạn có thể tự do mở bất kỳ file `02`, `03`, hoặc `04` lên và chạy độc lập tùy thích để debug hoặc thử nghiệm vì chúng sẽ đọc data từ `data/processed/` mà không cần chạy lại file `01`.

## 3. Input & Output của từng Notebook

- **Notebook 01: Khám phá & Tiền xử lý**
  - **Input**: Đọc tập dữ liệu thô `data/raw/yield_df.csv`
  - **Output**: Lưu dữ liệu đã xử lý vào `data/processed/scaled_data.csv` (Dùng cho Clustering/Classification) và `data/processed/discretized_data.csv` (Dùng cho Association Rules). Xuất các biểu đồ cấu trúc bộ dữ liệu ra `outputs/figures/eda_*.png`.

- **Notebook 02: Khai phá & Phân cụm (Mining & Clustering)**
  - **Input**: Bộ dữ liệu trung gian `data/processed/scaled_data.csv` và `data/processed/discretized_data.csv`.
  - **Output**: Biểu đồ phân nhóm thực địa tại `outputs/figures/clustering_profile.png`.

- **Notebook 03: Phân lớp (Classification Modeling)**
  - **Input**: Dữ liệu đã chuẩn hoá `data/processed/scaled_data.csv`.
  - **Output**: Matrix nhầm lẫn dự báo hỏng mùa `outputs/figures/classification_rf_cm.png`.

- **Notebook 04: Hồi quy & Phân tích mốc thời gian (Regression & Time-Series)**
  - **Input**: Đọc mốc thời gian từ `data/raw/yield_df.csv` kết hợp với thuật toán cắt Split-time.
  - **Output**: Lưu số liệu chênh lệch giữa Random Split và Time-Series vào `outputs/tables/regression_splits_comparison.csv` và biểu đồ cực hạn `outputs/figures/regression_cv_illusion.png`.

- **Notebook 05: Báo cáo Cuối Kỳ (Evaluation Report)**
  - **Input**: Sử dụng trực tiếp bảng số liệu `regression_splits_comparison.csv` từ Notebook 04 để phân tích chiến lược.
  - **Output**: Bản tóm tắt tư duy cho người đọc cuối cùng trên Jupyter.

(Lỗi sai vị trí của thư mục `processed` đã được khắc phục ở bản cập nhật code mới nhất).
