# Chương 2: Thiết kế Giải pháp và Quy trình Khai phá

## 2.1. Kiến trúc luồng của Pipeline
Mô hình vận hành hệ thống tuân theo chuẩn **Khai phá tri thức (KDD)** và tự động hoá hoàn toàn (End-to-End). Các khâu bao gồm:
1. Nguồn Dữ liệu Thô (Raw Input).
2. Xử lý Trạm 1: Sàng lọc rác, lấp khuyết, Scaling (EDA & Preprocessing).
3. Trạm 2: Rút trích Luật và Phân cụm vô hướng (Mining & Clustering).
4. Trạm 3: Dò đoán rủi ro phân khúc (Classification).
5. Trạm 4: Dự đoán chuỗi mốc tương lai (Regression & Time-Series).
6. Báo cáo (Evaluation Output).

## 2.2. Chiến lược Tiền xử lý (Preprocessing & Feature Engineering)
Sự thành bại của mô hình nằm ở nền móng làm sạch linh hoạt:
- **Kiểm soát rác & Outlier:** Nhổ bỏ hoàn toàn các hàng trùng lặp. Thanh lý cột mục tiêu `Yield=0` (Năng suất bằng 0 thường do lỗi thống kê thay vì thực tế 0 đồng đều). Hàm `clean_numeric` đánh bật các dạng string xen kẽ trong `Rainfall` bằng cách quăng `np.nan` và `dropna()`.
- **Rẽ nhánh Transformation:**
  - Đối phó với Regression (NB04): Ép dạng **Categorical One-hot Dummies (`float32`)** để chống rò rỉ trọng số nội sinh.
  - Đối phó với Clustering (NB02): Phủ **Z-Score StandardScaler** diện rộng, biến tất cả về phương sai 1 để tính khoảng cách tịnh tiến Euclidean không bị sai lệch. 
  - Đối phó với Apriori (NB02) / Classification (NB03): Trích cấu trúc Quantile (`qcut=3`) vỡ nát biến số thành các rổ "Low, Medium, High", tránh mất cân bằng lớp imbalanced trầm trọng.
- **Tính năng phái sinh (Feature Extraction):** Kéo rễ từ nguyên lý sinh thái Nông Lâm:
  - `Climate_Stress` (Áp lực Khí hậu) = `Avg_Temperature` - `Rainfall`: Vùng quá nóng mà thiếu mưa sẽ nổ tung cây trồng.
  - `Tropical_Index` (Chỉ số Nhiệt đới) = `Avg_Temperature` x `Rainfall`: Điều kiện nóng ẩm chéo (Sinh vi nấm, gây ép dùng Pesticide).

## 2.3. Cốt lõi Khai phá (Data Mining Core)
Hệ thống thể hiện chất xám **Phân tích (Analyze) & Đánh giá (Evaluate)** ở Notebook 02:

1. **Phân cụm Nông sinh thái (Clustering via K-Means)**:
   - Thay vì dùng thuật toán HDBSCAN nặng nề, K-Means tỏa sáng với số chiều bị bóp gọn sau Z-Score. Silhouette Index (>0.35) xác nhận điểm khuỷu tay (Elbow) tối ưu.
   - **Đánh giá Profiling Insight:** Khai phá bật ra Cụm Sinh Thái 1 (Ôn đới, đầu tư hoá chất ít) vs Cụm Sinh Thái 2 (Chảo lửa nhiệt đới, sâu bệnh cao) vs Cụm Sinh Thái 3 (Phát triển cao điểm thuốc sâu).
   - *Minh chứng cụm tại outputs/clustering_profile.png*

2. **Khai phá Luật Liên kết (Association Rule Mining)**:
   - Dùng FP-Growth thay vì Apriori chậm chạp.
   - Tìm kiếm các đường gãy đứt gãy đằng sau mảng Discretized. 
   - **Luật chiết xuất:** Lift > 1.5 với "Support tốt" đã chứng minh những cụm "Giai đoạn Nóng cực độ sẽ luôn kết hợp cùng việc bơm Siêu thuốc trừ sâu".

## 2.4. Lý do Thiết kế Thuật toán Dự báo
- **XGBoost Regressor / Classifier (Mạch Chủ Lực)**: So với Random Forest bảo thủ, cây Gradient Boosting bắt tốt quy luật Gradient đa chiều từ mớ Đặc trưng Phái sinh phức tạp trên (VD: Tropical Index). Hiệu năng bao quát ranh giới cong.
- **Deep Sequence (Mạch Time-Series thay vì Random)**: Kéo dãn bảng thành 3D `[samples, timesteps, features]`. Việc dùng `LSTM 64-32 units` với lookback=3 năm nhằm ghi tạc vào não Nơ-ron (Memory Cell) di chứng thời tiết của năm 1990 sẽ ảnh hưởng phân bón lên rễ năm 1993. Lý do là Ridge Regression hay XGBoost đều dốt về chiều thời gian tịnh tiến.
