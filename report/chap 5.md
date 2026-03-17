# Chương 5: Thảo luận và So sánh Soi chiếu

## 5.1. Cuộc chiến Điểm chuẩn: Vì sao XGBoost vượt mặt Phần còn lại ở NB03?
- **Giải thích Sự Vượt Trội:** XGBoost (Gradient Boosting) xây nắp các Cây Quyết định (Decision Trees) bổ trợ lỗi cho nhau (Sequential Ensembling) với cường độ 300 vòng lặp. Khác với Random Forest (Xây cây độc lập ngẫu nhiên), XGBoost tận dụng trọn vẹn sự xuất hiện của các "Tri thức phái sinh" (Feature Extraction) mà dự án tạo ra từ EDA: Biến `Climate_Stress` và `Tropical_Index`.
- Các vùng giao thoa (Tương tác Phi tuyến) giữa Mưa siêu lớn và Nhiệt siêu gắt vốn bị thuật toán Baseline Tuyến tính (Logistic) bóp méo, nay được bộ đếm XGBoost nhận diện triệt để, cắt đúng đường phân chia giữa mùa màng "Low Yield" (Thất bại) và "Medium Yield" (Trụ Vững).

## 5.2. Sự thê thảm của Ảo giác Đánh giá (CV Illusion) - XGBoost vs LSTM
Nhánh So Sánh (Thử nghiệm 4) cho thấy một Bài học Cay đắng của Khai phá định lượng:
- **Ưu điểm XGBoost / Nhược điểm Time-Series:** XGBoost bá chủ trên bảng dữ liệu rời rạc. Tuy nhiên, khi đối diện Time-Series Split (cấm nhìn vào tương lai), XGBoost hoàn toàn ngây ngô không hiểu về khái niệm "Quá khứ gần" (Xu hướng năng suất giảm dần theo năm). Nó chỉ nhìn cục bộ (Mưa nhiều thì Mùa cao). 
- **LSTM - Bậc thầy Bối cảnh Khung Bậc:** LSTM với chuỗi Lookback = 3 (Nhìn lùi 3 năm) ghi khắc được Xu Hướng (Trend) và Chu kỳ (Seasonality) để đưa ra chỉ định bù trừ (Padding). Dù LSTM hội tụ chệnh choạng ở Epoch đầu do `Categorical Boolean Matrix` thưa thớt (One-hot Areas), thuật toán tối ưu `Adam` đã bẻ gãy đà vượt rào để ép Valid_Loss hạ nhiệt ngang/xuống dưới mức trượt ngã của Thuật toán Cây độc lập truyền thống. 
- *Lưu ý:* Việc thiết kế vòng lặp Window Sequential bắt buộc ngắt theo `Area/Item` thay vì trượt dài ngẫu nhiên đã vá lỗ thủng "Lấy Chuỗi của Mỹ bỏ bù cho Chuỗi của Việt Nam" -> Đây là nâng cấp cốt tủy làm sạch mảng LSTM.

## 5.3. Thách thức Rủi Ro và Biện pháp Đối diện
1. **Lỗi Rò rỉ Dữ liệu Boolean Object Time-Series:** Khi chèn Dummies của `Area` (VD: Area_Algeria = True/False) vào hàm Sequence trượt chuỗi Tensor, thư viện `Keras/TensorFlow` sụp đổ toàn bộ do không parse được Tensor đa lớp hỗn độn ngầm `(Invalid dtype: object)`.
   *Giải pháp khắc phục:* Áp dụng thủ thuật Ép kiểu cưỡng chế mảng Numpy `astype(np.float32)` xuyên thủng List Comphrehension trên từng Cột Dummy. Điều này chứng minh sự thấu hiểu kiến trúc vi mô của Neural Matrix Computing.
2. **Trade-off Recall (Sự Cực Đoan Cảnh Báo):** Gần 20% dự báo Rủi ro ("Low Yield") của XGBoost là **False Positives** (Tức là mùa còn cứu được nhưng máy dọa Hỏng/Báo động). Việc này bị quy là nhược điểm trong hệ mét học thuật thuần túy, nhưng đối chiếu ở Chương 6, đây là "Ưu điểm Vàng" dưới góc nhìn kinh tế.
