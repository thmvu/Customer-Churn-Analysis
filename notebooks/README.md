# Notebook roadmap theo 7 bước giảng viên

Chạy notebook theo đúng thứ tự này để khớp với flow bài tập lớn:

1. `01_data_collection.ipynb`
   - Đọc dataset gốc
   - Xem shape, columns, sample rows
   - Giải thích nguồn và biến mục tiêu `Churn`

2. `02_data_cleaning_preprocessing.ipynb`
   - Xử lý `TotalCharges`
   - Bỏ `customerID`
   - Encode, split train/test, SMOTENC, StandardScaler

3. `03_eda_analysis.ipynb`
   - Vẽ bar chart, pie chart, histogram, boxplot, heatmap
   - Rút insight từ `Contract`, `tenure`, `MonthlyCharges`, `TechSupport`, `OnlineSecurity`

4. `04_model_training.ipynb`
   - Train các model với full features
   - Logistic Regression, Decision Tree, Random Forest, boosting models, Voting Ensemble

5. `05_feature_selection_optimization.ipynb`
   - Random Forest feature importance
   - Mutual Information
   - SelectKBest
   - So sánh full features vs selected features

6. `06_model_evaluation_comparison.ipynb`
   - So sánh accuracy, precision, recall, f1, roc_auc
   - Confusion matrix, ROC curve, feature importance

7. `07_result_communication.ipynb`
   - Tổng hợp insight
   - Gợi ý nội dung slide/report
   - Kết luận và hướng phát triển

8. `08_customer_prediction_demo.ipynb` (bổ sung cho slide demo)
   - Load `best_model.pkl`
   - Nhập hai khách hàng giả định
   - Dự đoán nhãn `Churn` và xác suất rời bỏ dịch vụ
   - Gợi ý hành động chăm sóc khách hàng theo mức rủi ro

Các notebook cũ đã được chuyển vào `notebooks/legacy/` để giữ lại lịch sử làm bài.
