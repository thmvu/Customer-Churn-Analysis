# Báo cáo bài tập lớn: Phân tích và dự đoán khách hàng rời bỏ dịch vụ viễn thông

## 1. Thông tin đề tài

### 1.1. Tên đề tài

**Phân tích và dự đoán khả năng khách hàng rời bỏ dịch vụ viễn thông bằng Machine Learning.**

Trong báo cáo này, hành vi khách hàng rời bỏ dịch vụ được gọi là **Customer Churn**. Đây là bài toán dự đoán xem một khách hàng có khả năng tiếp tục sử dụng dịch vụ hay sẽ ngừng sử dụng trong tương lai.

### 1.2. Lý do chọn đề tài

Trong lĩnh vực viễn thông, việc giữ chân khách hàng cũ thường quan trọng và tiết kiệm chi phí hơn so với việc tìm kiếm khách hàng mới. Nếu doanh nghiệp phát hiện sớm nhóm khách hàng có nguy cơ rời bỏ, doanh nghiệp có thể chủ động đưa ra các chính sách giữ chân như ưu đãi, chăm sóc cá nhân hóa, cải thiện dịch vụ hỗ trợ hoặc điều chỉnh gói cước.

Đề tài này được chọn vì có đầy đủ các yếu tố phù hợp với một bài toán phân tích dữ liệu và học máy:

- Dữ liệu thực tế, dễ hiểu và có ý nghĩa kinh doanh rõ ràng.
- Có bước làm sạch dữ liệu, xử lý dữ liệu thiếu và chuyển đổi kiểu dữ liệu.
- Có phân tích khám phá dữ liệu bằng biểu đồ.
- Có bài toán Machine Learning dạng phân loại nhị phân.
- Có xử lý mất cân bằng dữ liệu bằng SMOTENC.
- Có so sánh nhiều mô hình khác nhau.
- Có đánh giá mô hình bằng nhiều chỉ số như Accuracy, Precision, Recall, F1-score và ROC-AUC.
- Có thể rút ra insight thực tế để hỗ trợ quyết định kinh doanh.

### 1.3. Mục tiêu đề tài

Đề tài hướng đến các mục tiêu chính sau:

1. Thu thập và tìm hiểu bộ dữ liệu Telco Customer Churn.
2. Làm sạch và tiền xử lý dữ liệu để đưa về dạng phù hợp cho mô hình học máy.
3. Phân tích dữ liệu để tìm các yếu tố liên quan đến khả năng khách hàng rời bỏ dịch vụ.
4. Huấn luyện nhiều mô hình phân loại để dự đoán churn.
5. Thực hiện lựa chọn đặc trưng để kiểm tra feature nào quan trọng.
6. Đánh giá, so sánh mô hình và chọn mô hình tốt nhất.
7. Trình bày kết quả, kết luận và đề xuất hướng ứng dụng thực tế.

## 2. Dữ liệu sử dụng

### 2.1. Nguồn dữ liệu

Bộ dữ liệu được sử dụng là **Telco Customer Churn Dataset**, thường được dùng trong các bài toán dự đoán khách hàng rời bỏ dịch vụ. Trong project, dữ liệu gốc được lưu tại:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Mỗi dòng dữ liệu biểu diễn một khách hàng. Mỗi cột biểu diễn một đặc điểm của khách hàng như thông tin cá nhân, loại dịch vụ đang dùng, loại hợp đồng, phương thức thanh toán và chi phí.

### 2.2. Biến mục tiêu

Biến mục tiêu của bài toán là:

```text
Churn
```

Ý nghĩa:

- `Churn = Yes`: khách hàng đã rời bỏ dịch vụ.
- `Churn = No`: khách hàng vẫn tiếp tục sử dụng dịch vụ.

Đây là bài toán **binary classification**, tức là phân loại nhị phân với hai lớp đầu ra.

### 2.3. Các nhóm thuộc tính chính

Các cột dữ liệu được chia thành một số nhóm chính:

**Thông tin cá nhân**

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`

**Thông tin thời gian và chi phí**

- `tenure`
- `MonthlyCharges`
- `TotalCharges`

**Thông tin dịch vụ**

- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`

**Thông tin hợp đồng và thanh toán**

- `Contract`
- `PaperlessBilling`
- `PaymentMethod`

**Cột không dùng để train**

- `customerID`

`customerID` chỉ là mã định danh khách hàng, không mang ý nghĩa dự đoán nên được loại bỏ trong quá trình tiền xử lý.

## 3. Cấu trúc project

Project được tổ chức theo hướng rõ ràng, tách riêng dữ liệu, notebook, mã nguồn, mô hình và báo cáo:

```text
Customer-Churn-Analysis/
├── configs/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── main.py
├── README.md
└── requirements.txt
```

Ý nghĩa các thư mục:

- `configs/`: chứa file cấu hình `config.yaml`.
- `data/raw/`: chứa dữ liệu gốc.
- `data/processed/`: chứa dữ liệu sau xử lý.
- `src/`: chứa code chính của pipeline.
- `notebooks/`: chứa notebook trình bày từng bước theo yêu cầu giảng viên.
- `models/trained_models/`: lưu mô hình đã huấn luyện.
- `models/metrics/`: lưu kết quả đánh giá mô hình.
- `reports/figures/`: lưu biểu đồ.
- `reports/final_report/`: lưu báo cáo cuối cùng.

## 4. Quy trình thực hiện theo 7 bước

Project được chia thành 7 notebook tương ứng với 7 bước trong hướng dẫn làm bài tập lớn.

### 4.1. Notebook 01: Thu thập dữ liệu

File:

```text
notebooks/01_data_collection.ipynb
```

Mục tiêu:

- Đọc dữ liệu từ file CSV.
- Kiểm tra kích thước dữ liệu.
- Xem danh sách cột.
- Xem một số dòng dữ liệu mẫu.
- Xác định cột mục tiêu `Churn`.

Các thao tác chính:

```python
df_raw = load_dataset(PROJECT_ROOT / config["paths"]["raw_data"])
df_raw.shape
df_raw.head()
```

Bước này giúp xác nhận dữ liệu đã được tải đúng và hiểu sơ bộ cấu trúc dữ liệu trước khi xử lý.

### 4.2. Notebook 02: Làm sạch và tiền xử lý dữ liệu

File:

```text
notebooks/02_data_cleaning_preprocessing.ipynb
```

Mục tiêu:

- Xử lý lỗi kiểu dữ liệu.
- Xử lý giá trị thiếu.
- Loại bỏ cột không cần thiết.
- Mã hóa dữ liệu phân loại.
- Chia tập train/test.
- Cân bằng dữ liệu bằng SMOTENC.
- Chuẩn hóa dữ liệu số bằng StandardScaler.

#### Xử lý `TotalCharges`

Trong dữ liệu gốc, `TotalCharges` có thể chứa khoảng trắng hoặc giá trị không chuyển được sang số. Vì vậy, cột này được xử lý bằng:

```python
pd.to_numeric(df["TotalCharges"], errors="coerce")
```

Các giá trị lỗi được chuyển thành `NaN`, sau đó điền bằng median. Median được chọn vì ít bị ảnh hưởng bởi ngoại lệ hơn mean.

#### Loại bỏ `customerID`

`customerID` là mã định danh, không giúp mô hình học được quy luật churn. Vì vậy cột này được loại bỏ để tránh gây nhiễu.

#### Encode dữ liệu

Các mô hình Machine Learning không xử lý trực tiếp được chuỗi như `Yes`, `No`, `Month-to-month`. Vì vậy, dữ liệu phân loại được chuyển sang dạng số bằng `LabelEncoder`.

#### Chia train/test

Project sử dụng:

```text
test_size = 0.2
random_state = 42
stratify = y
```

Trong đó, `stratify = y` giúp giữ tỷ lệ churn ở tập train và test gần giống dữ liệu gốc.

#### Xử lý mất cân bằng bằng SMOTENC

Bài toán churn thường bị mất cân bằng vì số khách hàng không churn thường nhiều hơn số khách hàng churn. Nếu không xử lý, mô hình có thể thiên về lớp đông hơn.

Project sử dụng `SMOTENC` thay vì SMOTE thường vì dữ liệu có cả biến số và biến phân loại.

#### Chuẩn hóa bằng StandardScaler

Các biến số như `tenure`, `MonthlyCharges`, `TotalCharges` có thang đo khác nhau. StandardScaler đưa các biến số về thang đo chuẩn hơn, giúp các mô hình như Logistic Regression hoạt động ổn định.

### 4.3. Notebook 03: Khai phá và phân tích dữ liệu

File:

```text
notebooks/03_eda_analysis.ipynb
```

Mục tiêu:

- Trực quan hóa tỷ lệ churn.
- Phân tích churn theo từng nhóm khách hàng.
- Phân tích churn theo hợp đồng, phương thức thanh toán và dịch vụ.
- Tìm các insight có ý nghĩa nghiệp vụ.

Các biểu đồ sử dụng:

- Biểu đồ cột phân bố churn.
- Biểu đồ tròn tỷ lệ churn.
- Biểu đồ churn theo giới tính.
- Biểu đồ churn theo loại hợp đồng.
- Biểu đồ churn theo dịch vụ Internet.
- Histogram cho `tenure`.
- Boxplot cho `MonthlyCharges`.
- Heatmap tương quan.

Một số insight quan trọng:

- Khách hàng dùng hợp đồng `Month-to-month` thường có rủi ro churn cao hơn.
- Khách hàng có `tenure` thấp thường dễ rời bỏ hơn.
- `MonthlyCharges` cao có thể liên quan đến khả năng churn.
- Các dịch vụ hỗ trợ như `TechSupport` và `OnlineSecurity` có liên quan đến khả năng giữ chân khách hàng.

### 4.4. Notebook 04: Huấn luyện mô hình

File:

```text
notebooks/04_model_training.ipynb
```

Mục tiêu:

- Huấn luyện nhiều mô hình học máy.
- So sánh mô hình ở bước đầu.
- Chuẩn bị kết quả cho bước đánh giá sâu.

Các mô hình được sử dụng:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- LightGBM
- Notebook-style Voting Ensemble
- Teacher-style Voting Ensemble

Hàm chính:

```python
trained_models, tuning_summary = train_all_models(bundle.X_train, bundle.y_train, config)
results_df, report_map = evaluate_models(trained_models, bundle.X_test, bundle.y_test)
```

Notebook này giúp chứng minh rằng nhóm không chỉ thử một mô hình mà có so sánh nhiều phương pháp khác nhau.

### 4.5. Notebook 05: Lựa chọn feature và tối ưu hóa nhẹ

File:

```text
notebooks/05_feature_selection_optimization.ipynb
```

Mục tiêu:

- Xác định feature nào quan trọng.
- Kiểm tra việc dùng ít feature hơn có cải thiện mô hình không.
- Lưu kết quả feature selection để Notebook 06 sử dụng.

Các kỹ thuật được sử dụng:

**Random Forest Feature Importance**

Random Forest có thuộc tính `feature_importances_`, cho biết mỗi feature đóng góp tương đối như thế nào vào quá trình dự đoán.

**Mutual Information**

`mutual_info_classif` đo mức độ liên hệ giữa từng feature và biến mục tiêu `Churn`. Feature có điểm cao thường chứa nhiều thông tin hơn cho việc dự đoán.

**SelectKBest**

Project dùng:

```python
SelectKBest(score_func=mutual_info_classif, k=12)
```

để chọn 12 feature quan trọng nhất.

Các feature được chọn gồm:

- `tenure`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

Kết quả so sánh feature:

| Model | Feature set | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Gradient Boosting | Full features | 0.7573 | 0.5333 | 0.6845 | 0.5995 | 0.8283 |
| Gradient Boosting | Selected features | 0.7651 | 0.5440 | 0.7112 | 0.6165 | 0.8339 |
| Logistic Regression | Full features | 0.7367 | 0.5028 | 0.7166 | 0.5910 | 0.8155 |
| Logistic Regression | Selected features | 0.7402 | 0.5072 | 0.7487 | 0.6048 | 0.8230 |
| Random Forest | Full features | 0.7658 | 0.5480 | 0.6711 | 0.6034 | 0.8280 |
| Random Forest | Selected features | 0.7665 | 0.5490 | 0.6738 | 0.6050 | 0.8300 |

Nhận xét:

- Selected features giúp cải thiện F1-score ở cả ba mô hình được thử.
- Mức cải thiện lớn nhất nằm ở Gradient Boosting và Logistic Regression.
- Vì vậy, Notebook 06 có thể sử dụng selected features cho bước đánh giá chính thức.

### 4.6. Notebook 06: Đánh giá và so sánh mô hình

File:

```text
notebooks/06_model_evaluation_comparison.ipynb
```

Mục tiêu:

- Đọc kết quả từ Notebook 05.
- Quyết định dùng full features hay selected features.
- Huấn luyện và đánh giá toàn bộ mô hình.
- Chọn mô hình tốt nhất.
- Lưu mô hình tốt nhất và các kết quả đánh giá.

Notebook 06 có liên kết trực tiếp với Notebook 05. Nếu file `feature_selection_comparison.csv` và `selected_features.csv` tồn tại, Notebook 06 sẽ đọc kết quả này để quyết định chiến lược feature.

Trong lần chạy hiện tại:

```text
Feature strategy: selected_features
Best model: xgboost
```

Kết quả đánh giá mô hình:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.7722 | 0.5565 | 0.6979 | 0.6192 | 0.8336 |
| Gradient Boosting | 0.7651 | 0.5440 | 0.7112 | 0.6165 | 0.8339 |
| Notebook Voting Ensemble | 0.7537 | 0.5258 | 0.7353 | 0.6132 | 0.8333 |
| AdaBoost | 0.7445 | 0.5126 | 0.7620 | 0.6129 | 0.8296 |
| Teacher Voting Ensemble | 0.7708 | 0.5556 | 0.6818 | 0.6122 | 0.8329 |
| Random Forest | 0.7686 | 0.5529 | 0.6711 | 0.6063 | 0.8302 |
| Logistic Regression | 0.7402 | 0.5072 | 0.7487 | 0.6048 | 0.8230 |
| LightGBM | 0.7708 | 0.5584 | 0.6524 | 0.6017 | 0.8235 |
| Decision Tree | 0.7388 | 0.5066 | 0.6203 | 0.5577 | 0.7987 |

Mô hình tốt nhất theo F1-score là:

```text
XGBoost
```

Các file được lưu sau Notebook 06:

```text
models/trained_models/best_model.pkl
models/metrics/model_results.csv
models/metrics/classification_report.txt
models/metrics/best_model_summary.csv
```

### 4.7. Notebook 07: Truyền đạt kết quả

File:

```text
notebooks/07_result_communication.ipynb
```

Mục tiêu:

- Đọc lại kết quả đã lưu từ Notebook 06.
- Tổng hợp mô hình tốt nhất.
- Tổng hợp insight chính.
- Chuẩn bị nội dung đưa vào slide và báo cáo.

Notebook 07 không train lại mô hình. Nó chỉ đọc kết quả đã lưu để trình bày phần cuối.

Kết quả tóm tắt:

```text
Best model: XGBoost
Feature strategy: selected_features
Accuracy: 0.7722
Precision: 0.5565
Recall: 0.6979
F1-score: 0.6192
ROC-AUC: 0.8336
```

## 5. Giải thích các thuật toán sử dụng

### 5.1. Logistic Regression

Logistic Regression là mô hình tuyến tính dùng cho bài toán phân loại. Mô hình học mối quan hệ giữa các feature và xác suất khách hàng churn.

Ưu điểm:

- Nhanh.
- Dễ hiểu.
- Phù hợp làm baseline.

Hạn chế:

- Khó bắt được quan hệ phi tuyến phức tạp.
- Có thể kém hơn các mô hình cây trên dữ liệu dạng bảng.

### 5.2. Decision Tree

Decision Tree chia dữ liệu theo các điều kiện dạng cây quyết định. Ví dụ mô hình có thể học các quy tắc kiểu:

```text
Nếu Contract = Month-to-month và tenure thấp thì khả năng churn cao.
```

Ưu điểm:

- Dễ giải thích.
- Trực quan.

Hạn chế:

- Dễ overfit nếu cây quá sâu.

### 5.3. Random Forest

Random Forest là tập hợp nhiều Decision Tree. Mỗi cây học trên một phần dữ liệu và một phần feature khác nhau, sau đó kết quả được tổng hợp lại.

Ưu điểm:

- Ổn định hơn Decision Tree.
- Giảm overfitting.
- Có thể xem feature importance.

### 5.4. Gradient Boosting

Gradient Boosting xây dựng nhiều cây theo thứ tự. Cây sau tập trung sửa lỗi của cây trước.

Ý tưởng:

```text
Mô hình đầu tiên dự đoán chưa tốt.
Mô hình tiếp theo học phần sai còn lại.
Lặp lại nhiều lần để giảm lỗi.
```

Ưu điểm:

- Mạnh trên dữ liệu dạng bảng.
- Thường cho kết quả tốt.

### 5.5. AdaBoost

AdaBoost cũng là mô hình boosting. Nó tăng trọng số cho các mẫu bị dự đoán sai, để mô hình sau chú ý hơn đến những trường hợp khó.

Trong bài toán churn, AdaBoost có recall khá cao, nghĩa là bắt được nhiều khách hàng churn thật.

### 5.6. XGBoost

XGBoost là phiên bản boosting mạnh và tối ưu hơn. Nó thường đạt hiệu quả cao trên dữ liệu tabular như bộ Telco Customer Churn.

Ưu điểm:

- Hiệu năng tốt.
- Có cơ chế regularization giúp giảm overfitting.
- Phù hợp với bài toán phân loại nhị phân.

Trong project hiện tại, XGBoost là mô hình tốt nhất theo F1-score.

### 5.7. LightGBM

LightGBM là mô hình gradient boosting được tối ưu để train nhanh và xử lý dữ liệu lớn hiệu quả.

Ưu điểm:

- Nhanh.
- Mạnh trên dữ liệu dạng bảng.

### 5.8. Voting Ensemble

Voting Ensemble kết hợp nhiều mô hình lại để đưa ra dự đoán cuối cùng. Project có hai dạng ensemble:

- Notebook-style Voting Ensemble.
- Teacher-style Voting Ensemble.

Mục đích là thử cách kết hợp nhiều mô hình mạnh thay vì chỉ phụ thuộc vào một mô hình đơn lẻ.

## 6. Giải thích các chỉ số đánh giá

### 6.1. Accuracy

Accuracy là tỷ lệ dự đoán đúng trên toàn bộ tập test.

Tuy nhiên, với bài toán churn, accuracy không phải chỉ số duy nhất cần quan tâm vì dữ liệu có thể mất cân bằng.

### 6.2. Precision

Precision trả lời câu hỏi:

```text
Trong số khách hàng mô hình dự đoán là churn, có bao nhiêu người thật sự churn?
```

Precision cao giúp giảm báo động giả.

### 6.3. Recall

Recall trả lời câu hỏi:

```text
Trong số khách hàng thật sự churn, mô hình phát hiện được bao nhiêu?
```

Trong bài toán churn, recall rất quan trọng vì doanh nghiệp muốn phát hiện càng nhiều khách hàng có nguy cơ rời bỏ càng tốt.

### 6.4. F1-score

F1-score là chỉ số cân bằng giữa precision và recall.

Project chọn F1-score làm chỉ số chính vì bài toán cần cân bằng giữa:

- phát hiện đúng khách hàng churn,
- và hạn chế dự đoán sai quá nhiều.

### 6.5. ROC-AUC

ROC-AUC đo khả năng mô hình phân biệt giữa hai lớp churn và non-churn. Giá trị càng gần 1 thì khả năng phân tách càng tốt.

Trong project này, các mô hình tốt đạt ROC-AUC khoảng 0.83, cho thấy mô hình có khả năng phân biệt hai nhóm khách hàng tương đối tốt.

## 7. Kết quả và nhận xét

### 7.1. Mô hình tốt nhất

Mô hình tốt nhất theo F1-score là:

```text
XGBoost
```

Kết quả:

```text
Accuracy  = 0.7722
Precision = 0.5565
Recall    = 0.6979
F1-score  = 0.6192
ROC-AUC   = 0.8336
```

### 7.2. Vì sao không chỉ chọn theo Accuracy?

LightGBM và Teacher Voting Ensemble có accuracy khá cao, nhưng XGBoost có F1-score tốt nhất. Với bài toán churn, F1-score quan trọng hơn accuracy vì nó cân bằng giữa precision và recall.

Nếu chỉ nhìn accuracy, mô hình có thể dự đoán tốt lớp đông hơn nhưng bỏ sót khách hàng churn. Điều này không phù hợp với mục tiêu kinh doanh.

### 7.3. Ý nghĩa kết quả

Kết quả cho thấy mô hình có thể hỗ trợ doanh nghiệp xác định nhóm khách hàng có nguy cơ rời bỏ. Dù F1-score chưa quá cao, ROC-AUC đạt khoảng 0.83 cho thấy mô hình có khả năng phân biệt tốt giữa hai nhóm khách hàng.

Trong thực tế, doanh nghiệp có thể dùng mô hình này để:

- lọc ra nhóm khách hàng rủi ro cao,
- ưu tiên chăm sóc khách hàng có nguy cơ churn,
- thiết kế ưu đãi theo từng nhóm khách hàng,
- cải thiện dịch vụ hỗ trợ kỹ thuật và bảo mật.

## 8. Các insight nghiệp vụ quan trọng

Từ EDA và feature selection, một số yếu tố quan trọng liên quan đến churn gồm:

- `tenure`: khách hàng mới thường có nguy cơ churn cao hơn.
- `Contract`: hợp đồng ngắn hạn có rủi ro churn cao hơn hợp đồng dài hạn.
- `MonthlyCharges`: chi phí hàng tháng cao có thể làm tăng rủi ro churn.
- `TotalCharges`: phản ánh mức độ gắn bó và tổng chi tiêu của khách hàng.
- `InternetService`: loại dịch vụ internet có liên quan đến churn.
- `OnlineSecurity`: khách hàng không dùng bảo mật trực tuyến có thể dễ churn hơn.
- `TechSupport`: thiếu hỗ trợ kỹ thuật có thể làm tăng khả năng churn.
- `PaymentMethod`: phương thức thanh toán có liên quan đến hành vi rời bỏ.

Các insight này giúp doanh nghiệp không chỉ dự đoán churn mà còn hiểu nguyên nhân tiềm năng phía sau hành vi churn.

## 9. Hạn chế của project

Project vẫn còn một số hạn chế:

- Chưa tuning sâu toàn bộ mô hình bằng GridSearchCV hoặc RandomizedSearchCV.
- Chưa dùng các phương pháp giải thích mô hình nâng cao như SHAP hoặc LIME.
- Dữ liệu không có yếu tố thời gian chi tiết nên không triển khai theo hướng time series.
- Feature encoding hiện dùng LabelEncoder, có thể nâng cấp bằng OneHotEncoder cho một số mô hình tuyến tính.
- Mô hình chưa được triển khai thành dashboard hoặc web app.

## 10. Hướng phát triển

Trong tương lai, project có thể mở rộng theo các hướng:

- Tuning sâu XGBoost, LightGBM và Random Forest.
- Thử thêm RFE hoặc RFECV cho feature selection.
- Dùng SHAP để giải thích từng dự đoán cụ thể.
- Xây dựng dashboard trực quan hóa churn risk.
- Tạo form dự đoán churn cho khách hàng mới.
- Điều chỉnh threshold để tăng recall nếu doanh nghiệp ưu tiên bắt nhiều khách churn hơn.

## 11. Kết luận

Project đã hoàn thành đầy đủ quy trình phân tích dữ liệu và học máy cho bài toán Customer Churn Prediction:

1. Thu thập dữ liệu.
2. Làm sạch và tiền xử lý dữ liệu.
3. Khai phá và phân tích dữ liệu.
4. Huấn luyện nhiều mô hình học máy.
5. Lựa chọn feature và kiểm tra tác động của selected features.
6. Đánh giá và so sánh mô hình.
7. Truyền đạt kết quả và rút ra insight.

Kết quả tốt nhất hiện tại thuộc về mô hình XGBoost với F1-score khoảng 0.6192 và ROC-AUC khoảng 0.8336. Điều này cho thấy mô hình có khả năng hỗ trợ doanh nghiệp nhận diện khách hàng có nguy cơ rời bỏ dịch vụ.

Về mặt nghiệp vụ, các yếu tố như thời gian gắn bó, loại hợp đồng, phí hàng tháng, dịch vụ hỗ trợ và phương thức thanh toán là những biến quan trọng cần được chú ý khi xây dựng chiến lược giữ chân khách hàng.

## 12. Kịch bản trình bày ngắn gọn trước giảng viên

Khi thuyết trình, có thể trình bày theo luồng sau:

1. Nhóm chọn đề tài churn prediction vì bài toán có ý nghĩa thực tế trong việc giữ chân khách hàng.
2. Nhóm sử dụng bộ dữ liệu Telco Customer Churn, trong đó biến mục tiêu là `Churn`.
3. Nhóm làm sạch dữ liệu, xử lý `TotalCharges`, loại bỏ `customerID`, encode dữ liệu phân loại, chia train/test, dùng SMOTENC để xử lý mất cân bằng và dùng StandardScaler cho biến số.
4. Nhóm thực hiện EDA để phân tích churn theo hợp đồng, tenure, monthly charges, internet service, online security, tech support và payment method.
5. Nhóm huấn luyện nhiều mô hình như Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, LightGBM và Voting Ensemble.
6. Nhóm thực hiện feature selection bằng Random Forest importance, Mutual Information và SelectKBest. Kết quả selected features giúp cải thiện một số mô hình.
7. Nhóm đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score và ROC-AUC. Mô hình tốt nhất là XGBoost theo F1-score.
8. Cuối cùng, nhóm rút ra insight nghiệp vụ: khách hàng hợp đồng ngắn hạn, tenure thấp, monthly charges cao hoặc thiếu dịch vụ hỗ trợ có nguy cơ churn cao hơn.

## 13. Tài liệu và file liên quan

Các file chính trong project:

```text
notebooks/01_data_collection.ipynb
notebooks/02_data_cleaning_preprocessing.ipynb
notebooks/03_eda_analysis.ipynb
notebooks/04_model_training.ipynb
notebooks/05_feature_selection_optimization.ipynb
notebooks/06_model_evaluation_comparison.ipynb
notebooks/07_result_communication.ipynb
src/data/preprocess.py
src/models/train.py
src/models/evaluate.py
models/trained_models/best_model.pkl
models/metrics/model_results.csv
models/metrics/best_model_summary.csv
reports/figures/
```
