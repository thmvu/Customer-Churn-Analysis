# Báo cáo chi tiết dự án Customer Churn Prediction

## 1. Giới thiệu đề tài

### 1.1. Tên đề tài

Phân tích và dự đoán khả năng khách hàng rời bỏ dịch vụ viễn thông bằng Machine Learning.

### 1.2. Bối cảnh bài toán

Trong lĩnh vực viễn thông, việc giữ chân khách hàng cũ thường rẻ hơn rất nhiều so với việc tìm khách hàng mới. Vì vậy, nếu doanh nghiệp có thể dự đoán sớm khách hàng nào có nguy cơ rời bỏ dịch vụ, họ có thể:

- đưa ra chương trình khuyến mãi phù hợp,
- cải thiện dịch vụ hỗ trợ,
- điều chỉnh chính sách hợp đồng,
- và tối ưu chi phí chăm sóc khách hàng.

Từ đó, bài toán churn prediction có ý nghĩa thực tế rất rõ ràng: **dự đoán khách hàng nào có khả năng rời bỏ dịch vụ để doanh nghiệp can thiệp kịp thời**.

### 1.3. Mục tiêu nghiên cứu

Dự án này hướng đến 4 mục tiêu chính:

1. Hiểu cấu trúc và đặc điểm của dữ liệu khách hàng.
2. Xác định các yếu tố có liên quan mạnh đến hành vi rời bỏ dịch vụ.
3. Xây dựng và so sánh nhiều mô hình Machine Learning để dự đoán churn.
4. Chọn mô hình phù hợp nhất và giải thích kết quả theo góc nhìn nghiệp vụ.

## 2. Mô tả dữ liệu

### 2.1. Bộ dữ liệu sử dụng

Bộ dữ liệu được sử dụng là **Telco Customer Churn**, trong đó:

- mỗi dòng biểu diễn một khách hàng,
- mỗi cột biểu diễn một đặc điểm hoặc hành vi của khách hàng,
- cột mục tiêu là `Churn`.

### 2.2. Ý nghĩa các nhóm cột chính

Các cột trong dữ liệu có thể chia thành các nhóm sau:

#### Nhóm thông tin cá nhân

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`

#### Nhóm thời gian và mức phí

- `tenure`
- `MonthlyCharges`
- `TotalCharges`

#### Nhóm dịch vụ khách hàng đang dùng

- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`

#### Nhóm hành vi thanh toán và hợp đồng

- `Contract`
- `PaperlessBilling`
- `PaymentMethod`

#### Biến mục tiêu

- `Churn`

### 2.3. Cột không dùng để train

- `customerID`

Lý do:

- đây chỉ là mã định danh,
- không mang giá trị dự đoán,
- giữ lại chỉ làm tăng nhiễu cho mô hình.

## 3. Luồng hoạt động tổng thể của project

Toàn bộ project được tổ chức theo flow như sau:

1. Đọc dữ liệu từ file CSV.
2. Làm sạch dữ liệu.
3. Phân tích khám phá dữ liệu bằng biểu đồ.
4. Mã hóa dữ liệu để mô hình có thể học được.
5. Chia dữ liệu thành train/test.
6. Cân bằng dữ liệu bằng `SMOTENC`.
7. Chuẩn hóa các biến số.
8. Huấn luyện nhiều mô hình.
9. So sánh mô hình theo nhiều chỉ số.
10. Chọn mô hình tốt nhất.
11. Phân tích confusion matrix, ROC curve, feature importance.
12. Rút ra kết luận và insight nghiệp vụ.

Nói ngắn gọn hơn, dự án đi theo chuỗi:

**Dữ liệu thô -> làm sạch -> EDA -> chuẩn bị dữ liệu -> train model -> đánh giá -> giải thích**

## 4. Cấu trúc project và vai trò của từng notebook

### 4.1. Cấu trúc thư mục

Project được chia thành các thư mục chính:

- `data/`: dữ liệu gốc và dữ liệu đã xử lý
- `notebooks/`: nơi chạy các notebook phân tích
- `src/`: code chính của pipeline
- `models/`: lưu kết quả và model đã train
- `reports/`: lưu biểu đồ và báo cáo cuối cùng
- `configs/`: file cấu hình

### 4.2. Vai trò của từng notebook

#### `01_data_understanding_eda.ipynb`

Dùng để:

- hiểu dữ liệu,
- kiểm tra phân bố churn,
- phân tích các yếu tố liên quan đến churn bằng nhiều biểu đồ cột, tròn, histogram, boxplot.

#### `02_preprocessing_pipeline.ipynb`

Dùng để:

- giải thích rõ phần tiền xử lý,
- xử lý `TotalCharges`,
- bỏ `customerID`,
- encode dữ liệu,
- cân bằng dữ liệu bằng `SMOTENC`,
- scale các biến số.

#### `03_model_training_comparison.ipynb`

Dùng để:

- train nhiều mô hình,
- so sánh kết quả bằng biểu đồ cột,
- xem mô hình nào mạnh hơn theo từng metric.

#### `04_model_evaluation_and_interpretation.ipynb`

Dùng để:

- đánh giá sâu mô hình tốt nhất,
- xem confusion matrix,
- xem ROC curve,
- xem feature importance,
- rút ra kết luận cuối cùng.

## 5. Phân tích khám phá dữ liệu (EDA)

Phần EDA rất quan trọng vì nó giúp hiểu dữ liệu trước khi train model. Thay vì nhảy ngay vào machine learning, ta cần trả lời:

- Tỷ lệ churn là bao nhiêu?
- Nhóm khách hàng nào rời bỏ nhiều hơn?
- Hợp đồng nào rủi ro cao nhất?
- Mức phí hàng tháng có liên quan đến churn không?
- Dịch vụ hỗ trợ có làm giảm churn không?

### 5.1. Phân bố churn tổng thể

Notebook đầu tiên vẽ:

- sơ đồ cột cho số lượng `Churn = Yes` và `Churn = No`,
- sơ đồ tròn để thể hiện tỷ lệ churn tổng thể.

Ý nghĩa:

- giúp xác định bài toán có mất cân bằng dữ liệu hay không,
- cho thấy có đủ dữ liệu churn để học nhưng vẫn có chênh lệch giữa hai lớp.

### 5.2. Churn theo giới tính và nhóm khách hàng lớn tuổi

Các biểu đồ cột theo:

- `gender`
- `SeniorCitizen`

giúp kiểm tra xem churn có khác nhau giữa các nhóm khách hàng hay không.

Ý nghĩa:

- nếu nhóm khách hàng cao tuổi churn cao hơn, doanh nghiệp có thể xây gói chăm sóc riêng cho nhóm này.

### 5.3. Churn theo tình trạng gia đình

Các biến:

- `Partner`
- `Dependents`

giúp phản ánh phần nào sự ổn định của khách hàng.

Ý nghĩa:

- khách hàng có gia đình hoặc người phụ thuộc có thể có xu hướng gắn bó khác với khách hàng độc lập.

### 5.4. Churn theo loại hợp đồng

Đây là một trong những biểu đồ quan trọng nhất.

Thông thường:

- `Month-to-month` có tỷ lệ churn cao hơn,
- `One year` và `Two year` ổn định hơn.

Ý nghĩa nghiệp vụ:

- hợp đồng ngắn hạn thường khiến khách hàng rời bỏ dễ hơn,
- các hợp đồng dài hạn giúp tăng mức độ gắn bó.

### 5.5. Churn theo phương thức thanh toán

Biểu đồ này giúp xem phương thức thanh toán nào có tỷ lệ churn cao.

Ý nghĩa:

- một số hình thức thanh toán có thể gắn liền với trải nghiệm dịch vụ hoặc hành vi tiêu dùng khác nhau,
- từ đó doanh nghiệp có thể tối ưu luồng thanh toán.

### 5.6. Churn theo dịch vụ Internet và dịch vụ hỗ trợ

Các biến được phân tích mạnh ở notebook gồm:

- `InternetService`
- `OnlineSecurity`
- `TechSupport`

Ý nghĩa:

- khách hàng không có hỗ trợ kỹ thuật hoặc bảo mật trực tuyến thường dễ churn hơn,
- đây là insight rất tốt để đưa vào phần nhận xét.

### 5.7. Phân phối các biến số quan trọng

Các biểu đồ được vẽ:

- histogram của `tenure`,
- boxplot của `MonthlyCharges`,
- boxplot của `TotalCharges`.

Ý nghĩa:

- khách hàng mới sử dụng dịch vụ thường có churn cao hơn,
- khách hàng trả phí hàng tháng cao có thể nhạy cảm hơn với việc rời bỏ,
- `TotalCharges` phản ánh mức độ gắn bó dài hạn.

### 5.8. Churn theo nhóm tenure

Việc chia `tenure` thành các khoảng như:

- `0-12`
- `13-24`
- `25-48`
- `49-72`

giúp bài thuyết trình dễ hiểu hơn rất nhiều.

Ý nghĩa:

- thay vì nói bằng số thô, bạn có thể nói theo nhóm khách hàng mới, trung bình, lâu năm.

### 5.9. Kết luận phần EDA

Sau khi chạy notebook EDA, có thể rút ra các ý quan trọng:

- churn tồn tại ở mức đủ lớn để trở thành một bài toán kinh doanh nghiêm túc,
- `Contract`, `tenure`, `MonthlyCharges`, `PaymentMethod`, `InternetService`, `OnlineSecurity`, `TechSupport` là các biến đáng chú ý,
- khách hàng hợp đồng ngắn hạn và mức phí cao có xu hướng churn nhiều hơn,
- các dịch vụ hỗ trợ tốt có thể giúp giảm rủi ro churn.

## 6. Tiền xử lý dữ liệu

Phần preprocessing là bước biến dữ liệu thô thành dữ liệu có thể dùng cho machine learning.

### 6.1. Làm sạch dữ liệu text

Trong code, các cột text được:

- chuyển về dạng string,
- loại bỏ khoảng trắng đầu và cuối.

Mục đích:

- tránh lỗi do dữ liệu nhập không đồng nhất,
- giữ dữ liệu ổn định hơn trước khi encode.

### 6.2. Xử lý cột `TotalCharges`

Đây là cột đặc biệt quan trọng.

Vấn đề:

- một số hàng chứa khoảng trắng,
- nên nếu đọc trực tiếp thì cột này không phải dạng số đúng nghĩa.

Cách xử lý:

- dùng `pd.to_numeric(..., errors='coerce')`,
- chuyển giá trị lỗi thành `NaN`,
- điền `median`.

Vì sao dùng median:

- ít bị ảnh hưởng bởi giá trị ngoại lệ,
- an toàn hơn `mean` trong dữ liệu có độ lệch.

### 6.3. Loại bỏ `customerID`

Lý do:

- không mang thông tin dự đoán,
- chỉ là chỉ mục nhận diện khách hàng,
- giữ lại có thể làm mô hình học nhiễu.

### 6.4. Mã hóa dữ liệu phân loại

Các model không học trực tiếp từ text như:

- `Male`
- `Female`
- `Yes`
- `No`
- `Month-to-month`

Vì vậy, notebook và code dùng `LabelEncoder` để chuyển chúng về số.

Ví dụ:

- `Yes -> 1`
- `No -> 0`

### 6.5. Tách feature và target

Sau encode:

- `X` là toàn bộ feature
- `y` là cột `Churn`

Mục tiêu là dự đoán `Churn` từ các biến còn lại.

### 6.6. Chia train/test

Project dùng:

- `test_size = 0.2`
- `stratify = y`

Ý nghĩa:

- 80 phần trăm dữ liệu để train,
- 20 phần trăm để test,
- giữ tỷ lệ churn giữa train và test gần giống nhau.

### 6.7. Cân bằng dữ liệu bằng `SMOTENC`

Đây là điểm nâng cấp quan trọng theo hướng của giảng viên.

Vấn đề:

- số lượng khách hàng churn thường ít hơn non-churn,
- nếu giữ nguyên, model dễ thiên về lớp đông hơn.

Giải pháp:

- dùng `SMOTENC` để tăng số mẫu của lớp thiểu số trong tập train.

Vì sao là `SMOTENC` chứ không phải `SMOTE` thường:

- dataset có cả numeric và categorical,
- `SMOTENC` phù hợp hơn cho trường hợp có feature phân loại.

### 6.8. Scale các biến số

Các cột số được scale như:

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Ý nghĩa:

- đưa các cột số về cùng thang đo,
- giúp các model như Logistic Regression hoạt động ổn định hơn.

### 6.9. Tương quan với `Churn`

Notebook preprocessing còn thêm:

- biểu đồ thanh cho top biến có tương quan với `Churn`,
- heatmap của các biến được chọn.

Ý nghĩa:

- giúp biết biến nào đang liên quan mạnh nhất đến mục tiêu,
- hỗ trợ bạn giải thích vì sao các cột này đáng dùng để train.

### 6.10. Kết luận phần preprocessing

Sau bước này:

- dữ liệu sạch hơn,
- feature đã được encode,
- tập train đã cân bằng hơn nhờ `SMOTENC`,
- dữ liệu sẵn sàng cho bước train model.

## 7. Các cột nên dùng để train model

Các cột nên giữ gần như đầy đủ, gồm:

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

Biến mục tiêu:

- `Churn`

Không dùng:

- `customerID`

## 8. Các mô hình sử dụng trong dự án

Project không chỉ train một model mà so sánh nhiều mô hình để chọn ra phương án tốt nhất.

### 8.1. Logistic Regression

Đây là model baseline.

Ưu điểm:

- đơn giản,
- nhanh,
- dễ giải thích.

Nhược điểm:

- có thể không mạnh bằng các model boosting trên dữ liệu bảng.

### 8.2. Decision Tree

Ưu điểm:

- dễ hiểu,
- có thể diễn giải bằng dạng cây.

Nhược điểm:

- dễ overfit nếu không kiểm soát tốt.

### 8.3. Random Forest

Là tập hợp nhiều cây quyết định.

Ưu điểm:

- ổn định hơn Decision Tree,
- thường cho kết quả tốt trên dữ liệu bảng.

### 8.4. Gradient Boosting

Ý tưởng:

- cây sau học cách sửa lỗi của cây trước.

Ưu điểm:

- mạnh trên dữ liệu tabular,
- thường cho kết quả tốt hơn các model cơ bản.

### 8.5. AdaBoost

Ý tưởng:

- tăng trọng số cho các mẫu bị dự đoán sai,
- để mô hình học tập trung hơn vào các trường hợp khó.

### 8.6. XGBoost

Ưu điểm:

- rất mạnh với dữ liệu bảng,
- hiệu quả cao,
- thường là một trong các model tốt nhất trong bài toán churn.

### 8.7. LightGBM

Ưu điểm:

- huấn luyện nhanh,
- hiệu quả tốt,
- phù hợp với bài toán tabular.

### 8.8. Notebook-style Voting Ensemble

Mô hình này kết hợp:

- Gradient Boosting
- Logistic Regression
- AdaBoost

Mục tiêu:

- bám theo tinh thần notebook gốc,
- tận dụng lợi thế của nhiều model cùng lúc.

### 8.9. Teacher-style Voting Ensemble

Mô hình này kết hợp ưu tiên:

- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting

Mục tiêu:

- gần với định hướng ensemble mạnh của giảng viên,
- tận dụng các model tabular mạnh trong cùng một hệ.

## 9. Cách huấn luyện và so sánh mô hình

Notebook thứ ba sẽ:

1. train toàn bộ model,
2. thu kết quả vào bảng,
3. vẽ biểu đồ cột so sánh theo:
   - `accuracy`
   - `precision`
   - `recall`
   - `f1`
   - `roc_auc`

### 9.1. Vì sao không chỉ nhìn Accuracy?

Trong bài toán churn:

- nếu model đoán hầu hết là `No Churn`, accuracy vẫn có thể nhìn đẹp,
- nhưng model lại không bắt được khách có nguy cơ rời bỏ.

Do đó, cần xem thêm:

- `precision`
- `recall`
- `f1-score`
- `roc_auc`

### 9.2. Ý nghĩa từng chỉ số

#### Accuracy

Tỷ lệ dự đoán đúng trên toàn bộ tập test.

#### Precision

Trong số các khách hàng bị dự đoán là churn, có bao nhiêu người thực sự churn.

#### Recall

Trong số khách hàng churn thật, mô hình phát hiện được bao nhiêu.

#### F1-score

Là trung bình điều hòa giữa precision và recall.

Đây là chỉ số rất quan trọng khi muốn cân bằng giữa:

- bắt được churn,
- nhưng không báo động giả quá nhiều.

#### ROC-AUC

Đo khả năng phân tách hai lớp của mô hình.

Giá trị càng gần 1 càng tốt.

### 9.3. Kết luận phần so sánh model

Sau notebook training, bạn nên chốt:

- model nào mạnh nhất theo `F1-score`,
- model nào mạnh theo `ROC-AUC`,
- ensemble có cải thiện hay không,
- và vì sao nhóm chọn model cuối cùng.

## 10. Đánh giá sâu mô hình tốt nhất

Notebook thứ tư tập trung vào mô hình tốt nhất.

### 10.1. Classification report

Classification report cho biết:

- precision
- recall
- f1-score

ở từng lớp.

Ý nghĩa:

- giúp nhìn rõ model xử lý lớp churn tốt đến đâu.

### 10.2. Confusion Matrix

Confusion matrix giúp trả lời:

- model đúng bao nhiêu khách churn,
- model bỏ sót bao nhiêu khách churn,
- model nhầm bao nhiêu khách không churn thành churn.

Ý nghĩa nghiệp vụ:

- nếu bỏ sót quá nhiều khách churn, doanh nghiệp sẽ mất cơ hội giữ chân,
- nếu báo động giả quá nhiều, doanh nghiệp sẽ tốn nguồn lực không cần thiết.

### 10.3. ROC Curve

ROC Curve cho biết:

- mô hình có phân biệt tốt hai lớp hay không,
- ngưỡng phân loại có đang hợp lý không.

Đi cùng ROC Curve là AUC:

- càng cao càng tốt,
- thể hiện mô hình có khả năng tách lớp mạnh.

### 10.4. Feature Importance

Đây là phần rất quan trọng trong báo cáo.

Mục tiêu:

- không chỉ biết model đúng hay sai,
- mà còn biết model đang dựa vào yếu tố nào để dự đoán.

Các biến thường đáng chú ý nhất:

- `Contract`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `InternetService`
- `TechSupport`
- `OnlineSecurity`
- `PaymentMethod`

### 10.5. Ý nghĩa nghiệp vụ của feature importance

Nếu các biến trên có importance cao, doanh nghiệp có thể:

- thiết kế lại chính sách hợp đồng,
- hỗ trợ khách hàng mới tốt hơn,
- đưa gói hỗ trợ kỹ thuật hợp lý,
- tối ưu trải nghiệm thanh toán,
- tập trung giữ chân nhóm có rủi ro cao.

## 11. Giải thích code theo cách dễ hiểu

### 11.1. `clean_dataframe(...)`

Hàm này chịu trách nhiệm:

- làm sạch dữ liệu text,
- xử lý `TotalCharges`,
- điền giá trị thiếu,
- loại bỏ `customerID`.

### 11.2. `encode_dataframe(...)`

Hàm này:

- tìm cột numeric,
- tìm cột categorical,
- encode các cột phân loại sang dạng số.

### 11.3. `split_scale_balance(...)`

Hàm này:

- chia train/test,
- cân bằng dữ liệu train bằng `SMOTENC`,
- scale các cột số.

### 11.4. `preprocess_pipeline(...)`

Đây là hàm gộp:

- clean dữ liệu,
- encode,
- split,
- balance,
- scale.

Nó trả về một bundle để dùng xuyên suốt các notebook.

### 11.5. `train_all_models(...)`

Hàm này:

- tạo toàn bộ model,
- train từng model,
- tạo thêm các ensemble.

### 11.6. `evaluate_models(...)`

Hàm này:

- chạy dự đoán trên tập test,
- tính metric cho từng model,
- trả về bảng kết quả để so sánh.

## 12. Cách bạn nên trình bày trước giảng viên

Bạn có thể nói theo đúng luồng sau:

1. Nhóm chọn bài toán churn vì có ý nghĩa thực tế cao.
2. Nhóm sử dụng bộ dữ liệu Telco Customer Churn.
3. Nhóm làm sạch dữ liệu, đặc biệt xử lý `TotalCharges` và bỏ `customerID`.
4. Nhóm phân tích EDA để tìm insight ban đầu.
5. Nhóm encode dữ liệu, chia train/test và cân bằng dữ liệu bằng `SMOTENC`.
6. Nhóm huấn luyện nhiều mô hình khác nhau để so sánh khách quan.
7. Nhóm đánh giá mô hình bằng nhiều metric chứ không chỉ accuracy.
8. Nhóm chọn mô hình tốt nhất dựa trên F1-score, ROC-AUC và chất lượng tổng thể.
9. Nhóm giải thích các yếu tố ảnh hưởng mạnh đến churn bằng feature importance.
10. Từ đó, nhóm rút ra hàm ý nghiệp vụ cho doanh nghiệp.

## 13. Kết luận chung của dự án

Về mặt kỹ thuật, dự án này đầy đủ vì:

- có làm sạch dữ liệu,
- có EDA,
- có xử lý mất cân bằng,
- có chuẩn hóa dữ liệu,
- có so sánh nhiều mô hình,
- có ensemble,
- có đánh giá bằng nhiều metric,
- có giải thích đặc trưng quan trọng.

Về mặt trình bày, dự án này mạnh vì:

- bài toán thực tế,
- dữ liệu dễ hiểu,
- nhiều biểu đồ đẹp,
- có thể kể thành câu chuyện nghiệp vụ rõ ràng,
- phù hợp cả phong cách data analysis lẫn machine learning.

## 14. Những ý chốt có thể đưa thẳng vào báo cáo hoặc slide

Bạn có thể dùng các ý sau để kết luận:

- Bài toán churn prediction giúp doanh nghiệp xác định sớm khách hàng có nguy cơ rời bỏ.
- Phân tích dữ liệu cho thấy hợp đồng, thời gian gắn bó, chi phí hàng tháng và dịch vụ hỗ trợ là những yếu tố quan trọng.
- Việc làm sạch dữ liệu và cân bằng dữ liệu bằng `SMOTENC` giúp cải thiện chất lượng đầu vào cho mô hình.
- So sánh nhiều model cho thấy các mô hình boosting và ensemble thường cho hiệu quả tốt hơn model baseline.
- Mô hình tốt nhất không chỉ được chọn vì accuracy, mà còn vì khả năng cân bằng giữa precision, recall và F1-score.
- Feature importance giúp nhóm không chỉ dự đoán churn mà còn hiểu tại sao churn xảy ra.

## 15. Bước tiếp theo bạn nên làm

1. Chạy lần lượt 4 notebook.
2. Chọn 5 đến 7 biểu đồ đẹp nhất để đưa vào slide.
3. Lấy bảng so sánh model trong notebook thứ 3.
4. Lấy confusion matrix, ROC curve, feature importance trong notebook thứ 4.
5. Viết phần kết luận dựa trên insight từ EDA và model tốt nhất.
6. Khi thuyết trình, bám đúng flow của report này để nói sẽ rất mượt.
