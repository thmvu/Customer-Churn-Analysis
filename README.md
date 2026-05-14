# Customer Churn Analysis

Du an nay to chuc bai toan `Telco Customer Churn` theo huong de nop bai, chay local, va day len Git sach se. Workflow duoc ket hop tu:

- Notebook goc cua ban: EDA, preprocessing, train nhieu mo hinh, confusion matrix, ROC.
- Phong cach cua giang vien: bo sung `SMOTE`, so sanh nhieu kich ban, va them ensemble de nang cap ket qua.

## Cau truc thu muc

```text
Customer-Churn-Analysis/
├── configs/
│   └── config.yaml
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── metrics/
│   └── trained_models/
├── notebooks/
│   ├── experiments/
│   └── README.md
├── reports/
│   ├── figures/
│   ├── final_report/
│   └── presentation/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── tests/
├── .gitignore
├── LICENSE
├── main.py
└── requirements.txt
```

## Step-by-step de chay local

### 1. Tao moi truong ao

```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Cai thu vien

```powershell
pip install -r requirements.txt
```

### 3. Kiem tra du lieu goc

Dam bao file CSV nam o:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 4. Chay toan bo pipeline

```powershell
python main.py
```

Pipeline se tu dong:

1. Doc du lieu goc.
2. Lam sach cot `TotalCharges`, bo `customerID`.
3. Encode categorical features.
4. Chia `train/test`.
5. Can bang du lieu bang `SMOTENC` neu bat trong config.
6. Scale cac cot so.
7. Train nhieu model:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - AdaBoost
   - XGBoost neu da cai
   - LightGBM neu da cai
   - Notebook-style Voting Ensemble
   - Teacher-style Voting Ensemble
8. Luu bang ket qua, classification report, hinh confusion matrix, ROC curve, feature importance.
9. Luu model tot nhat vao `models/trained_models/`.

## Artifacts du kien sau khi chay

- `data/processed/cleaned_data.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`
- `models/metrics/model_results.csv`
- `models/metrics/classification_report.txt`
- `models/trained_models/best_model.pkl`
- `reports/figures/*.png`

## Giai thich nang cap so voi notebook goc

- `SMOTENC` duoc dung thay vi `SMOTE` thuong de xu ly du lieu co nhieu cot categorical mot cach hop ly hon.
- Co them hai ensemble:
  - `notebook_voting_ensemble`: gan voi logic trong notebook cua ban.
  - `teacher_voting_ensemble`: gan voi huong ensemble cua giang vien.
- Co file `config.yaml` de ban co the bat/tat `SMOTE`, tuning, danh sach model, va duong dan output.

## Goi y chia notebook

- `01_eda_visualization.ipynb`: churn distribution, churn by gender, contract, monthly charges.
- `02_preprocessing.ipynb`: clean `TotalCharges`, encode, split, SMOTENC, scale.
- `03_model_training.ipynb`: train va so sanh model.
- `04_model_evaluation.ipynb`: confusion matrix, ROC, feature importance.

## Test co ban

```powershell
pytest
```

## Huong Git commit dep

```powershell
git add .
git commit -m "Scaffold customer churn project structure"
git commit -m "Implement preprocessing and SMOTENC pipeline"
git commit -m "Add baseline and ensemble model comparison"
git commit -m "Add README and reporting artifacts"
```
