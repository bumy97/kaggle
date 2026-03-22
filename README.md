# 🚀 Kaggle Playground Series: Customer Churn Prediction

이 저장소는 캐글(Kaggle)의 **"고객 이탈률 예측(Customer Churn Prediction)"** 대회를 위한 데이터 분석 및 머신러닝 파이프라인 코드를 포함하고 있습니다. 

## 📌 프로젝트 개요
- **주제**: 통신사 고객 데이터를 바탕으로 이탈(Churn) 여부 예측
- **목표**: 인구통계학적 정보와 서비스 이용 내역을 분석하여 고객 이탈 확률 계산
- **평가 지표**: ROC-AUC Score (현재 베이스라인 기준 **0.913+** 달성)

## 🛠 주요 처리 과정 (Pipeline)

### 1. 데이터 전처리 (Preprocessing)
- **Label Encoding**: 타겟 변수(`Churn`)를 모델 학습이 가능하도록 1(Yes)과 0(No)으로 변환.
- **Categorical Encoding**: 범주형 변수(`Contract`, `PaymentMethod` 등)를 `pd.get_dummies`를 통해 수치형 데이터로 변환.
- **Data Integrity**: `TotalCharges` 내의 결측치를 `MonthlyCharges` 데이터를 참조하여 보정 및 수치형 변환.

### 2. 피처 엔지니어링 (Feature Engineering)
모델의 변별력을 높이기 위해 도메인 지식을 기반으로 다음의 파생 변수를 생성했습니다:
- **ChargeDiff (요금 급증 확인)**: 현재 월 요금이 과거 평균 요금보다 얼마나 상승했는지 측정 ($MonthlyCharges - AvgCharges$). 약정 종료 등으로 인한 가격 저항도를 수치화.
- **IsFamily (가족 결합도)**: 배우자 및 부양가족 유무를 결합하여 가족 단위 고객 여부 판단. '락인(Lock-in) 효과'를 반영하는 핵심 지표.
- **ServiceCount (서비스 이용도)**: 고객이 가입한 부가 서비스의 총 개수 계산. 서비스 의존도가 높을수록 이탈률이 낮아지는 경향을 포착.

### 3. 모델링 (Modeling)
- **알고리즘**: XGBoost Classifier (최신 v2.0+ 문법 적용)
- **전략**:
  - **5-Fold Stratified Cross Validation**: 타겟 클래스 비율을 유지하며 전체 데이터셋을 5개로 나누어 교차 검증을 수행함으로써 모델의 안정성 확보.
  - **Early Stopping**: 50회 이상 성능 개선이 없을 시 학습을 조기 종료하여 과적합(Overfitting) 방지.
  - **Ensemble Averaging**: 5개 폴드 모델의 예측 확률값을 평균(Soft Voting)하여 최종 예측치를 산출, 리더보드 점수 최적화.

## 📊 결과 및 성능
- **최종 평균 ROC-AUC**: **0.9131**
- **주요 피처 기여도**: `PaymentMethod_Electronic check`, `Contract`, `ChargeDiff` 순으로 이탈 예측에 높은 영향력을 보임.
