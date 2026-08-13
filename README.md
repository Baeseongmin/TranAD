# TranAD-based Time Series Anomaly Detection

## 1. Project Overview

본 프로젝트는 **TranAD(Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data)**를 기반으로, 다변량 센서 데이터의 이상 상태를 탐지하기 위해 기존 모델을 수정·적용한 프로젝트입니다.

정상 상태의 센서 데이터를 기반으로 모델이 정상 패턴을 학습하도록 구성하고, 실제 환경에서 발생할 수 있는 이상 상황을 고려하여 데이터를 구성하였습니다. 또한 기존 TranAD 구조 및 학습·평가 과정을 프로젝트 환경에 맞게 수정하여 이상탐지 성능을 검증했습니다.

### 주요 목표

* 다변량 시계열 센서 데이터의 정상 패턴 학습
* Transformer 기반 재구성 오차를 활용한 이상 탐지
* 프로젝트 데이터 구조에 맞춘 TranAD 모델 및 학습 과정 수정
* 이상 점수(Anomaly Score)를 활용한 이상 구간 탐지
* 실제 시계열 이상탐지 환경에서의 모델 성능 검증

---

## 2. Background

시계열 센서 데이터의 이상탐지는 제조 공정, 설비 모니터링, 안전 관리 등 다양한 산업 환경에서 활용될 수 있습니다.

하지만 실제 환경에서는 충분한 이상 데이터를 확보하기 어려운 경우가 많습니다. 본 프로젝트에서도 **정상 데이터를 중심으로 학습하고 정상 패턴에서 벗어나는 정도를 이용해 이상을 탐지하는 방식**을 적용했습니다.

TranAD는 Transformer 기반의 시계열 이상탐지 모델로, 시계열 데이터의 시간적 관계와 여러 변수 사이의 관계를 학습하고 재구성 결과를 이용해 이상 여부를 판단합니다.

---

## 3. Model

### TranAD

본 프로젝트에서는 다변량 시계열 이상탐지를 위해 TranAD를 기반 모델로 사용했습니다.

입력된 시계열 데이터를 일정한 Window 단위로 구성하고 모델이 정상 데이터의 패턴을 학습하도록 했습니다. 이후 입력 데이터와 모델의 재구성 결과 사이의 차이를 기반으로 **Anomaly Score**를 계산하여 이상 상태를 탐지했습니다.

### Processing Flow

```text
Multivariate Sensor Data
        ↓
Data Preprocessing
        ↓
Sliding Window
        ↓
TranAD
        ↓
Reconstruction
        ↓
Reconstruction Error
        ↓
Anomaly Score
        ↓
Threshold
        ↓
Normal / Anomaly
```

---

## 4. Modifications

기존 TranAD 구현을 그대로 사용하는 것이 아니라 프로젝트의 데이터 및 실험 환경에 맞게 일부 구조와 학습 과정을 수정했습니다.

주요 수정 사항은 다음과 같습니다.

* 프로젝트 센서 데이터 형식에 맞춘 Data Loader 구성
* 다변량 시계열 입력을 위한 Sliding Window 전처리
* 입력 Feature 수에 따른 모델 구조 및 Parameter 조정
* 정상 데이터 중심의 학습 Pipeline 구성
* Reconstruction Error 기반 Anomaly Score 계산
* 프로젝트 환경에 맞춘 Threshold 적용
* 모델 학습 및 평가 과정 수정
* 이상탐지 결과 시각화 기능 추가

> 본 프로젝트의 TranAD 모델은 기존 TranAD 연구 및 공개 구현을 기반으로 하며, 프로젝트 목적과 데이터 환경에 맞게 수정하여 사용했습니다.

---

## 5. Dataset

사용 데이터는 여러 센서로부터 수집된 **다변량 시계열 데이터**로 구성되어 있습니다.

주요 변수의 예시는 다음과 같습니다.

```text
Temperature
Humidity
Gas Sensor 1
Gas Sensor 2
...
```

정상 데이터를 기반으로 정상 상태의 시계열 패턴을 학습하고, 정상 패턴과 차이가 발생하는 구간을 이상 상태로 판단하도록 구성했습니다.

※ 실제 데이터는 보안 및 데이터 사용 정책에 따라 Repository에 포함하지 않을 수 있습니다.

---

## 6. Anomaly Detection

모델의 예측값과 실제 입력값 사이의 Reconstruction Error를 이용하여 Anomaly Score를 계산합니다.

```text
Input Time Series
        ↓
TranAD Reconstruction
        ↓
Actual - Reconstruction
        ↓
Reconstruction Error
        ↓
Anomaly Score
```

Anomaly Score가 설정된 Threshold를 초과하는 경우 해당 시점을 이상 상태로 판단합니다.

Threshold는 실험 환경에 따라 고정 임계값뿐만 아니라 통계적 방법을 적용할 수 있도록 구성했습니다.

---

## 7. Evaluation

이상탐지 성능 평가에는 Precision, Recall, F1-score 등을 활용했습니다.

특히 시계열 이상탐지에서는 단일 시점의 정확도뿐만 아니라 **이상 구간을 실제로 탐지했는지**가 중요하기 때문에 이상 구간 단위의 성능도 함께 고려했습니다.

| Metric    | Description               |
| --------- | ------------------------- |
| Precision | 이상으로 탐지한 데이터 중 실제 이상 비율   |
| Recall    | 실제 이상 중 모델이 탐지한 비율        |
| F1-score  | Precision과 Recall의 조화평균   |
| PA-F1     | 이상 구간 탐지 특성을 고려한 F1-score |

---

## 8. Tech Stack

```text
Python
PyTorch
NumPy
Pandas
Scikit-learn
Matplotlib
```

---

## 9. Repository Structure

```text
.
├── data/
├── models/
│   └── tranad.py
├── utils/
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

※ 실제 Repository 구조에 따라 수정해주세요.

---

## 10. Results

TranAD를 프로젝트 데이터에 적용하여 정상 시계열 패턴과 이상 상태를 구분하는 실험을 수행했습니다.

모델의 Reconstruction Error와 Threshold를 이용하여 이상 구간을 탐지했으며, 실험 과정에서 Threshold 설정 방법과 시계열 Window 구성에 따라 탐지 성능이 달라지는 것을 확인했습니다.

최종적으로 시계열 데이터의 특성을 고려한 전처리, 모델 학습, Threshold 설정 및 평가 과정을 하나의 이상탐지 Pipeline으로 구현했습니다.

---

## 11. Reference

This project is based on **TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data**.

The original TranAD architecture and research belong to their respective authors. This repository contains modifications and implementations made for experimental and educational purposes.

## 12. What I Learned

본 프로젝트를 통해 단순히 공개된 모델을 실행하는 것에서 그치지 않고, 실제 데이터 환경에 맞게 모델의 입력 구조와 학습 Pipeline을 수정하고 이상탐지 기준을 설계하는 경험을 할 수 있었습니다.

특히 모델 자체의 성능뿐만 아니라 **데이터 전처리, Window 구성, Reconstruction Error 계산, Threshold 설정**이 실제 이상탐지 성능에 큰 영향을 미친다는 점을 확인했습니다.
