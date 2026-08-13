# TranAD-based Gas Sensor Anomaly Detection

## Project Overview

본 프로젝트는 TranAD(Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data)의 공개 구현을 기반으로, 가스 센서 시계열 데이터의 이상 상태를 탐지하기 위해 데이터 전처리, 이상 데이터 생성, 평가 방법 및 시각화 기능을 수정확장한 프로젝트입니다.

실제 환경에서는 충분한 가스 누출 이상 데이터를 확보하기 어렵다는 문제를 고려하여 정상 센서 데이터를 기반으로 학습 데이터를 구성하고, 가스 누출 상황을 모사한 연속형 이상 데이터를 생성하여 모델의 탐지 성능을 평가했습니다.

---

## Key Contributions

### 1. 가스 센서 데이터에 맞춘 전처리 Pipeline 구성

원본 TranAD의 공개 데이터셋 대신 실제 가스 센서 데이터에 적용할 수 있도록 별도의 전처리 과정을 구현했습니다.

주요 사용 Feature는 다음과 같습니다.

```text
Gas(MQ-2)
Gas(MQ-4)
```

여러 CSV 및 Excel 파일에서 센서 데이터를 수집하여 하나의 시계열 데이터셋으로 통합하고, 결측값과 비정상적인 값을 처리했습니다.

또한 각 센서별 평균과 표준편차를 계산하여 **3-Sigma 범위를 벗어나는 데이터를 제거**함으로써 정상 상태 중심의 학습 데이터를 구성했습니다.

---

### 2. 가스 누출 상황을 모사한 이상 데이터 생성

실제 환경에서는 충분한 이상 데이터를 확보하기 어렵기 때문에 Test 데이터에 인위적인 이상 구간을 생성했습니다.

가스 누출과 관련된 상수 및 누출 조건을 기반으로 농도 증가량을 계산하고, Test 데이터의 특정 연속 구간에 시간에 따라 증가하는 값을 추가했습니다.

```text
Normal Gas Signal
        ↓
Random anomaly segment selection
        ↓
Gas leakage concentration added over time
        ↓
Continuous anomaly sequence
        ↓
Label = 1
```

단일 시점에 큰 값을 추가하는 방식이 아니라, **시간이 지남에 따라 가스 농도가 점진적으로 증가하는 형태**로 이상 패턴을 구성했습니다.

이를 통해 실제 가스 누출 상황과 유사한 연속형 이상 구간을 생성하고 모델 성능을 검증했습니다.

---

### 3. TranAD 입력 구조 수정

가스 센서 데이터를 TranAD에 입력하기 위해 Sliding Window 기반 입력 구조를 적용했습니다.

```text
Gas Sensor Time Series
        ↓
Windowing
        ↓
TranAD
        ↓
Reconstruction Error
        ↓
Anomaly Score
```

본 프로젝트에서는 60개의 시점을 하나의 Window로 구성하여 시계열의 시간적 패턴을 학습하도록 설정했습니다.

---

### 4. Dynamic Threshold 기반 이상 탐지

고정된 임계값만으로 이상 여부를 판단할 경우 데이터 분포 변화에 대응하기 어렵기 때문에 Reconstruction Error를 기반으로 **Dynamic Threshold**를 계산하도록 평가 과정을 수정했습니다.

각 시점의 Anomaly Score와 동적으로 계산된 Threshold를 비교하여 이상 여부를 판단합니다.

```text
Anomaly Score > Dynamic Threshold
                ↓
              Anomaly
```

이를 통해 시계열 데이터의 변화에 따라 이상 판단 기준이 조정될 수 있도록 구성했습니다.

---

### 5. 시계열 이상탐지 평가 방법 확장

시계열 이상탐지는 개별 데이터 포인트뿐만 아니라 **이상 구간을 탐지했는지 여부**가 중요하기 때문에 다양한 평가 방식을 추가했습니다.

사용한 평가 방식은 다음과 같습니다.

* Precision
* Recall
* F1-score
* ROC-AUC
* Point Adjustment (PA)
* PA%K
* AF-β Score
* Anomaly Segment Detection Rate

특히 PA와 PA%K를 활용하여 연속적인 이상 구간에 대한 탐지 성능을 추가적으로 평가했습니다.

---

## Dataset

전체 데이터는 가스 센서 기반의 다변량 시계열 데이터로 구성되어 있습니다.

Repository 실험 기준 데이터 구성은 다음과 같습니다.

```text
Total Data : 5,953,348
Train      : 4,762,678
Test       : 1,190,670
Anomaly    : 11,880
```

Train 데이터는 정상 데이터를 중심으로 구성하고, Test 데이터에는 인위적으로 생성된 가스 누출 이상 구간을 포함했습니다.

---

## Model

본 프로젝트에서는 Transformer 기반 시계열 이상탐지 모델인 **TranAD**를 사용했습니다.

TranAD는 정상 데이터를 기반으로 시계열 패턴을 학습하고 입력값과 재구성 결과 사이의 차이를 이용하여 Anomaly Score를 계산합니다.

### Hyperparameters

```text
Model       : TranAD
Window Size : 60
Learning Rate : 0.0001
Input Dimension : 2
Batch Size  : 128
```

---

## Evaluation Pipeline

```text
Gas Sensor Data
        ↓
Preprocessing
        ↓
3-Sigma Outlier Removal
        ↓
Train / Test Split
        ↓
Synthetic Gas Leak Generation
        ↓
Sliding Window
        ↓
TranAD Training
        ↓
Reconstruction Error
        ↓
Dynamic Threshold
        ↓
Anomaly Detection
        ↓
PA / PA%K / F1 / ROC-AUC Evaluation
```

---

## Visualization

모델 평가 결과를 직관적으로 확인할 수 있도록 다음 데이터를 시각화했습니다.

* Original Sensor Signal
* Predicted / Reconstructed Signal
* Anomaly Score
* Dynamic Threshold
* Detected Anomaly Region
* Training Loss
* Learning Rate

Anomaly Score와 Threshold를 함께 출력하여 모델이 어느 시점에서 이상을 판단했는지 확인할 수 있도록 구성했습니다.

---

## How to Run

### Data Preprocessing

```bash
python preprocess3.py
```

가스 센서 데이터를 전처리하고 Train/Test 데이터를 `.npy` 형식으로 저장합니다.

### Train

```bash
python main.py --model TranAD --dataset GAS --retrain
```

### Test

```bash
python main.py --model TranAD --dataset GAS --test
```

### Continue Training

```bash
python main.py --model TranAD --dataset GAS
```

---

## Environment

```text
Python
PyTorch 1.8.1
CUDA 10.2
NumPy
Pandas
Scikit-learn
Matplotlib
```

---

## Repository Structure

```text
.
├── data/
├── plots/
├── results/
├── src/
│   ├── models.py
│   ├── utils.py
│   ├── plotting.py
│   ├── pot.py
│   └── spot.py
│
├── main.py
├── preprocess.py
├── preprocess2.py
├── preprocess3.py
├── requirements.txt
└── README.md
```

---

## What I Modified

본 프로젝트에서는 TranAD의 핵심 모델 구조 자체를 새롭게 제안한 것이 아니라, 공개된 TranAD 구현을 기반으로 실제 가스 센서 이상탐지 문제에 적용하기 위한 Pipeline을 수정, 확장했습니다.

주요 수정 내용은 다음과 같습니다.

* 가스 센서 데이터 전용 전처리 구현
* MQ-2 / MQ-4 센서 데이터 활용
* 3-Sigma 기반 정상 데이터 정제
* 가스 누출 상황을 반영한 연속형 이상 데이터 생성
* TranAD 입력 Window 구성 수정
* 가스 센서 Feature 수에 맞춘 모델 입력 구조 조정
* Dynamic Threshold 기반 이상 판단 로직 적용
* Point Adjustment(PA) 평가 구현
* PA%K 평가 구현
* AF-β Score 평가 추가
* 이상 구간 탐지율 평가 추가
* Anomaly Score / Threshold / Prediction 시각화 기능 추가
* GAS Dataset 학습 및 테스트 Pipeline 구성

---

## Reference

This repository is forked from the original TranAD implementation:

**TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data**

The original TranAD architecture and implementation belong to the original authors.

This repository focuses on modifying and extending the original implementation for a gas sensor anomaly detection experiment.
