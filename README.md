# TranAD-based Gas Sensor Anomaly Detection

## Project Overview

본 프로젝트는 TranAD(Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data)의 공개 구현을 기반으로, 가스 센서 시계열 데이터의 이상 상태를 탐지하기 위해 데이터 전처리, 이상 데이터 생성, 평가 방법 및 시각화 기능을 수정·확장한 프로젝트입니다.

실제 환경에서는 충분한 가스 누출 이상 데이터를 확보하기 어렵다는 문제를 고려하여 정상 센서 데이터를 기반으로 학습 데이터를 구성하고, 가스 누출 상황을 모사한 연속형 이상 데이터를 생성하여 모델의 탐지 성능을 평가했습니다.

### Key Results

* Best F1-score: **0.917** (Window Size 600, PA%K)
* Best Precision: **0.983** (Window Size 600, PA%K)
* Best Segment Detection Rate: **90.77%** (Window Size 60, PA)
* Maximum Detected Anomaly Segments: **59 / 65**

---

## Key Contributions

### 1. 가스 센서 데이터 전처리 Pipeline 구성

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

가스 누출과 관련된 조건을 기반으로 농도 증가량을 계산하고, Test 데이터의 특정 연속 구간에 시간에 따라 증가하는 값을 추가했습니다.

```text
Normal Gas Signal
        ↓
Anomaly Segment Selection
        ↓
Gas Leakage Simulation
        ↓
Gas Concentration Increase
        ↓
Continuous Anomaly Sequence
        ↓
Anomaly Label
```

단일 시점에 임의의 큰 값을 추가하는 방식이 아니라 **시간이 지남에 따라 가스 농도가 변화하는 연속적인 이상 패턴**을 구성하여 실제 가스 누출 상황을 모사했습니다.

---

### 3. TranAD 입력 구조 구성

가스 센서 데이터를 TranAD에 입력하기 위해 Sliding Window 기반 입력 구조를 적용했습니다.

```text
Gas Sensor Time Series
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
```

Window Size에 따른 이상탐지 성능 차이를 확인하기 위해 **60, 300, 600**의 Window Size를 비교했습니다.

---

### 4. Dynamic Threshold 기반 이상 탐지

고정된 임계값만으로 이상 여부를 판단할 경우 시계열 데이터의 변화에 대응하기 어렵기 때문에 Reconstruction Error를 기반으로 **Dynamic Threshold**를 적용했습니다.

각 시점의 Anomaly Score와 동적으로 계산된 Threshold를 비교하여 이상 여부를 판단하도록 구성했습니다.

```text
Anomaly Score > Dynamic Threshold
                ↓
              Anomaly
```

---

### 5. 시계열 이상탐지 평가 방법 확장

개별 시점의 탐지 성능뿐만 아니라 **연속적인 이상 구간을 실제로 탐지했는지** 평가하기 위해 다양한 평가 방식을 적용했습니다.

사용한 평가 지표 및 방식은 다음과 같습니다.

* Precision
* Recall
* F1-score
* ROC-AUC
* Point Adjustment (PA)
* PA%K
* AF-β Score
* Anomaly Segment Detection Rate

이를 통해 단순 Point 단위 성능뿐만 아니라 실제 이상 구간 탐지 성능까지 함께 비교했습니다.

---

## Dataset

사용 데이터는 가스 센서 기반의 다변량 시계열 데이터로 구성되어 있습니다.

```text
Total Data : 5,953,348
Train      : 4,762,678
Test       : 1,190,670
Anomaly    : 11,880
```

Train 데이터는 정상 데이터를 중심으로 구성하고, Test 데이터에는 가스 누출 상황을 모사하여 생성한 이상 구간을 포함했습니다.

---

## Model

본 프로젝트에서는 Transformer 기반 시계열 이상탐지 모델인 **TranAD**를 사용했습니다.

TranAD는 정상 시계열 데이터의 패턴을 학습하고 입력값과 모델의 재구성 결과 사이의 차이를 이용하여 Anomaly Score를 계산합니다.

전체 Pipeline은 다음과 같습니다.

```text
Gas Sensor Data
        ↓
Preprocessing
        ↓
Synthetic Anomaly Generation
        ↓
Sliding Window
        ↓
TranAD
        ↓
Reconstruction Error
        ↓
Anomaly Score
        ↓
Dynamic Threshold
        ↓
Anomaly Detection
        ↓
PA / PA%K / Segment Evaluation
```

---

## Experimental Setup

```text
Model           : TranAD
Window Size     : 60 / 300 / 600
Input Dimension : 2
Learning Rate   : 0.0001
Batch Size      : 128
```

---

## Experimental Results

### Window Size = 60

| Detection Method | Detected Anomalies | F1-score | Precision |    Recall | Detected Segments | Segment Detection Rate |
| ---------------- | -----------------: | -------: | --------: | --------: | ----------------: | ---------------------: |
| Normal           | 11,632 / 1,190,670 |    0.862 |     0.871 |     0.853 |           57 / 65 |                 87.69% |
| PA               | 15,120 / 1,190,670 |    0.782 |     0.698 | **0.889** |       **59 / 65** |             **90.77%** |
| PA%K             | 12,420 / 1,190,670 |    0.860 |     0.841 |     0.879 |           57 / 65 |                 87.69% |

**AF-β Score: 0.8761**

---

### Window Size = 300

| Detection Method | Detected Anomalies |  F1-score | Precision | Recall | Detected Segments | Segment Detection Rate |
| ---------------- | -----------------: | --------: | --------: | -----: | ----------------: | ---------------------: |
| Normal           | 11,090 / 1,190,670 |     0.861 |     0.892 |  0.833 |           56 / 65 |                 86.15% |
| PA               | 15,000 / 1,190,670 |     0.764 |     0.684 |  0.864 |           57 / 65 |                 87.69% |
| PA%K             | 10,500 / 1,190,670 | **0.907** | **0.966** |  0.854 |           56 / 65 |                 86.15% |

**AF-β Score: 0.8751**

---

### Window Size = 600

| Detection Method | Detected Anomalies |  F1-score | Precision | Recall | Detected Segments | Segment Detection Rate |
| ---------------- | -----------------: | --------: | --------: | -----: | ----------------: | ---------------------: |
| Normal           | 10,953 / 1,190,670 |     0.869 |     0.905 |  0.835 |           56 / 65 |                 86.15% |
| PA               | 15,660 / 1,190,670 |     0.750 |     0.659 |  0.869 |           57 / 65 |                 87.69% |
| PA%K             | 10,380 / 1,190,670 | **0.917** | **0.983** |  0.859 |           56 / 65 |                 86.15% |

**AF-β Score: 0.8779**

---

## Result Analysis

가장 높은 F1-score는 **Window Size 600에서 PA%K를 적용했을 때 0.917**로 나타났으며, Precision 또한 **0.983**으로 가장 높은 성능을 기록했습니다.

Window Size 300에서도 PA%K 적용 시 F1-score 0.907을 기록하여 Window Size 증가에 따라 PA%K 기준의 F1-score가 향상되는 결과를 확인했습니다.

반면 PA 방식은 상대적으로 높은 Recall과 Segment Detection Rate를 보였습니다. 특히 Window Size 60에서는 **65개의 실제 이상 구간 중 59개를 탐지하여 90.77%의 가장 높은 Segment Detection Rate**를 기록했습니다.

```text
Best Point-level Performance

Window Size       : 600
Detection Method  : PA%K
F1-score          : 0.917
Precision         : 0.983
Recall            : 0.859
```

```text
Best Segment-level Performance

Window Size       : 60
Detection Method  : PA
Detected Segments : 59 / 65
Detection Rate    : 90.77%
Recall            : 0.889
```

이를 통해 하나의 F1-score만을 기준으로 모델을 평가하기보다 **Precision, Recall과 실제 이상 구간 탐지 성능을 함께 고려해야 한다는 점을 확인했습니다.**

---

## Tech Stack

```text
Python
PyTorch
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
├── main.py
├── preprocess.py
├── preprocess2.py
├── preprocess3.py
├── requirements.txt
└── README.md
```

---

## How to Run

### Data Preprocessing

```bash
python preprocess3.py
```

### Train

```bash
python main.py --model TranAD --dataset GAS --retrain
```

### Test

```bash
python main.py --model TranAD --dataset GAS --test
```

---

## What I Modified

본 프로젝트에서는 TranAD의 핵심 모델 구조 자체를 새롭게 제안한 것이 아니라, 공개된 TranAD 구현을 기반으로 **가스 센서 이상탐지 문제에 적용하기 위한 Pipeline을 수정·확장했습니다.**

주요 수정 및 구현 내용은 다음과 같습니다.

* 가스 센서 데이터 전용 전처리 구현
* MQ-2 / MQ-4 센서 데이터 활용
* 3-Sigma 기반 정상 데이터 정제
* 가스 누출 상황을 반영한 연속형 이상 데이터 생성
* 가스 센서 Feature에 맞춘 TranAD 입력 구성
* Sliding Window 및 Window Size별 비교 실험
* Dynamic Threshold 기반 이상 판단
* Point Adjustment(PA) 평가 적용
* PA%K 평가 적용
* AF-β Score 평가
* 이상 구간 탐지율 평가
* Anomaly Score / Threshold / Prediction 시각화
* GAS Dataset 학습 및 평가 Pipeline 구성

---

## Reference

This project is based on:

**TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data**

The original TranAD architecture and implementation belong to the original authors.

This repository modifies and extends the original TranAD implementation for gas sensor time-series anomaly detection, including data preprocessing, synthetic gas leakage anomaly generation, dynamic thresholding, and additional time-series anomaly detection evaluation methods.
