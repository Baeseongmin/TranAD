import numpy as np

from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *


def create_anomaly_segments(data_length, segment_size=60):
    """
    데이터 길이를 기준으로 일정한 크기의 anomaly segments를 생성합니다.

    Parameters:
    - data_length: 전체 데이터의 길이 (예: len(anomaly_scores))
    - segment_size: 각 anomaly segment의 길이 (기본값: 60)

    Returns:
    - anomaly_segments: 각 segment의 인덱스를 담은 리스트
    """
    anomaly_segments = []
    for start in range(0, data_length, segment_size):
        end = min(start + segment_size, data_length)
        anomaly_segments.append(slice(start, end))  # 각 구간을 슬라이스로 저장


    return anomaly_segments



def calculate_pa_k(anomaly_scores, dynamic_thresholds, k_percent):
    """
    PA%k 지표를 계산합니다.

    Parameters:
    - predictions: 모델이 예측한 정상/이상치 레이블 (1: 이상치, 0: 정상)
    - anomaly_scores: 각 시점에 대한 이상치 점수
    - dynamic_thresholds: 각 컬럼에 대한 동적 임계값
    - k_percent: 구간 내 이상치 점수가 임계값을 초과하는 비율의 기준 (예: 0.1 -> 10%)

    Returns:
    - pa_k_predictions: PA%k 기반으로 재계산된 예측 레이블
    """
    anomaly_scores = smooth_losses(anomaly_scores)

    # 각 컬럼별로 결과를 처리할 배열 생성
    num_columns = anomaly_scores.shape[1]
    pa_k_predictions_per_column = np.zeros_like(anomaly_scores, dtype=int)

    # anomaly_segments 생성
    data_length = anomaly_scores.shape[0]
    anomaly_segments = create_anomaly_segments(data_length)

    # 각 컬럼별로 구간별 이상치 판단
    for col in range(num_columns):
        for segment in anomaly_segments:
            # anomaly_scores에서 해당 segment의 점수들 추출 (특정 컬럼에 대해서만)
            segment_scores = anomaly_scores[segment, col]

            # 해당 컬럼의 dynamic_thresholds를 segment와 같은 형상으로 가져오기
            segment_thresholds = dynamic_thresholds[segment, col]

            # threshold를 초과한 시점 수 계산
            above_delta_count = np.sum(segment_scores > segment_thresholds)
            total_count = len(segment_scores)

            # 임계값을 초과하는 비율이 k_percent 이상이면 해당 구간 전체를 이상치로 간주
            if total_count > 0 and (above_delta_count / total_count >= k_percent):
                pa_k_predictions_per_column[segment, col] = 1  # 해당 구간을 이상치로 설정

    # 각 행에서 하나라도 이상치인 경우 해당 시점을 이상치로 설정
    pa_k_predictions = np.any(pa_k_predictions_per_column, axis=1).astype(int)

    print(f"PA%k predictions labels Number of anomalies detected: {np.sum(pa_k_predictions)}/{data_length}")


    return pa_k_predictions


def calculate_pa(anomaly_scores, dynamic_thresholds):
    """
    PA 지표를 계산합니다 (구간 내 하나라도 이상치로 판단되면 해당 구간 전체를 이상치로 처리).

    Parameters:
    - predictions: 모델이 예측한 정상/이상치 레이블 (1: 이상치, 0: 정상)
    - anomaly_scores: 각 시점에 대한 이상치 점수
    - dynamic_thresholds: 각 컬럼에 대한 동적 임계값
    - k_percent: (사용되지 않음, 기존 코드와의 호환성을 위해 유지)

    Returns:
    - pa_k_predictions: PA 기반으로 재계산된 예측 레이블
    """
    anomaly_scores = smooth_losses(anomaly_scores)

    # 각 컬럼별로 결과를 처리할 배열 생성
    num_columns = anomaly_scores.shape[1]
    pa_k_predictions_per_column = np.zeros_like(anomaly_scores, dtype=int)

    # anomaly_segments 생성
    data_length = anomaly_scores.shape[0]
    anomaly_segments = create_anomaly_segments(data_length)

    # 각 컬럼별로 구간별 이상치 판단
    for col in range(num_columns):
        for segment in anomaly_segments:
            # anomaly_scores에서 해당 segment의 점수들 추출 (특정 컬럼에 대해서만)
            segment_scores = anomaly_scores[segment, col]

            # 해당 컬럼의 dynamic_thresholds를 segment와 같은 형상으로 가져오기
            segment_thresholds = dynamic_thresholds[segment, col]

            # threshold를 초과한 시점 수 계산
            above_delta_count = np.sum(segment_scores > segment_thresholds)

            # 구간 내 하나라도 임계값 초과 시, 해당 구간 전체를 이상치로 간주
            if above_delta_count > 0:  # 1개 이상 초과하면 전체를 이상치로 처리
                pa_k_predictions_per_column[segment, col] = 1  # 해당 구간을 이상치로 설정

    # 각 행에서 하나라도 이상치인 경우 해당 시점을 이상치로 설정
    pa_predictions = np.any(pa_k_predictions_per_column, axis=1).astype(int)

    print(f"PA predictions labels Number of anomalies detected: {np.sum(pa_predictions)}/{data_length}")

    return pa_predictions



def calculate_af_beta(true_anomalies, predicted_anomalies, tolerance_window=5, beta=1):
    """
    Af β (Affiliation-based F-score)를 계산합니다.

    Parameters:
    - true_anomalies: 실제 이상치 레이블 (1: 이상, 0: 정상)
    - predicted_anomalies: 예측된 이상치 레이블 (1: 이상, 0: 정상)
    - tolerance_window: 근접성을 판단하기 위한 허용 창 (int)
    - beta: Precision과 Recall의 가중치를 조정하는 파라미터 (float)

    Returns:
    - af_beta_score: 계산된 Af β 점수
    """

    # 실제 이상치와 예측된 이상치의 인덱스를 추출
    true_indices = np.where(true_anomalies == 1)[0]
    predicted_indices = np.where(predicted_anomalies == 1)[0]

    # True Positive, False Positive, False Negative 초기화
    tp = 0
    matched_predicted = set()  # 이미 매칭된 예측 인덱스를 추적
    for true_idx in true_indices:
        # 허용 창 내에서 예측된 이상치를 확인
        is_match = False
        for pred_idx in predicted_indices:
            if pred_idx in matched_predicted:
                continue  # 이미 매칭된 예측은 무시
            if abs(true_idx - pred_idx) <= tolerance_window:
                tp += 1
                matched_predicted.add(pred_idx)
                is_match = True
                break

    # False Positive (매칭되지 않은 예측)
    fp = len(predicted_indices) - len(matched_predicted)
    # False Negative (매칭되지 않은 실제 이상)
    fn = len(true_indices) - tp

    # Precision, Recall 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Af β 계산
    beta_squared = beta ** 2
    if precision + recall == 0:
        af_beta_score = 0
    else:
        af_beta_score = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)

    return af_beta_score



# 지수 가중 이동 평균 (EWMA)를 사용하여 스무딩하는 함수
def smooth_losses(test_losses, alpha=0.3):
    """
    지수 가중 이동 평균 (EWMA)를 사용하여 잔차를 스무딩.
    """
    smoothed_test_losses = np.zeros_like(test_losses)
    smoothed_test_losses[0] = test_losses[0]  # 초기값 설정
    for t in range(1, len(test_losses)):
        smoothed_test_losses[t] = alpha * test_losses[t] + (1 - alpha) * smoothed_test_losses[t - 1]
    return smoothed_test_losses

# Gas 데이터 개별로 계산 
# 실시간 동적 임계값 계산 함수 (개별 컬럼별로 처리)
# z 값 : 2~10 사이의 값으로 실험적으로 찾아내는 값
def calculate_dynamic_threshold_per_column(test_losses, z=7, window_size=2500):
    num_columns = test_losses.shape[1]
    dynamic_thresholds = np.zeros(test_losses.shape)

    for col in range(num_columns):
        for t in range(len(test_losses)):
            # 최근 window_size개의 손실만 포함
            start_idx = max(0, t - window_size + 1)
            current_window = test_losses[start_idx:t + 1, col]

            # 현재 윈도우의 평균과 표준 편차 계산
            current_mean = np.mean(current_window).item()
            current_std = np.std(current_window).item()

            # 동적 임계값 계산
            dynamic_thresholds[t, col] = current_mean + z * current_std

    return dynamic_thresholds

# # 실시간 이상 탐지 함수 (개별 컬럼별로 처리)
# def evaluate_anomalies_online(test_losses, z=7):
#     smoothed_losses = smooth_losses(test_losses)

#     # 각 컬럼별로 동적 임계값 계산
#     dynamic_thresholds = calculate_dynamic_threshold_per_column(smoothed_losses, z=z)

#     # 각 컬럼별로 이상치 여부 판단
#     anomalies = smoothed_losses > dynamic_thresholds
#     # 각 행에서 하나의 컬럼이라도 이상치인 경우 이상치로 판단
#     anomalies_per_row = np.any(anomalies, axis=1)
#     num_anomalies = np.sum(anomalies_per_row)  # 전체 행에서 하나라도 True인 경우의 수

#     total_per_column = smoothed_losses.shape[0]

#     print(f"Number of anomalies detected: {num_anomalies}/{total_per_column}")

#     return anomalies, dynamic_thresholds, smoothed_losses
# 실시간 이상 탐지 및 레이블 생성 함수 (개별 컬럼별로 처리)

def evaluate_anomalies_normal(test_losses, z=7):
    smoothed_losses = smooth_losses(test_losses)

    # 각 컬럼별로 동적 임계값 계산
    dynamic_thresholds = calculate_dynamic_threshold_per_column(smoothed_losses, z=z)

    # 각 컬럼별로 이상치 여부 판단
    anomalies = smoothed_losses > dynamic_thresholds
    # 각 행에서 하나의 컬럼이라도 이상치인 경우 이상치로 판단
    anomalies_per_row = np.any(anomalies, axis=1)
    num_anomalies = np.sum(anomalies_per_row)  # 전체 행에서 하나라도 True인 경우의 수

    total_per_column = smoothed_losses.shape[0]
    print(f"Normal labels Number of anomalies detected: {num_anomalies}/{total_per_column}")

    # 레이블 생성
    labels = np.where(anomalies_per_row, 1, 0)  # 이상치가 있는 행은 1, 그렇지 않으면 0

    return labels, anomalies, dynamic_thresholds, smoothed_losses


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    # 부동소수점 연산을 위해 float 변환 및 소수점 3째 자리까지 반올림
    TP = round(float(np.sum(predict * actual)), 3)
    TN = round(float(np.sum((1 - predict) * (1 - actual))), 3)
    FP = round(float(np.sum(predict * (1 - actual))), 3)
    FN = round(float(np.sum((1 - predict) * actual)), 3)
    precision = round(TP / (TP + FP + 0.00001), 3)
    recall = round(TP / (TP + FN + 0.00001), 3)
    f1 = round(2 * precision * recall / (precision + recall + 0.00001), 3)
    
    try:
        roc_auc = round(float(roc_auc_score(actual, predict)), 3)
    except:
        roc_auc = 0.0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc





def generate_true_anomaly_segments(true_labels):
    """
    실제 이상치 레이블 배열에서 이상치 구간을 생성.

    Parameters:
    - true_labels: 실제 이상치 레이블 배열 (1: 이상치, 0: 정상)

    Returns:
    - true_anomaly_segments: 이상치 구간 리스트 (slice 객체로 반환)
    """
    true_anomaly_segments = []
    in_anomaly = False
    start = 0

    for i, label in enumerate(true_labels):
        if label == 1 and not in_anomaly:
            # 새로운 이상치 구간 시작
            start = i
            in_anomaly = True
        elif label == 0 and in_anomaly:
            # 이상치 구간 종료
            true_anomaly_segments.append(slice(start, i))
            in_anomaly = False

    # 끝까지 이상치 구간이 유지된 경우 처리
    if in_anomaly:
        true_anomaly_segments.append(slice(start, len(true_labels)))

    return true_anomaly_segments


def evaluate_anomaly_segments(actual, predicted_anomalies, threshold):
    """
    이상치 구간 평가 함수.
    
    Parameters:
    - true_anomaly_segments: 실제 이상치 구간 리스트 (슬라이스 또는 인덱스 리스트)
    - predicted_anomalies: 모델이 예측한 이상치 레이블 (1: 이상, 0: 정상)
    - threshold: 구간 내 데이터의 이상 탐지 비율 기준 (기본값: 0.5)
    
    Returns:
    - detected_segments: 탐지된 이상치 구간 수
    - total_segments: 전체 이상치 구간 수
    - accuracy: 탐지 정확도 (탐지된 구간 / 전체 구간)
    """

    true_anomaly_segments = generate_true_anomaly_segments(actual)
    detected_segments = 0
    total_segments = len(true_anomaly_segments)

    for segment in true_anomaly_segments:
        # 구간 내 예측된 이상치 데이터 개수 계산
        predicted_anomaly_count = np.sum(predicted_anomalies[segment])
        total_count = len(range(*segment.indices(len(predicted_anomalies))))  # 구간 내 전체 데이터 수
        
        # 이상 탐지 비율이 threshold 이상이면 해당 구간을 탐지된 것으로 간주
        if total_count > 0 and (predicted_anomaly_count / total_count) >= threshold:
            detected_segments += 1

    # 탐지 정확도 계산
    accuracy = detected_segments / total_segments if total_segments > 0 else 0


    return detected_segments, total_segments, accuracy


# Normal labels generate
def generate_labels(dynamic_threshold, anomaly_score):

    anomaly_score = smooth_losses(anomaly_score)

    anomaly_score = np.mean(anomaly_score, axis=1)
    if len(dynamic_threshold) != len(anomaly_score):
        raise ValueError("dynamic_threshold and anomaly_score must have the same length")
    
    # 비교하여 레이블 생성
    labels = [1 if (threshold < score).all() else 0 for threshold, score in zip(dynamic_threshold, anomaly_score)]

    return labels

# # the below function is taken from OmniAnomaly code base directly
# def adjust_predicts(score, label,
#                     threshold=None,
#                     pred=None,
#                     calc_latency=False):
 
#     if len(score) != len(label):
#         raise ValueError("score and label must have the same length")
#     score = np.asarray(score)
#     label = np.asarray(label)
#     latency = 0
#     if pred is None:
#         predict = score > threshold
#     else:
#         predict = pred
#     actual = label > 0.1
#     anomaly_state = False
#     anomaly_count = 0
#     for i in range(len(score)):
#         if actual[i] and predict[i] and not anomaly_state:
#                 anomaly_state = True
#                 anomaly_count += 1
#                 for j in range(i, 0, -1):
#                     if not actual[j]:
#                         break
#                     else:
#                         if not predict[j]:
#                             predict[j] = True
#                             latency += 1
#         elif not actual[i]:
#             anomaly_state = False
#         if anomaly_state:
#             predict[i] = True
#     if calc_latency:
#         return predict, latency / (anomaly_count + 1e-4)
#     else:
#         return predict


# def calc_seq(score, label, threshold, calc_latency=False):
#     """
#     Calculate f1 score for a score sequence
#     """
#     if calc_latency:
#         predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
#         t = list(calc_point2point(predict, label))
#         t.append(latency)
#         return t
#     else:
#         predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
#         return calc_point2point(predict, label)


# def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
#     """
#     Find the best-f1 score by searching best `threshold` in [`start`, `end`).
#     Returns:
#         list: list for results
#         float: the `threshold` for best-f1
#     """
#     if step_num is None or end is None:
#         end = start
#         step_num = 1
#     search_step, search_range, search_lower_bound = step_num, end - start, start
#     if verbose:
#         print("search range: ", search_lower_bound, search_lower_bound + search_range)
#     threshold = search_lower_bound
#     m = (-1., -1., -1.)
#     m_t = 0.0
#     for i in range(search_step):
#         threshold += search_range / float(search_step)
#         target = calc_seq(score, label, threshold, calc_latency=True)
#         if target[0] > m[0]:
#             m_t = threshold
#             m = target
#         if verbose and i % display_freq == 0:
#             print("cur thr: ", threshold, target, m, m_t)
#     print(m, m_t)
#     return m, m_t


# def pot_eval(init_score, score, label, q=1e-5, level=0.02):
#     """
#     Run POT method on given score.
#     Args:
#         init_score (np.ndarray): The data to get init threshold.
#             it should be the anomaly score of train set.
#         score (np.ndarray): The data to run POT method.
#             it should be the anomaly score of test set.
#         label:
#         q (float): Detection level (risk)
#         level (float): Probability associated with the initial threshold t
#     Returns:
#         dict: pot result dict
#     """
#     lms = lm[0]
#     while True:
#         try:
#             s = SPOT(q)  # SPOT object
#             s.fit(init_score, score)  # data import
#             s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
#         except: lms = lms * 0.999
#         else: break
#     ret = s.run(dynamic=False)  # run
#     # print(len(ret['alarms']))
#     # print(len(ret['thresholds']))
#     pot_th = np.mean(ret['thresholds']) * lm[1]
#     # pot_th = np.percentile(score, 100 * lm[0])
#     # np.percentile(score, 100 * lm[0])
#     pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
#     # DEBUG - np.save(f'{debug}.npy', np.array(pred))
#     # DEBUG - print(np.argwhere(np.array(pred)))
#     p_t = calc_point2point(pred, label)
#     # print('POT result: ', p_t, pot_th, p_latency)
#     return {
#         'f1': p_t[0],
#         'precision': p_t[1],
#         'recall': p_t[2],
#         'TP': p_t[3],
#         'TN': p_t[4],
#         'FP': p_t[5],
#         'FN': p_t[6],
#         'ROC/AUC': p_t[7],
#         'threshold': pot_th,
#         # 'pot-latency': p_latency
#     }, np.array(pred)


# def pot_calculate_thresholds_per_column(init_scores, scores, q=1e-5, level=0.02, window_size=2500):
#     """
#     Calculate dynamic thresholds per column using the POT method.

#     Parameters:
#         init_scores (np.ndarray): Training set anomaly scores (2D array, columns for each gas).
#         scores (np.ndarray): Test set anomaly scores (2D array, columns for each gas).
#         q (float): Detection level (risk).
#         level (float): Probability associated with the initial threshold.
#         window_size (int): Size of each window for recalculating thresholds.

#     Returns:
#         np.ndarray: Dynamic thresholds for each column in the test set, calculated per window.
#     """
#     import numpy as np

#     # Ensure inputs are numpy arrays
#     init_scores = np.array(init_scores)
#     scores = np.array(scores)

#     # Initialize output array
#     dynamic_thresholds = np.zeros(scores.shape)

#     # Iterate over each column
#     num_columns = scores.shape[1]
#     for col in range(num_columns):
#         # Training scores for current column
#         init_score = init_scores[:, col] if init_scores.ndim > 1 else init_scores

#         for t in range(len(scores)):
#             # Define the sliding window
#             start_idx = max(0, t - window_size + 1)
#             current_window = scores[start_idx:t + 1, col]

#             # Mean and standard deviation for the current window
#             current_mean = np.mean(current_window).item()
#             current_std = np.std(current_window).item()

#             # SPOT threshold adjustment
#             s = SPOT(q)
#             s.fit(init_score, current_window)
#             s.initialize(level=level, min_extrema=False, verbose=False)
#             results = s.run(dynamic=False)
#             threshold = results['thresholds'][-1] if results['thresholds'] else current_mean + current_std

#             # Store the dynamic threshold
#             dynamic_thresholds[t, col] = threshold

#     return dynamic_thresholds