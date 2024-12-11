import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os, torch
import numpy as np

plt.style.use('default')
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)


def smooth(y, box_pts=30):
    # 3차원 배열일 경우 평균을 구해 1차원으로 축소
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()  # torch.Tensor를 numpy 배열로 변환
    box = np.ones(box_pts) / box_pts  # 스무딩 필터 생성
    
    if y.ndim == 3:  # 3차원 배열일 경우
        y = np.mean(y, axis=(1, 2))  # 각 차원의 평균을 구해 1차원으로 변환
    elif y.ndim == 2:  # 2차원 배열일 경우
        y = np.mean(y, axis=1)  # 각 행의 평균을 구해 1차원으로 변환
    
    y_smooth = np.convolve(y, box, mode='same')  # 스무딩 적용
    
    return y_smooth

# def plotter(name, y_true, y_pred, ascore, dynamic_thresholds=None, anomalies=None):
#     # TranAD 모델의 경우, y_true를 한 칸 롤링
#     if 'TranAD' in name:
#         y_true = torch.roll(y_true, 1, 0)
    
#     # PDF 파일을 저장할 디렉터리 경로
#     os.makedirs(os.path.join('plots', name), exist_ok=True)
#     pdf = PdfPages(f'plots/{name}/dual_dim_output.pdf')

#     # 모든 피처를 하나의 플롯으로 시각화
#     y_t = smooth(y_true)
#     y_p = smooth(y_pred)
#     a_s = smooth(ascore)

#     # Y축 범위를 True 데이터의 최소/최대 값으로 설정
#     y_min, y_max = np.min(y_t), np.max(y_t)

#     # 첫 번째 플롯: True 값과 예측 값 시각화
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     ax1.set_title(f'{name} - True vs Predicted Visualization')
    
#     # True 값, 예측 값 플롯
#     ax1.plot(y_t, label='True Value', color='blue', alpha=0.6, linewidth=0.6)
#     ax1.plot(y_p, label='Predicted Value', color='orange', linestyle='-', linewidth=0.6)
    
#     # Y축 범위를 최소, 최대 값으로 설정
#     ax1.set_ylim(0.55,y_max)
#     ax1.set_ylabel('Values')
#     ax1.legend(loc='upper left')
    
#     pdf.savefig(fig)
#     plt.close()

#     # 두 번째 플롯: Anomaly Score 및 Dynamic Threshold 시각화
#     fig, ax2 = plt.subplots(figsize=(10, 6))
#     ax2.set_title(f'{name} - Anomaly Score and Dynamic Threshold Visualization')
    
#     # 이상치 점수 시각화
#     ax2.plot(a_s, label='Anomaly Score', color='green', alpha=0.5, linewidth=0.4)
#     ax2.set_ylabel('Anomaly Score')
    
#     # 동적 임계값 플롯
#     if dynamic_thresholds is not None:
#         ax2.plot(dynamic_thresholds, 'r--', linewidth=0.5, label='Dynamic Threshold')
    
#     # 이상치 구간에 색을 칠함
#     if anomalies is not None:
#         for idx, is_anomaly in enumerate(anomalies):
#             if is_anomaly:
#                 ax2.axvspan(idx - 0.5, idx + 0.5, color='red', alpha=0.1)

#     ax2.legend(loc='upper right')
    
#     pdf.savefig(fig)
#     plt.close()
#     pdf.close()

# 각 컬럼별 시각화
def plotter(name, y_true, y_pred, ascore, dynamic_thresholds=None, anomalies=None):
    # TranAD 모델의 경우, y_true를 한 칸 롤링
    if 'TranAD' in name:
        y_true = torch.roll(y_true, 1, 0)

    # PDF 파일을 저장할 디렉터리 경로
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/detailed_output.pdf')

    num_columns = ascore.shape[1]  # 컬럼 개수

    # 각 컬럼별로 개별 플롯 생성
    for col in range(num_columns):
        # Y축 범위를 True 데이터의 최소/최대 값으로 설정
        col_anomalies = anomalies[:, col] if anomalies is not None else None
        col_dynamic_threshold = dynamic_thresholds[:, col] if dynamic_thresholds is not None else None

        # 첫 번째 플롯: True 값과 예측 값 시각화 (평균 스무딩 적용)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_title(f'{name} - Column {col + 1} True vs Predicted')
        ax1.set_ylim(0,2000)
        y_t_col = smooth(y_true[:, col]) if y_true.ndim > 1 else smooth(y_true)
        y_p_col = smooth(y_pred[:, col]) if y_pred.ndim > 1 else smooth(y_pred)

        ax1.plot(y_t_col, label='True Value', color='blue', alpha=0.6, linewidth=0.6)
        ax1.plot(y_p_col, label='Predicted Value', color='orange', linestyle='-', linewidth=0.6)
        
        ax1.set_ylabel('Values')
        ax1.legend(loc='upper left')

        pdf.savefig(fig)
        plt.close()

        # 두 번째 플롯: Anomaly Score 및 Dynamic Threshold 시각화
        fig, ax2 = plt.subplots(figsize=(10, 6))
        ax2.set_title(f'{name} - Column {col + 1} Anomaly Score and Dynamic Threshold')
        ax2.set_ylim(0,100)

        # 이상치 점수 시각화
        a_s_col = ascore[:, col] if ascore.ndim > 1 else ascore
        ax2.plot(a_s_col, label='Anomaly Score', color='green', alpha=0.5, linewidth=0.4)
        ax2.set_ylabel('Anomaly Score')

        # 동적 임계값 플롯
        if col_dynamic_threshold is not None:
            ax2.plot(col_dynamic_threshold, 'r--', linewidth=0.5, label='Dynamic Threshold')

        # 이상치 구간에 색을 칠함
        if col_anomalies is not None:
            for idx, is_anomaly in enumerate(col_anomalies):
                if is_anomaly:
                    ax2.axvspan(idx - 0.5, idx + 0.5, color='red', alpha=0.1)

        ax2.legend(loc='upper right')

        pdf.savefig(fig)
        plt.close()

    pdf.close()
