import os
import pandas as pd
import numpy as np


"""
python preprocess3.py

이상치 포함 데이터 전처리 셋 
'Gas(MQ-2)', 'Gas(MQ-4)' 데이터만 사용

"""

# 상수 값 설정
c = 0.6  # 상수
A = 0.000314  # 주어진 누출 구경
V = 448.1  # 주어진 누출 압력
M = 1000

# 농도 C 계산 함수 (상수 및 d, P만 사용)
"""
d : 누출 구경(cm)
p : 배관 내 압력
k : 기체 상수
"""
def calculate_concentration():
    C = (c * A * V)/M * (10**5) # 원래 10**6 인데 test를 위해 10**5로 진행
    return C

# 최대 절대값 정규화
def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return (a / 2 + 0.5)

def load_data():
    dataset_folder = r'C:\Users\piai\Desktop\TranAD_Anomaly_Detection\dgsp1'  # 데이터 폴더 경로
    output_folder = os.path.join(dataset_folder, 'processed')  # 처리된 데이터를 저장할 폴더
    os.makedirs(output_folder, exist_ok=True)
    
    file_list = os.listdir(dataset_folder)
    all_data = []

    for filename in file_list:
        if filename.endswith('.xls') or filename.endswith('.xlsx') or filename.endswith('.csv'):
            file_path = os.path.join(dataset_folder, filename)

            try:
                # 엑셀/CSV 파일 읽기
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)

                # 파일이 비어있거나 열이 없는 경우 건너뜀
                if df.empty or df.shape[1] == 0:
                    print(f"File {filename} is empty or has no columns. Skipping.")
                    continue

                # 필요한 센서 데이터 컬럼 리스트
                sensor_columns = ['Gas(MQ-2)', 'Gas(MQ-4)']
                sensor_columns_present = [col for col in sensor_columns if col in df.columns]
                
                if sensor_columns_present:  # 정규화할 열이 있으면
                    df_numeric = df[sensor_columns_present].apply(pd.to_numeric, errors='coerce').dropna()
                    df_normalized_clean = df_numeric[~np.isnan(df_numeric).any(axis=1)]
                    
                    all_data.append(df_normalized_clean)
            except Exception as e:
                continue

    # 데이터를 병합
    if all_data:
        all_data_combined = np.vstack(all_data)

        # 3시그마 이상치 제거
        df_all = pd.DataFrame(all_data_combined, columns=['Gas(MQ-2)', 'Gas(MQ-4)'])
        for col in df_all.columns:
            mean, std = df_all[col].mean(), df_all[col].std()
            df_all = df_all[(df_all[col] >= mean - 3 * std) & (df_all[col] <= mean + 3 * std)]

        # 데이터셋을 8:2로 분리
        split_point = int(0.8 * len(df_all))
        train_data_df = df_all.iloc[:split_point]
        test_data_df = df_all.iloc[split_point:].copy()

        # test 데이터에 label 컬럼을 추가하고 0으로 초기화
        test_data_df['label'] = 0
        
        # 이상치 데이터를 적용할 구간 설정
        total_data_len = len(test_data_df) // 180  # 600개 연속 구간을 고려할 수 있는 범위
        selected_indices = np.random.choice(total_data_len, size=66, replace=False)  # 50%
        selected_indices *= 180

        # C 값을 계산
        C = calculate_concentration()

        # 이상치 데이터 생성 
        for idx in selected_indices:
            gas_column = np.random.choice([0, 1])  # 가스 1,2를 랜덤하게 선택
            for i in range(180):  # 선택된 구간의 각 데이터에 대해 증가하는 값을 더함
                test_data_df.iloc[idx + i, gas_column] += (i + 1) * C  # (i+1) * C를 더함
            test_data_df.iloc[idx:idx + 180, 2] = 1  # 라벨을 1로 설정

        # # 정규화 부분
        # # 훈련 및 테스트 데이터를 결합하여 정규화
        # combined_data_df = pd.concat([train_data_df, test_data_df], ignore_index=True)
        # normalized_data = normalize(combined_data_df.iloc[:, :-1].values)  # 라벨 제외하고 정규화

        # # 정규화된 데이터에 라벨 추가
        # normalized_data_with_label = np.hstack([normalized_data, combined_data_df['label'].values.reshape(-1, 1)])

        # # 정규화된 데이터를 다시 훈련 및 테스트 데이터로 분리
        # normalized_train_data = normalized_data_with_label[:len(train_data_df)]
        # normalized_test_data = normalized_data_with_label[len(train_data_df):]
        
        # 데이터 저장
        np.save(os.path.join(output_folder, 'train_anomaly_normalized.npy'), train_data_df)
        np.save(os.path.join(output_folder, 'test_anomaly_normalized.npy'), test_data_df)

    else:
        print("No data processed. Please check the input files.")

if __name__ == '__main__':
    load_data()
