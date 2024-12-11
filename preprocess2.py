import os
import pandas as pd
import numpy as np

#### 이상치 데이터 포함 데이터 셋 ####


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
                # sensor_columns = ['Temprature', 'Gas(MQ-2)', 'Gas(MQ-4)']
                sensor_columns_present = [col for col in sensor_columns if col in df.columns]
                
                if sensor_columns_present:  # 정규화할 열이 있으면
                    df_numeric = df[sensor_columns_present].apply(pd.to_numeric, errors='coerce').dropna()
                    # 정규화 x
                    df_normalized_clean = df_numeric[~np.isnan(df_numeric).any(axis=1)]

                    # # 정규화
                    # df_normalized = normalize(df_numeric.values)
                    # # NaN이 있는 행 제거
                    # df_normalized_clean = df_normalized[~np.isnan(df_normalized).any(axis=1)]

                    
                    all_data.append(df_normalized_clean)
            except Exception as e:
                continue

    # 데이터를 병합
    if all_data:
        all_data_combined = np.vstack(all_data)

         # 전체 데이터의 절반만 사용
        half_data_len = int(len(all_data_combined) * 0.001)  # 0.01은 원하는 비율
        all_data_combined = all_data_combined[:half_data_len]  # 절반만 사용

        # 전체 데이터
        half_data_len = len(all_data_combined)  # 0.01은 원하는 비율
        all_data_combined = all_data_combined[:half_data_len]  # 절반만 사용

        # 3시그마 이상치 제거
        # df_all = pd.DataFrame(all_data_combined, columns=['Temprature', 'Gas(MQ-2)', 'Gas(MQ-4)'])
        df_all = pd.DataFrame(all_data_combined, columns=[ 'Gas(MQ-2)', 'Gas(MQ-4)'])
        for col in df_all.columns:
            mean, std = df_all[col].mean(), df_all[col].std()
            df_all = df_all[(df_all[col] >= mean - 3 * std) & (df_all[col] <= mean + 3 * std)]

        # 데이터셋을 8:2로 분리
        split_point = int(0.8 * len(df_all))
        train_data_df = df_all.iloc[:split_point]
        test_data_df = df_all.iloc[split_point:].copy()

        # test 데이터에 label 컬럼을 추가하고 0으로 초기화
        test_data_df['label'] = 0
        
        # 약 1200000개의 초 데이터 중 0.01%인 12000개 정도 이상치 데이터 생성할 것 그 중 6분인 360개의 데이터를 하나의 구간으로
        # 연속된 600개 구간에서 20개의 구간을 랜덤하게 선택하여 *5 적용
        total_data_len = len(test_data_df)//60  # 600개 연속 구간을 고려할 수 있는 범위
     
        selected_indices = np.random.choice(total_data_len, size=11, replace=False) # 50%
        selected_indices *= 60

        for idx in selected_indices:
            gas_column = np.random.choice([0, 1]) # 가스 1,2를 랜덤하게 choice
            test_data_df.iloc[idx:idx+360, gas_column] *= 5  # 연속 구간을 *5하여 확장, 라벨 제외 컬럼만 적용
            test_data_df.iloc[idx:idx+360, 2] = 1  # 확장된 구간의 label을 1로 설정


        # # # 훈련 및 테스트 데이터를 결합하여 정규화
        # combined_data_df = pd.concat([train_data_df, test_data_df], ignore_index=True)
        # normalized_data = normalize(combined_data_df.iloc[:, :-1].values)  # 라벨 제외하고 정규화

        # # 정규화된 데이터에 라벨 추가
        # normalized_data_with_label = np.hstack([normalized_data, combined_data_df['label'].values.reshape(-1, 1)])

        # # 정규화된 데이터를 다시 훈련 및 테스트 데이터로 분리
        # normalized_train_data = normalized_data_with_label[:len(train_data_df)]
        # normalized_test_data = normalized_data_with_label[len(train_data_df):]
        
        # 정규화된 데이터를 저장
        np.save(os.path.join(output_folder, 'train_anomaly_normalized.npy'), train_data_df)
        np.save(os.path.join(output_folder, 'test_anomaly_normalized.npy'), test_data_df)

        # np.save(os.path.join(output_folder, 'train_anomaly_normalized.npy'), train_data_df)
        # np.save(os.path.join(output_folder, 'test_anomaly_normalized.npy'), test_data_df)

    else:
        print("No data processed. Please check the input files.")

if __name__ == '__main__':
    load_data()





