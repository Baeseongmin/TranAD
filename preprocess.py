import os
import pandas as pd
import numpy as np





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
                sensor_columns = ['Temprature', 'Gas(MQ-2)', 'Gas(MQ-4)']

                # 실제로 데이터프레임에 존재하는 센서 데이터 열만 선택
                sensor_columns_present = [col for col in sensor_columns if col in df.columns]

                if sensor_columns_present:  # 정규화할 열이 있으면
                    
                    # 숫자형 데이터만 선택하고, 문자열 및 NaN이 있는 행은 제거
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
    if all_data:  # 데이터가 있으면
        all_data_combined = np.vstack(all_data)

        # 전체 데이터 상용
        data_subset = all_data_combined[:int(len(all_data_combined))]

        # # 전체 데이터 사용
        # data_subset = all_data_combined[:int(len(all_data_combined))]
        # train, test 데이터 분리 (80%는 train, 20%는 test로 나눔)
        split_point = int(0.8 * len(data_subset))
        train_data = data_subset[:split_point]
        test_data = data_subset[split_point:]

        # 학습 데이터와 테스트 데이터를 .npy 파일로 저장
        np.save(os.path.join(output_folder, 'train.npy'), train_data)
        np.save(os.path.join(output_folder, 'test.npy'), test_data)

    else:
        print("No data processed. Please check the input files.")

if __name__ == '__main__':
    load_data()
