from src.parser import *
from src.folderconstants import *

# Threshold parameters (GAS 데이터에 맞게 설정)
lm_d = {
    'GAS': [(0.1, 0.9), (0.1, 0.95)]  # 정규화된 값에 맞는 threshold 설정
}
lm = lm_d['GAS'][1 if 'TranAD' in args.model else 0]

# Hyperparameters (GAS 데이터에 맞게 설정)
lr_d = {
    'GAS': 0.001  # GAS 데이터에 맞는 learning rate 설정
}
lr = lr_d['GAS']

# Debugging (GAS 데이터에 맞게 설정)
percentiles = {
    'GAS': (95, 10)  # GAS 데이터에 맞는 percentiles 설정
}
percentile_merlin = percentiles['GAS'][0]
cvp = percentiles['GAS'][1]

# 추가적인 변수 설정
preds = []
debug = 9
