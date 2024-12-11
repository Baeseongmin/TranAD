import numpy as np
from sklearn.metrics import ndcg_score
from src.constants import lm

# 모델이 예측한 순위 리스트에서 상위 **p%**에 **실제 이상치(또는 레이블이 1인 데이터)**가 얼마나 포함되어 있는지를 평가
#  **Hit@100%**가 0.75라면, 상위 100%의 항목 중 75%가 실제 이상치로 잘 맞춰졌다는 의미
def hit_att(ascore, labels, ps = [100, 150]):
	res = {}
	for p in ps:
		hit_score = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0])
			if l:
				size = round(p * len(l) / 100)
				a_p = set(a[:size])
				intersect = a_p.intersection(l)
				hit = len(intersect) / len(l)
				hit_score.append(hit)
		res[f'Hit@{p}%'] = np.mean(hit_score)
	return res

# 순위가 높은 위치에서 예측된 이상치가 얼마나 잘 맞았는지를 가중치를 두고 평가
# *NDCG@100%**가 0.85라면, 상위 100%에서 모델의 예측이 이상적으로 맞춘 결과의 85% 정도의 성능을 보여주고 있다는 뜻
def ndcg(ascore, labels, ps = [100, 150]):
	res = {}
	for p in ps:
		ndcg_scores = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			labs = list(np.where(l == 1)[0])
			if labs:
				k_p = round(p * len(labs) / 100)
				try:
					hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k = k_p)
				except Exception as e:
					return {}
				ndcg_scores.append(hit)
		res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
	return res



