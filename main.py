import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
import torch.utils.checkpoint as checkpoint  # checkpoint 모듈 임포트
from src.changedata import *



def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, _ in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w.float() if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1).float())  # float32로 변환

	return torch.stack(windows)



def load_dataset(dataset):
    # 전처리된 데이터 로드 (단일 데이터셋)
    dataset_folder = r'C:\Users\piai\Desktop\TranAD_Anomaly_Detection\dgsp1\processed'  # 전처리된 데이터를 저장한 폴더
	
	# 이상치 데이터 셋에 이상치+Label 포함
    train_data = np.load(os.path.join(dataset_folder, 'train_anomaly_normalized.npy'))
    test_data = np.load(os.path.join(dataset_folder, 'test_anomaly_normalized.npy'))

    # PyTorch DataLoader로 변환
    train_loader = DataLoader(train_data, batch_size=train_data.shape[0])
    test_loader = DataLoader(test_data, batch_size=test_data.shape[0])
    
    return train_loader, test_loader


def save_model(model, optimizer, scheduler, epoch, accuracy_list, hyperparameters):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)

    # 하이퍼파라미터를 기반으로 파일명 생성
    hyper_str = f"{hyperparameters['model']}_ws{hyperparameters['window_size']}_lr{hyperparameters['lr']}_dims{hyperparameters['dims']}_bs{hyperparameters['batch_size']}"
    file_path = f'{folder}/model_{hyper_str}.ckpt'

    # 모델 저장prepr
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list,
        'hyperparameters': hyperparameters
    }, file_path)
    print(f"Model saved to {file_path}")


def load_model(modelname,hyperparameters):
	import src.models
	model_class = getattr(src.models, modelname) #모델 가져오기
	model = model_class(hyperparameters['dims']).double() # 가져온 모델 클래스를 인스턴스화, dims는 입력 차원 수로 사용 / 이 값을 바탕으로 모델 초기화

	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

	# 하이퍼파라미터를 기반으로 파일명 생성
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	hyper_str = f"{hyperparameters['model']}_ws{hyperparameters['window_size']}_lr{hyperparameters['lr']}_dims{hyperparameters['dims']}_bs{hyperparameters['batch_size']}"
	fname = f'{folder}/model_{hyper_str}.ckpt'

	print(f"Path exists: {os.path.exists(fname)}")
	print(f"Args retrain: {args.retrain}")
	print(f"Args test: {args.test}")

	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		#새로운 모델 생성
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list



def backprop(epoch, model, data, optimizer, scheduler, training = True):

	has_printed = False
	feats = data.shape[1]
	# MSELoss(Mean Squared Error 손실 함수)를 사용
	l = nn.MSELoss(reduction = 'mean' if training else 'none')

	if 'DAGMM' in model.name:
		l = nn.MSELoss(reduction = 'none')
		compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
		n = epoch + 1; w_size = model.n_window
		l1s = []; l2s = []
		if training:
			for d in data:
				_, x_hat, z, gamma = model(d)
				l1, l2 = l(x_hat, d), l(gamma, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1) + torch.mean(l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s = []
			for d in data: 
				_, x_hat, _, _ = model(d)
				ae1s.append(x_hat)
			ae1s = torch.stack(ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	# if 'Attention' in model.name:
	# 	l = nn.MSELoss(reduction = 'none')
	# 	n = epoch + 1; w_size = model.n_window
	# 	l1s = []; res = []
	# 	if training:
	# 		for d in data:
	# 			ae, ats = model(d)
	# 			# res.append(torch.mean(ats, axis=0).view(-1))
	# 			l1 = l(ae, d)
	# 			l1s.append(torch.mean(l1).item())
	# 			loss = torch.mean(l1)
	# 			optimizer.zero_grad()
	# 			loss.backward()
	# 			optimizer.step()
	# 		# res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
	# 		scheduler.step()
	# 		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
	# 		return np.mean(l1s), optimizer.param_groups[0]['lr']
	# 	else:
	# 		ae1s, y_pred = [], []
	# 		for d in data: 
	# 			ae1 = model(d)
	# 			y_pred.append(ae1[-1])
	# 			ae1s.append(ae1)
	# 		ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
	# 		loss = torch.mean(l(ae1s, data), axis=1)
	# 		return loss.detach().numpy(), y_pred.detach().numpy()
	# elif 'OmniAnomaly' in model.name:
	# 	if training:
	# 		mses, klds = [], []
	# 		for i, d in enumerate(data):
	# 			y_pred, mu, logvar, hidden = model(d, hidden if i else None)
	# 			MSE = l(y_pred, d)
	# 			KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
	# 			loss = MSE + model.beta * KLD
	# 			mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
	# 			optimizer.zero_grad()
	# 			loss.backward()
	# 			optimizer.step()
	# 		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
	# 		scheduler.step()
	# 		return loss.item(), optimizer.param_groups[0]['lr']
	# 	else:
	# 		y_preds = []
	# 		for i, d in enumerate(data):
	# 			y_pred, _, _, hidden = model(d, hidden if i else None)
	# 			y_preds.append(y_pred)
	# 		y_pred = torch.stack(y_preds)
	# 		MSE = l(y_pred, data)
	# 		return MSE.detach().numpy(), y_pred.detach().numpy()
	# elif 'USAD' in model.name:
	# 	l = nn.MSELoss(reduction = 'none')
	# 	n = epoch + 1; w_size = model.n_window
	# 	l1s, l2s = [], []
	# 	if training:
	# 		for d in data:
	# 			ae1s, ae2s, ae2ae1s = model(d)
	# 			l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
	# 			l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
	# 			l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
	# 			loss = torch.mean(l1 + l2)
	# 			optimizer.zero_grad()
	# 			loss.backward()
	# 			optimizer.step()
	# 		scheduler.step()
	# 		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
	# 		return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
	# 	else:
	# 		ae1s, ae2s, ae2ae1s = [], [], []
	# 		for d in data: 
	# 			ae1, ae2, ae2ae1 = model(d)
	# 			ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
	# 		ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
	# 		y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
	# 		loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
	# 		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
	# 		return loss.detach().numpy(), y_pred.detach().numpy()
	elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		l = nn.MSELoss(reduction='none')
		n = epoch + 1
		w_size = model.n_window
		l1s = []

		if training:
			h = None
			dataloader = DataLoader(TensorDataset(data), batch_size=bs)
			for i, d in enumerate(dataloader):
				d = d.to(device0)  # 데이터도 GPU로 이동
				if 'MTAD_GAT' in model.name:
					h = h.to(device0) if h is not None else None  # 상태 h를 GPU로 이동
					x, h = model(d, h if i else None)
				else:
					x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()

				# 역전파와 그래프 디버깅 활성화
				with torch.autograd.set_detect_anomaly(True):
					retain_graph = i < len(data) - 1  # 마지막 배치에서는 그래프 유지 불필요
					loss.backward(retain_graph=retain_graph)  # 역전파

				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			xs = []
			for d in data:
				d = d.to(device0)  # 데이터도 GPU로 이동
				if 'MTAD_GAT' in model.name:
					x, h = model(d, None)
				else:
					x = model(d)
				xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data.to(device0))  # 여기서도 data를 GPU로 이동
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()  # GPU 텐서를 CPU로 이동하여 반환
	# elif 'GAN' in model.name:
	# 	l = nn.MSELoss(reduction = 'none')
	# 	bcel = nn.BCELoss(reduction = 'mean')
	# 	msel = nn.MSELoss(reduction = 'mean')
	# 	real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
	# 	real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
	# 	n = epoch + 1; w_size = model.n_window
	# 	mses, gls, dls = [], [], []
	# 	if training:
	# 		for d in data:
	# 			# training discriminator
	# 			model.discriminator.zero_grad()
	# 			_, real, fake = model(d)
	# 			dl = bcel(real, real_label) + bcel(fake, fake_label)
	# 			dl.backward()
	# 			model.generator.zero_grad()
	# 			optimizer.step()
	# 			# training generator
	# 			z, _, fake = model(d)
	# 			mse = msel(z, d) 
	# 			gl = bcel(fake, real_label)
	# 			tl = gl + mse
	# 			tl.backward()
	# 			model.discriminator.zero_grad()
	# 			optimizer.step()
	# 			mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
	# 			# tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
	# 		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
	# 		return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
	# 	else:
	# 		outputs = []
	# 		for d in data: 
	# 			z, _, _ = model(d)
	# 			outputs.append(z)
	# 		outputs = torch.stack(outputs)
	# 		y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
	# 		loss = l(outputs, data)
	# 		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
	# 		return loss.detach().numpy(), y_pred.detach().numpy()
	
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction='mean')
		bs = model.batch if training else len(data)
		
		if training:
			dataloader = DataLoader(TensorDataset(data), batch_size=bs)
			print(f"Epoch: {epoch}, Training: {training}")  # 현재 Epoch와 Training 여부 확인
			l1s = []
			for batch_data in dataloader: 
				# Half precision and move to device for memory efficiency
				batch_data = batch_data[0].permute(1, 0, 2).to(device0).float()
				elem = batch_data[-1, :, :].view(1, -1, model.n_feats).float()
				
				z = model(batch_data, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else \
					(1 / (epoch + 1)) * l(z[0], elem) + (1 - 1/(epoch + 1)) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]

				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr'],l1s
		
		else:
			# 데이터 로더 생성
			dataloader = DataLoader(TensorDataset(data), batch_size=128)

			# 전체 테스트 손실 및 예측값을 저장할 리스트
			losses, predictions = [], []

			for batch_data in dataloader:  # dataloader에서 데이터를 배치 단위로 반복 (batch_size, sequence_length, feature_count) (128, 60, 3)

				# 데이터를 필요한 장치로 옮기고 차원을 조정
				# (batch_size, sequence_length, feature_count) → (sequence_length, batch_size, feature_count) / (128, 60, 3) → (60, 128, 3).
				batch_data = batch_data[0].permute(1, 0, 2).to(device0).float() 

				# batch_data의 마지막 시점 (-1)에서 batch_size와 feature_count를 추출하여 elem을 구성
				# (sequence_length, batch_size, feature_count) → (batch_size, feature_count) → (1, batch_size, feature_count). (60, 128, 3)에서 -1 인덱스로 선택 → (128, 3) → (1, 128, 3).
				elem = batch_data[-1, :, :].view(1, -1, model.n_feats).float()
				
				# 모델에 데이터를 전달하고 예측값(z) 생성
				z = model(batch_data, elem)
				
				if isinstance(z, tuple):
					z = z[1]  # tuple 형태면 Phase 2의 결과로 설정

				# 손실 계산 및 저장
				loss = torch.abs(z - elem)
				#losess.append(loss.detach().cpu().numpy().reshape(-1, model.n_feats)) # sequense 별로
				losses.append(loss.detach().cpu().numpy())
				predictions.append(z.detach().cpu().numpy().reshape(-1, model.n_feats))  # 예측값을 동일한 형태로 변환해 저장

				# 모든 배치의 결과를 결합하여 반환 (마지막 배치 크기가 작아도 처리 가능)
				losses = [loss.reshape(-1, model.n_feats) for loss in losses]  # 모든 배치 크기 통일
				predictions = [pred.reshape(-1, model.n_feats) for pred in predictions]
			
			# 모든 배치의 결과를 결합하여 반환 (마지막 배치 크기 다를 경우에도 자동으로 결합)
			return np.vstack(losses), np.vstack(predictions)

	else:
		y_pred = model(data.to(device0))  # 데이터를 GPU로 전송
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()


if __name__ == '__main__':

	device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Cuda available: {torch.cuda.is_available()}")

	# Load data
	train_loader, test_loader = load_dataset(args.dataset)

	hyperparameters = {
		'model' : 'TranAD',
		'window_size': 60,
		'lr': 0.0001,
		'dims': 2,
		'batch_size': 128
	}
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, hyperparameters)

	# model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, 2)
	model = model.to(device0).float()  # Move model to device in half precision
	
	# Load data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	actual = testD[:, 2].cpu().numpy()

	# Use only required columns for training and testing
	trainD = trainD[:, :2].float()
	testD = testD[:, :2].float()
	
	# Convert to device1
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

	testO = testD

	# Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 5; e = epoch + 1; start = time()
		
		for e in tqdm(range(epoch+1, epoch+num_epochs+1), dynamic_ncols=True):
			lossT, lr, train_loss = backprop(e, model, trainD, optimizer, scheduler)
			accuracy_list.append((lossT, lr))  
			

		print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time()-start) + ' s' + color.ENDC)
		#save_model(model, optimizer, scheduler, e, accuracy_list)

		hyperparameters = {
			'model' : 'TranAD',
			'window_size': 60,
			'lr': 0.0001,
			'dims': 2,
			'batch_size': 128
		}
		save_model(model, optimizer, scheduler, epoch, accuracy_list, hyperparameters)

		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')


	with torch.no_grad():
		testD = testD.to(device0)

		model.eval()
		print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')

		test_losses, y_pred = backprop(0, model, testD, optimizer, scheduler, training=False)

		pred_labels, anomalies, dynamic_thresholds, smoothed_losses = evaluate_anomalies_normal(test_losses, z=7)

		# pot_dynamic_threshold = pot_calculate_thresholds_per_column(train_loss, test_losses, q=1e-5, level=0.02, window_size=2500)

		# Nomal confusion metrix
		pred_labels, actual = np.array(pred_labels, dtype=int), np.array(actual, dtype=int)


		# PA 예측값 계산
		pa_predictions = calculate_pa(test_losses, dynamic_thresholds)

		# PA%k 예측값 계산
		k_percent = 0.5  # 30% 이상의 시점이 임계값을 넘으면 구간을 이상치로 간주
		pa_k_predictions = calculate_pa_k(test_losses, dynamic_thresholds, k_percent)
		pa_k_predictions = pa_k_predictions.squeeze() # 불필요한 차원 제거

		# Af β Score
		af_beta_score = calculate_af_beta(actual, pred_labels)
		d_s1, t_s2, accuracy1 = evaluate_anomaly_segments(actual, pred_labels, threshold=0.5)
		d_s2, t_s2, accuracy2 = evaluate_anomaly_segments(actual, pa_predictions, threshold=0.5)
		d_s3, t_s3, accuracy3 = evaluate_anomaly_segments(actual, pa_k_predictions, threshold=0.5)

		normal_f1, normal_precision, normal_recall, normal_TP, normal_TN, normal_f1_FP, normal_FN, normal_roc_auc = calc_point2point(pred_labels, actual)
		PA_f1, PA_precision, PA_recall, PA_TP, PA_TN, PA_FP, PA_FN, PA_roc_auc = calc_point2point(pa_predictions, actual)
		PA_K_f1, PA_K_precision, PA_K_recall, PA_K_TP, PA_K_TN, PA_K_FP, PA_K_FN, PA_K_roc_auc = calc_point2point(pa_k_predictions, actual)

		print(f"Normal F1 Score: {normal_f1}, Precision: {normal_precision}, Recall: {normal_recall}, ROC AUC: {normal_roc_auc}")
		print(f"Normal Detected anomaly segments: {d_s1}/{t_s2}")
		print(f"Accuracy of segment detection: {accuracy1:.2%}\n")


		print(f"PA F1 Score: {PA_f1}, Precision: {PA_precision}, Recall: {PA_recall}, ROC AUC: {PA_roc_auc}")
		print(f"PA Detected anomaly segments: {d_s2}/{t_s2}")
		print(f"Accuracy of segment detection: {accuracy2:.2%}\n")

		print(f"PA%k F1 Score: {PA_K_f1}, Precision: {PA_K_precision}, Recall: {PA_K_recall}, ROC AUC: {PA_K_roc_auc}")
		print(f"PA%k Detected anomaly segments: {d_s3}/{t_s3}")
		print(f"Accuracy of segment detection: {accuracy3:.2%}\n")

		print(f"Af β Score: {af_beta_score}")

		# Plot
		if 'TranAD' in model.name:
			testO = torch.roll(testO, 1, 0)
		plotter(f'{args.model}_{args.dataset}', testO, y_pred, smoothed_losses, dynamic_thresholds=dynamic_thresholds, anomalies=anomalies)


