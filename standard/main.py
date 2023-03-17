import torch
from Trainer import Trainer
from Evaluator import Evaluator
from Model import Model
from Dataset import Dataset

def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f'Using {device=}')

	lr = 1e-3
	batch_size = 100

	dataset = Dataset(10000)
	train, valid, test = torch.utils.data.random_split(dataset, [7000, 1500, 1500])
	dl_train = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
	dl_valid = torch.utils.data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=True)
	dl_test = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
	
	
	model = Model().double().to(device)

	criterion = torch.nn.MSELoss()

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	trainer = Trainer(model=model,
					criterion=criterion,
					optimizer=optimizer,
					train_data=dl_train,
					valid_data=dl_valid,
					device=device,
					verbose=True)
	
	evaluator = Evaluator(criterion=criterion,
						test_data=dl_test,
						num_classes=2,
						device=device,
						verbose=True)
	
	trainer.train(100)
	loss, metric = evaluator.test(model)
	print(f'Final {loss=}')
	print(f'Confusion matrix:\n{metric.compute()}')


if __name__ == '__main__':
	main()