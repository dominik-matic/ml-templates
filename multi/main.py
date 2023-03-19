import torch
from Trainer import Trainer
from Evaluator import Evaluator
from Model import Model
from Dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group


def main():
	lr = 1e-3
	batch_size = 100

	dataset = Dataset(10000)
	train, valid, test = torch.utils.data.random_split(dataset, [7000, 1500, 1500])
	dl_train = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train))
	dl_valid = torch.utils.data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(valid))
	dl_test = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(test))
	
	
	model = Model().double()

	criterion = torch.nn.MSELoss()

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	trainer = Trainer(model=model,
					criterion=criterion,
					optimizer=optimizer,
					train_data=dl_train,
					valid_data=dl_valid,
					verbose=True)
	
	evaluator = Evaluator(criterion=criterion,
						test_data=dl_test,
						num_classes=2,
						verbose=True)
	
	trainer.train(100)
	loss, metric = evaluator.test(model)
	print(f'Final {loss=}')
	print(f'Confusion matrix:\n{metric.compute()}')


if __name__ == '__main__':
	init_process_group(backend="nccl")
	main()
	destroy_process_group()