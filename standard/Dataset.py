import torch

class Dataset(torch.utils.data.Dataset):
	def __init__(self, N):
		# dummy dataset
		import numpy as np
		self.Xs = np.random.random(size=(N, 2))
		self.Xs = [[i * 3 - 1.5, j * 2 - 1] for i, j in self.Xs]
		self.Ys = [0 if x2 < x1**3 - x1 else 1 for x1, x2 in self.Xs]

	def __getitem__(self, index):
		return torch.tensor(self.Xs[index], dtype=torch.double), torch.tensor(self.Ys[index], dtype=torch.double).unsqueeze(0)

	def __len__(self):
		return len(self.Xs)


def main():
	dataset = Dataset(100)
	print(dataset[0][1].size())
	print(len(dataset))

if __name__ == '__main__':
	main()