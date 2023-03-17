import torch

class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.name = "DummyModel"
		self.fc1 = torch.nn.Linear(in_features=2, out_features=8)
		self.fc2 = torch.nn.Linear(in_features=8, out_features=16)
		self.fc3 = torch.nn.Linear(in_features=16, out_features=1)

		self.relu = torch.nn.ReLU()
		self.sigmoid = torch.nn.Sigmoid()
		

	def forward(self, X):
		Y = self.fc1(X)
		Y = self.relu(Y)
		
		Y = self.fc2(Y)
		Y = self.relu(Y)
		
		Y = self.fc3(Y)
		return self.sigmoid(Y)
		