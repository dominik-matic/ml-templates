import torch
from tqdm import tqdm
from torcheval.metrics import MulticlassConfusionMatrix

class Evaluator:
	def __init__(self,
				criterion,
				test_data,
				num_classes=2, # could be completely unnecessary if you're not doing a classification task
				device='cpu',
				verbose=True):
		self.criterion = criterion
		self.test_data = test_data
		self.num_classes = num_classes
		self.device = device
		self.verbose = verbose

	def _test_epoch(self, model):
		model.eval()
		
		# Change and modify this metric to whatever you want to measure
		metric = MulticlassConfusionMatrix(num_classes=self.num_classes)
		
		losses = []
		for X, Y in (pbar := tqdm(self.test_data, desc="loss=")):
			with torch.no_grad():
				y = model(X.to(self.device))
				loss = self.criterion(y, Y.to(self.device))
				losses.append(loss.item())
				
				# Change and modify this metric to whatever you want to measure
				metric.update(Y.squeeze(1).to(torch.long), torch.round(y.cpu().squeeze(1)).to(torch.long))
				
				pbar.set_description(f"loss={losses[-1]}")
		return sum(losses) / len(losses), metric

	def test(self, model):
		if self.verbose:
			print("Testing...")
		return self._test_epoch(model)
	
	