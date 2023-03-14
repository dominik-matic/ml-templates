class Evaluator:
	def __init__(self,
				criterion,
				test_data,
				num_classes,
				device='cpu',
				verbose=True):
		self.criterion = criterion
		self.test_data = test_data
		self.num_classes = num_classes
		self.device = device
		self.verbose = verbose

	def _test_epoch(self, model):
		model.eval()
		metric = MulticlassConfusionMatrix(num_classes=self.num_classes)
		losses = []
		for i, X, Y in (pbar := tqdm(enumerate(self.test_data), desc="loss = ")):
			with torch.no_grad():
				y = model(X.to(self.device))
				loss = self.criterion(Y.to(self.device), y)
				losses.append(loss.item())
				metric.update(Y, y.cpu())
				pbar.set_description(f"loss = {losses[-1]}")
		return sum(losses) / len(losses), metric

	def test(self, model):
		if self.verbose:
			print("Testing...")
		return _epoch(model)
	
	