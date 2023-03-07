class Evaluator:
	def __init__(self,
				criterion,
				num_classes,
				valid_data,
				test_data=None,
				device='cpu',
				save_path="models/model.pt",
				verbose=True,
				print_interval=10):
		self.criterion = criterion
		self.num_classes = num_classes
		self.valid_data = valid_data
		self.test_data = test_data
		self.device = device
		self.save_path = save_path
		self.verbose = verbose
		self.print_interval = print_interval

		self.valid_losses = []
		self.best_valid_loss = None
	
	def _save_model(self, model):
		if self.verbose:
			print(f"Saving model to {self.save_path}...", end=" ")
		torch.save(model.state_dict(), self.save_path)
		if self.verbose:
			print("DONE.")

	def _epoch(self, model, data):
		model.eval()
		metric = MulticlassConfusionMatrix(num_classes=self.num_classes)
		losses = []
		for i, X, Y in enumerate(data):
			with torch.no_grad():
				y = model(X.to(self.device))
				loss = self.criterion(Y.to(self.device), y)
				losses.append(loss.item())
				metric.update(Y, y.cpu())
				if verbose and (i % self.print_interval == 0):
					print(f"\tStep {i}:\tloss={losses[-1]}")
		return metric, losses

	def validate(self, model):
		if self.verbose:
			print("Validating...")
		metric, losses = _epoch(model, self.valid_data)
		mean_valid_loss = np.mean(losses)
		self.valid_losses.append(mean_valid_loss)
		if self.best_valid_loss is None or mean_valid_loss < self.best_valid_loss:
			self._save_model(model)
		return metric, losses
	
	def test(self, model):
		if self.verbose:
			print("Testing...")
		return _epoch(model, self.test_data)
		
	def reset(self):
		self.valid_losses = []
		self.best_valid_loss = None
	
	