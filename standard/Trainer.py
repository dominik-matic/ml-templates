class Trainer:
	def __init__(self,
				model,
				train_data,
				criterion,
				evaluator=None,
				save_train_losses=True,
				device='cpu',
				epoch_save_interval=10,
				save_path="snapshots/snapshot.pt",
				verbose=True,
				print_interval=10):
		self.model = model
		self.train_data = train_data
		self.criterion = criterion
		self.evaluator = evaluator
		self.save_train_losses = save_train_losses
		self.device = device
		self.epoch_save_interval = epoch_save_interval
		self.save_path = save_path
		self.verbose = verbose
		self.print_interval = print_interval
		self.current_epoch = 0
		self.losses = []
		
	
	def _epoch(self):
		self.model.train()
		for i, X, Y in enumerate(self.train_data):
			self.optimizer.zero_grad()
			y = self.model(X.to(self.device))
			loss = self.criterion(Y.to(self.device), y)
			loss.backward()
			self.optimizer.step()
			self.losses.append(loss.item())
			if verbose and (i % self.print_interval == 0):
				print(f"\tStep {i}:\tloss={losses[-1]}")

	def _save_snapshot(self, epoch):
		if self.verbose:
			print(f"Saving snapshot to {self.save_path}...", end=" ")
		snapshot = {"state_dict": self.model.state_dict(),
					"current_epoch": epoch,
					"losses": self.losses}
		torch.save(snapshot, self.save_path)
		if self.verbose:
			print("DONE.")
	
	def load_snapshot(self, load_path):
		if self.verbose:
			print(f"Loading snapshot from {load_path}...", end=" ")
		snapshot = torch.load(load_path)
		self.model.load_state_dict(snapshot["state_dict"], map_location=device)
		self.current_epoch = snapshot["current_epoch"]
		self.losses = snapshot["losses"]
		if self.verbose:
			print("DONE")

	def train(self):
		for i in range(self.current_epoch, n_epochs):
			if self.verbose:
				print(f"Epoch {i}:")
			_epoch()
			if i % epoch_save_interval == 0:
				self._save(i)
			if self.evaluator is not None:
				self.evaluator.validate(model)
		self._save(n_epochs)

def main():
	pass

if __name__ == '__main__':
	main()