class Trainer:
	def __init__(self, model, dataloader, criterion, device, epoch_save_interval):
		self.model = model
		self.dataloader = dataloader
		self.criterion = criterion
		self.device = device
		self.epoch_save_interval = epoch_save_interval
		self.save_path = ""
		self.current_epoch = 0
		pass
	
	def _save(self, epoch):
		snapshot = {"state_dict": self.model.state_dict(), "current_epoch": epoch}
		torch.save(snapshot, self.save_path)
	
	def _load(self):
		snapshot = torch.load(self.save_path, map_location=self.device) # maybe won't work?
		self.model.load_state_dict(snapshot["state_dict"])
		self.current_epoch = snapshot["current_epoch"]


	def _epoch(self):
		for X, Y in dataloader:
			self.optimizer.zero_grad()
			y = self.model(X.to(self.device))
			loss = self.criterion(Y.to(self.device), y)
			loss.backward()
			self.optimizer.step()
			self.evaluator.update_loss(loss.cpu().detach()) # something like that
			
	def train(self):
		for i in range(self.current_epoch, n_epochs):
			_epoch()
			if (i + 1) % epoch_save_interval == 0:
				self._save(i)
		self._save(n_epochs)

def main():
	pass

if __name__ == '__main__':
	main()