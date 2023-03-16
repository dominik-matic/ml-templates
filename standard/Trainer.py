import tqdm

class Trainer:
	def __init__(self,
				model,
				criterion,
				train_data,
				valid_data=None,
				save_train_losses=True,
				save_valid_losses=True,
				device='cpu',
				epoch_save_interval=10,
				snapshot_path="snapshots/snapshot.ss",
				model_folder="models/",
				verbose=True):
		self.model = model
		self.criterion = criterion
		self.train_data = train_data
		self.valid_data = valid_data
		self.save_train_losses = save_train_losses
		self.save_valid_losses = save_valid_losses
		self.device = device
		self.epoch_save_interval = epoch_save_interval
		self.snapshot_path = snapshot_path
		self.model_folder = model_folder
		self.verbose = verbose
		self.current_epoch = 0
		self.train_losses = []
		self.valid_losses = []
		self.lowest_valid_loss = None
		
	

	def _train_epoch(self):
		self.model.train()
		losses = []
		for i, X, Y in (pbar := tqdm(enumerate(self.train_data), desc="loss = ")):
			self.optimizer.zero_grad()
			y = self.model(X.to(self.device))
			loss = self.criterion(Y.to(self.device), y)
			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())
			pbar.set_description(f"loss = {losses[-1]}")
		return sum(losses) / len(losses)
		#self.train_losses.append(sum(losses)/len(losses))
	
	def _valid_epoch(self, data):
		model.eval()
		losses = []
		for i, X, Y in (pbar := tqdm(enumerate(data), desc="loss = ")):
			with torch.no_grad():
				y = model(X.to(self.device))
				loss = self.criterion(Y.to(self.device), y)
				losses.append(loss.item())
				pbar.set_description(f"loss = {losses[-1]}")
		return sum(losses) / len(losses)



	def _save_snapshot(self, epoch):
		if self.verbose:
			print(f"Saving snapshot to {self.snapshot_path}...", end=" ")
		snapshot = {"state_dict": self.model.state_dict(),
					"current_epoch": epoch,
					"train_losses": self.train_losses,
					"valid_losses": self.valid_losses,
					"lowest_valid_loss": self.lowest_valid_loss}
		torch.save(snapshot, self.snapshot_path)
		if self.verbose:
			print("DONE.")
	
	def load_snapshot(self, load_path):
		if self.verbose:
			print(f"Loading snapshot from {load_path}...", end=" ")
		snapshot = torch.load(load_path)
		self.model.load_state_dict(snapshot["state_dict"], map_location=device)
		self.current_epoch = snapshot["current_epoch"]
		self.train_losses = snapshot["train_losses"]
		self.valid_losses = snapshot["valid_losses"]
		self.lowest_valid_loss = snapshot["lowest_valid_loss"]
		if self.verbose:
			print("DONE")
	
	def _save_model(self, epoch):
		save_path = self.model_folder
		if hasattr(self.model, "name"):
			save_path += model.name + f"_e{epoch}.pt"
		else:
			save_path += f"model_e{epoch}.pt"
		if self.verbose:
			print(f"Saving model to {save_path}...", end=" ")
		torch.save(self.model.state_dict(), save_path)
		if self.verbose:
			print("DONE.")

	def train(self, n_epochs):
		# maybe this initial loss calculation is unnecessary?
		if self.current_epoch == 0:
			if verbose:
				print("Calculating initial losses on training and valid sets")
			l = self._valid_epoch(self.train_data)
			self.train_losses.append(l)
			l = self._valid_epoch(self.valid_data)
			self.valid_losses.append(l)
		for i in (pbar := tqdm(range(self.current_epoch, n_epochs), desc="Epoch 0 loss = ")):
			l = self._train_epoch()
			self.train_losses.append(l)
			l = self._valid_epoch(self.valid_data)
			self.valid_losses.append(l)
			if self.lowest_valid_loss is None or l < self.lowest_valid_loss:
				self.lowest_valid_loss = l
				self._save_model(i)
			if i % epoch_save_interval == 0:
				self._save_snapshot(i)
		self._save_snapshot(n_epochs)

def main():
	pass

if __name__ == '__main__':
	main()