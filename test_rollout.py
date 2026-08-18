import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# model definition
class HeatCNN(nn.Module):
    def __init__(self):
        super(HeatCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeatCNN().to(device)

# generate a test sample
grid_size = 64
num_simulations = 10
diffusion_coefficient = 0.1
dt = 0.1
time_steps = 20

initial_heat = torch.zeros((num_simulations, 1, grid_size, grid_size), device=device)
for i in range(num_simulations):
    num_spots = torch.randint(1, 4, (1,)).item()
    for _ in range(num_spots):
        x, y = torch.randint(0, grid_size, (2,)).tolist()
        initial_heat[i, 0, x, y] = 100.0

all_simulations = torch.zeros((num_simulations, time_steps, 1, grid_size, grid_size), device=device)
all_simulations[:, 0, :, :, :] = initial_heat
laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=device)
current_state = initial_heat.clone()
for t in range(1, time_steps):
    laplacian = torch.nn.functional.conv2d(current_state, laplacian_kernel, padding=1)
    current_state = current_state + dt * (diffusion_coefficient * laplacian)
    all_simulations[:, t, :, :, :] = current_state

# train the model again to get the weights
from torch.utils.data import Dataset, DataLoader
class HeatEquationDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return self.data.shape[0] * self.data.shape[1] 
    def __getitem__(self, idx):
        sim_idx = idx // self.data.shape[1]
        t_idx = idx % self.data.shape[1]
        x = self.data[sim_idx, t_idx, :, :, :] 
        y = self.data[sim_idx, t_idx + 1, :, :, :] if t_idx < self.data.shape[1] - 1 else self.data[sim_idx, t_idx, :, :, :]
        return x, y

dataset = HeatEquationDataset(all_simulations)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Re-training model for a few epochs to get weights...")
for epoch in range(20):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print("Done re-training!")

# run the rollout test
# start from simulation 0 at time step 0
model.eval()
starting_state = all_simulations[0, 0, :, :, :].unsqueeze(0).to(device)  # Shape: (1, 1, 64, 64)

# get the ground truth from the physics solver
ground_truth_steps = all_simulations[0, :, :, :, :].cpu().numpy()

# make predictions step by step
predicted_steps = [starting_state.cpu().numpy()]

current_input = starting_state
for t in range(1, time_steps):
    with torch.no_grad():
        next_state = model(current_input)  # Predict t+1
    predicted_steps.append(next_state.cpu().numpy())
    current_input = next_state  # Feed prediction back into the model

predicted_steps = np.array(predicted_steps) # Shape: (20, 1, 64, 64)

# compare the results
# look at the final time step to see how much the model drifted
plt.figure(figsize=(15, 4))

# remove the extra dimensions so the image is (64,64)
plt.subplot(1, 3, 1)
plt.imshow(ground_truth_steps[0].squeeze(), cmap='hot')
plt.title("Ground Truth (t=0)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(ground_truth_steps[-1].squeeze(), cmap='hot')
plt.title("Ground Truth (t=19)")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(predicted_steps[-1].squeeze(), cmap='hot')
plt.title("CNN Rollout Prediction (t=19)")
plt.colorbar()

plt.suptitle("Long-term Rollout Test")
plt.tight_layout()
plt.show()

# calculate the error
error = np.abs(ground_truth_steps[-1].squeeze() - predicted_steps[-1].squeeze())
plt.figure(figsize=(6, 5))
plt.imshow(error, cmap='viridis')
plt.title(f"Absolute Error at t=19 (Max: {error.max():.2f})")
plt.colorbar()
plt.show()
