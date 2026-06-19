import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from neuralop.models import FNO  # <--- The magic library!

# --- 1. SETUP & DATA GENERATION (Same as before) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

grid_size = 64
num_simulations = 10
diffusion_coefficient = 0.1
dt = 0.1
time_steps = 20

# Generate data
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

# --- 2. DATASET PREP ---
class HeatEquationDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0] * self.data.shape[1] 
    def __getitem__(self, idx):
        sim_idx = idx // self.data.shape[1]
        t_idx = idx % self.data.shape[1]
        x = self.data[sim_idx, t_idx, :, :, :] 
        if t_idx < self.data.shape[1] - 1:
            y = self.data[sim_idx, t_idx + 1, :, :, :]
        else:
            y = self.data[sim_idx, t_idx, :, :, :] 
        return x, y

dataloader = DataLoader(HeatEquationDataset(all_simulations), batch_size=16, shuffle=True)

# --- 3. DEFINE THE FNO MODEL ---
# An FNO is much more complex, but the library makes it easy!
# We choose:
#   n_modes: How many Fourier frequencies to keep (higher = more accurate, slower)
#   hidden_channels: How many "neurons" in the middle
model = FNO(n_modes=(16, 16), 
            hidden_channels=32, 
            in_channels=1, 
            out_channels=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"FNO has {sum(p.numel() for p in model.parameters())} parameters")

# --- 4. TRAINING LOOP ---
num_epochs = 50
print("\nStarting FNO Training...")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.6f}")

# --- 5. FNO ROLLOUT TEST ---
model.eval()
starting_state = all_simulations[0, 0, :, :, :].unsqueeze(0).to(device)
ground_truth_steps = all_simulations[0, :, :, :, :].cpu().numpy()

predicted_steps = [starting_state.cpu().numpy()]
current_input = starting_state
for t in range(1, time_steps):
    with torch.no_grad():
        next_state = model(current_input)
    predicted_steps.append(next_state.cpu().numpy())
    current_input = next_state

predicted_steps = np.array(predicted_steps)

# --- 6. PLOT COMPARISON ---
plt.figure(figsize=(15, 4))

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
plt.title("FNO Rollout Prediction (t=19)")
plt.colorbar()

plt.suptitle("FNO Long-term Rollout Test")
plt.tight_layout()
plt.show()

error = np.abs(ground_truth_steps[-1].squeeze() - predicted_steps[-1].squeeze())
plt.figure(figsize=(6, 5))
plt.imshow(error, cmap='viridis')
plt.title(f"FNO Absolute Error at t=19 (Max: {error.max():.2f})")
plt.colorbar()
plt.show()