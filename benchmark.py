import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from neuralop.models import FNO
import numpy as np
import time
import matplotlib.pyplot as plt

# --- SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Benchmarking on: {device}")

grid_size = 64
num_simulations = 10
diffusion_coefficient = 0.1
dt = 0.1
time_steps = 20

# --- DATA GENERATION ---
def generate_data():
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
    return all_simulations

all_simulations = generate_data()

# --- DATASET ---
class HeatEquationDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0] * self.data.shape[1] 
    def __getitem__(self, idx):
        sim_idx = idx // self.data.shape[1]
        t_idx = idx % self.data.shape[1]
        x = self.data[sim_idx, t_idx, :, :, :] 
        y = self.data[sim_idx, t_idx + 1, :, :, :] if t_idx < self.data.shape[1] - 1 else self.data[sim_idx, t_idx, :, :, :]
        return x, y

dataloader = DataLoader(HeatEquationDataset(all_simulations), batch_size=16, shuffle=True)

# --- MODEL DEFINITIONS ---
class HeatCNN(nn.Module):
    def __init__(self):
        super(HeatCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- TRAIN FUNCTION ---
def train_model(model, name, epochs=20):
    print(f"Training {name} for {epochs} epochs...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"{name} training complete. Saving to disk...")
    torch.save(model.state_dict(), f"{name}_model.pth")
    return model

# Load or Train CNN
cnn = HeatCNN().to(device)
try:
    cnn.load_state_dict(torch.load("cnn_model.pth", map_location=device))
    print("Loaded CNN from disk.")
except:
    cnn = train_model(cnn, "cnn", epochs=20)

# Load or Train FNO
fno = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=1, out_channels=1).to(device)
try:
    fno.load_state_dict(torch.load("fno_model.pth", map_location=device))
    print("Loaded FNO from disk.")
except:
    fno = train_model(fno, "fno", epochs=20)

# --- BENCHMARK 1: SPEED COMPARISON ---
print("\nRunning speed benchmarks...")
test_input = all_simulations[0, 0, :, :, :].unsqueeze(0).to(device)

# Physics Solver (rolling out 20 steps)
start = time.time()
for _ in range(100):
    state = test_input.clone()
    for t in range(1, time_steps):
        lap = torch.nn.functional.conv2d(state, torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=device), padding=1)
        state = state + dt * (diffusion_coefficient * lap)
physics_time = time.time() - start

# CNN (rolling out 20 steps)
start = time.time()
for _ in range(100):
    state = test_input.clone()
    for t in range(1, time_steps):
        state = cnn(state)
cnn_time = time.time() - start

# FNO (rolling out 20 steps)
start = time.time()
for _ in range(100):
    state = test_input.clone()
    for t in range(1, time_steps):
        state = fno(state)
fno_time = time.time() - start

# Plot Speed
plt.figure(figsize=(8, 5))
plt.bar(["Physics Solver", "CNN", "FNO"], [physics_time, cnn_time, fno_time], color=['gray', 'orange', 'blue'])
plt.ylabel("Time (seconds) for 100 rollouts (20 steps each)")
plt.title("Inference Speed Benchmark (CPU)")
plt.grid(axis='y', linestyle='--')
plt.savefig("results/speed_benchmark.png")
plt.close()

# --- BENCHMARK 2: ERROR ACCUMULATION OVER TIME ---
print("Calculating error accumulation over time...")
cnn_errors = []
fno_errors = []
gt = all_simulations[0, :, :, :, :].cpu().numpy()

state_cnn = all_simulations[0, 0, :, :, :].unsqueeze(0).to(device)
state_fno = state_cnn.clone()

for t in range(1, time_steps):
    state_cnn = cnn(state_cnn)
    state_fno = fno(state_fno)
    
    cnn_err = torch.nn.functional.mse_loss(state_cnn, torch.tensor(gt[t]).unsqueeze(0).to(device)).item()
    fno_err = torch.nn.functional.mse_loss(state_fno, torch.tensor(gt[t]).unsqueeze(0).to(device)).item()
    
    cnn_errors.append(cnn_err)
    fno_errors.append(fno_err)

# Plot Error
plt.figure(figsize=(8, 5))
plt.plot(range(1, time_steps), cnn_errors, label='CNN Error', color='orange', marker='o')
plt.plot(range(1, time_steps), fno_errors, label='FNO Error', color='blue', marker='s')
plt.xlabel("Time Step (t)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Error Accumulation Over Time (Autoregressive Rollout)")
plt.legend()
plt.grid(True, linestyle='--')
plt.savefig("results/error_accumulation.png")
plt.close()

print("\nBenchmarks complete!")
print(f"Graphs saved to: results/speed_benchmark.png and results/error_accumulation.png")
print(f"Models saved to: cnn_model.pth and fno_model.pth")