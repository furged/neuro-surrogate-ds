import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from neuralop.models import FNO
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# --- 1. HPC SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 CONNECTED TO: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# --- 2. SUPERCOMPUTER PARAMETERS ---
dt = 0.1
time_steps = 100          # Predict 100 steps deep
train_sims_per_coeff = 10000  # 10,000 per coefficient = 40,000 total
test_sims = 500

# --- 3. BATCHED PHYSICS SOLVER ---
def generate_batch(num_sim, grid_size, diffusion_coeff):
    heat = torch.zeros((num_sim, 1, grid_size, grid_size), device=device)
    for i in range(num_sim):
        num_spots = torch.randint(1, 4, (1,)).item()
        for _ in range(num_spots):
            x, y = torch.randint(0, grid_size, (2,)).tolist()
            heat[i, 0, x, y] = 100.0
            
    sims = torch.zeros((num_sim, time_steps, 1, grid_size, grid_size), device=device)
    sims[:, 0, :, :, :] = heat
    state = heat.clone()
    lap_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=device)
    for t in range(1, time_steps):
        lap = torch.nn.functional.conv2d(state, lap_kernel, padding=1)
        state = state + dt * (diffusion_coeff * lap)
        sims[:, t, :, :, :] = state
    return sims

# --- 4. GENERATE MASSIVE DATA ---
print("\n⚙️ Generating 40,000+ simulations on the GPU...")
train_data_64 = torch.cat([
    generate_batch(train_sims_per_coeff, 64, 0.05),
    generate_batch(train_sims_per_coeff, 64, 0.10),
    generate_batch(train_sims_per_coeff, 64, 0.15),
    generate_batch(train_sims_per_coeff, 64, 0.20)
], dim=0)

test_data_64 = generate_batch(test_sims, 64, 0.25)
print(f"Train shape: {train_data_64.shape}")

# --- 5. DATASET CLASS ---
class HeatDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0] * self.data.shape[1] 
    def __getitem__(self, idx):
        si = idx // self.data.shape[1]
        ti = idx % self.data.shape[1]
        if ti >= self.data.shape[1] - 1: ti = self.data.shape[1] - 2
        x = self.data[si, ti, :, :, :] 
        y = self.data[si, ti+1, :, :, :] 
        return x, y

train_loader = DataLoader(HeatDataset(train_data_64), batch_size=128, shuffle=True)

# --- 6. SCALED-UP FNO MODEL ---
model = FNO(n_modes=(24, 24), hidden_channels=128, in_channels=1, out_channels=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n🔥 Training FNO on {len(train_data_64)} simulations...")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

epochs = 30
for epoch in range(epochs):
    epoch_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/len(train_loader):.6f}")

# --- 7. RESOLUTION SCALING TEST ---
print("\n🧪 Testing FNO on HIGHER RESOLUTIONS (Zero-shot scaling)...")
model.eval()
resolutions = [64, 128, 256, 512]
res_errors = []

for res in resolutions:
    print(f"Testing {res}x{res}...")
    test_res = generate_batch(20, res, 0.25)
    errors = []
    with torch.no_grad():
        for i in range(20):
            state = test_res[i, 0, :, :, :].unsqueeze(0).to(device)
            gt = test_res[i, -1, :, :, :].cpu().numpy()
            for t in range(1, time_steps):
                state = model(state)
            err = torch.nn.functional.mse_loss(state, torch.tensor(gt).unsqueeze(0).to(device)).item()
            errors.append(err)
    avg_err = np.mean(errors)
    res_errors.append(avg_err)
    print(f"Grid {res}x{res} | MSE: {avg_err:.6f}")

# --- 8. SAVE SCALING RESULTS ---
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(10,6))
plt.bar([f"{r}x{r}" for r in resolutions], res_errors, color='blue')
plt.title("FNO Zero-Shot Resolution Scaling (Supercomputer Scale)")
plt.ylabel("Mean Squared Error")
plt.yscale('log')
plt.grid(axis='y', linestyle='--')
plt.savefig("results/super_resolution_scaling.png")

# --- 9. SPEED DEMO (512x512 in milliseconds) ---
print("\n⚡ Running final speed demonstration on 512x512 grid...")
state_512 = generate_batch(1, 512, 0.25)
start_time = time.time()
with torch.no_grad():
    pred_512 = model(state_512[:, 0, :, :, :])
elapsed = time.time() - start_time
print(f"FNO predicted a 512x512 future heat map in {elapsed*1000:.2f} milliseconds!")

plt.figure(figsize=(8,6))
plt.imshow(pred_512[0, 0].cpu().numpy(), cmap='hot')
plt.title(f"512x512 Prediction (took {elapsed*1000:.2f} ms)")
plt.colorbar()
plt.savefig("results/512_prediction_demo.png")

# Save the model
torch.save(model.state_dict(), "results/fno_super_model.pth")

print("\n✅ SUPERCOMPUTER RUN COMPLETE!")
print("Files saved to 'results/' folder:")
print("   - super_resolution_scaling.png")
print("   - 512_prediction_demo.png")
print("   - fno_super_model.pth")