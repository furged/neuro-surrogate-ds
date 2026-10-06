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

# Device setup (safe for both CUDA and CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"CONNECTED TO: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("CUDA not available, running on CPU")

# Configurable parameters (allows tuning via env vars or defaults)
dt = 0.1
time_steps = int(os.environ.get('TIME_STEPS', 100))
# Default 250 per coeff = 1,000 total (can be set to 10000 on HPC cluster)
train_sims_per_coeff = int(os.environ.get('SIMS_PER_COEFF', 250))
test_sims = int(os.environ.get('TEST_SIMS', 50))
epochs = int(os.environ.get('EPOCHS', 30))
batch_size = int(os.environ.get('BATCH_SIZE', 128))

os.makedirs("results", exist_ok=True)
checkpoint_path = "results/fno_checkpoint.pth"
data_cache_path = "results/sim_data_cache.pt"

# Batched physics solver
def generate_batch(num_sim, grid_size, diffusion_coeff, target_device='cpu'):
    # Compute on GPU if available for maximum speed, then store on target_device
    sim_device = device if torch.cuda.is_available() else torch.device('cpu')
    heat = torch.zeros((num_sim, 1, grid_size, grid_size), device=sim_device)
    for i in range(num_sim):
        num_spots = torch.randint(1, 4, (1,)).item()
        for _ in range(num_spots):
            x, y = torch.randint(0, grid_size, (2,)).tolist()
            heat[i, 0, x, y] = 100.0
            
    sims = torch.zeros((num_sim, time_steps, 1, grid_size, grid_size), device=sim_device)
    sims[:, 0, :, :, :] = heat
    state = heat.clone()
    lap_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=sim_device)
    for t in range(1, time_steps):
        lap = torch.nn.functional.conv2d(state, lap_kernel, padding=1)
        state = state + dt * (diffusion_coeff * lap)
        sims[:, t, :, :, :] = state
    return sims.to(target_device)

# Generate or load cached simulation data
if os.path.exists(data_cache_path):
    print(f"\nLoading cached dataset from {data_cache_path}...")
    cached_payload = torch.load(data_cache_path, map_location='cpu', weights_only=False)
    train_data_64 = cached_payload['train_data']
    test_data_64 = cached_payload['test_data']
    print(f"Loaded train shape: {train_data_64.shape}")
else:
    total_train = train_sims_per_coeff * 4
    print(f"\nGenerating {total_train:,} simulations ({train_sims_per_coeff:,} per coefficient)...")
    train_data_64 = torch.cat([
        generate_batch(train_sims_per_coeff, 64, 0.05),
        generate_batch(train_sims_per_coeff, 64, 0.10),
        generate_batch(train_sims_per_coeff, 64, 0.15),
        generate_batch(train_sims_per_coeff, 64, 0.20)
    ], dim=0)

    test_data_64 = generate_batch(test_sims, 64, 0.25)
    print(f"Train shape: {train_data_64.shape}")
    print(f"Caching generated dataset to {data_cache_path}...")
    torch.save({'train_data': train_data_64, 'test_data': test_data_64}, data_cache_path)

# Dataset class
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

pin_mem = torch.cuda.is_available()
train_loader = DataLoader(HeatDataset(train_data_64), batch_size=batch_size, shuffle=True, pin_memory=pin_mem)

# Scaled up FNO model
model = FNO(n_modes=(24, 24), hidden_channels=128, in_channels=1, out_channels=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Checkpoint resume logic
start_epoch = 0
loss_history = []
if os.path.exists(checkpoint_path):
    print(f"\nFound existing checkpoint at {checkpoint_path}! Resuming training...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    loss_history = ckpt.get('loss_history', [])
    print(f"Resuming from Epoch {start_epoch + 1}/{epochs} (Previous Loss: {ckpt.get('loss', 0.0):.6f})")

interrupted = False
if start_epoch >= epochs:
    print(f"\nTraining already completed ({epochs}/{epochs} epochs).")
else:
    print(f"\nTraining FNO on {len(train_data_64)} simulations (Epochs {start_epoch+1} to {epochs})...")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("Tip: You can pause anytime (Ctrl+C). A checkpoint is saved after every epoch!\n")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
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
            
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")
            
            # Save checkpoint after each epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history
            }, checkpoint_path)
            
    except KeyboardInterrupt:
        print("\n\n[PAUSED] Training interrupted by user. Checkpoint safely saved.")
        print(f"Run again anytime to resume from epoch {len(loss_history) + 1}!")
        interrupted = True

if not interrupted:
    # Resolution scaling test
    print("\nTesting FNO on HIGHER RESOLUTIONS (Zero-shot scaling)...")
    model.eval()
    resolutions = [64, 128, 256, 512]
    res_errors = []

    for res in resolutions:
        print(f"Testing {res}x{res}...")
        test_res = generate_batch(20, res, 0.25, target_device=device)
        errors = []
        with torch.no_grad():
            for i in range(20):
                state = test_res[i, 0, :, :, :].unsqueeze(0).to(device)
                gt = test_res[i, -1, :, :, :].cpu().numpy()
                for t in range(1, time_steps):
                    state = model(state)
                err = torch.nn.functional.mse_loss(state, torch.tensor(gt, device=device).unsqueeze(0)).item()
                errors.append(err)
        avg_err = np.mean(errors)
        res_errors.append(avg_err)
        print(f"Grid {res}x{res} | MSE: {avg_err:.6f}")

    # Save scaling results
    plt.figure(figsize=(10,6))
    plt.bar([f"{r}x{r}" for r in resolutions], res_errors, color='blue')
    plt.title("FNO Zero-Shot Resolution Scaling (Supercomputer Scale)")
    plt.ylabel("Mean Squared Error")
    plt.yscale('log')
    plt.grid(axis='y', linestyle='--')
    plt.savefig("results/super_resolution_scaling.png")
    plt.close()

    # Speed demo (512x512 in milliseconds)
    print("\nRunning final speed demonstration on 512x512 grid...")
    state_512 = generate_batch(1, 512, 0.25, target_device=device)
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
    plt.close()

    # Save final model
    torch.save(model.state_dict(), "results/fno_super_model.pth")

    print("\n SUPERCOMPUTER RUN COMPLETE!")
    print("Files saved to 'results/' folder:")
    print("   - super_resolution_scaling.png")
    print("   - 512_prediction_demo.png")
    print("   - fno_super_model.pth")
    print("   - fno_checkpoint.pth")
