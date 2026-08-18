import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# rerun the data generation
# we need the data in memory for this step
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
grid_size = 64
num_simulations = 10
diffusion_coefficient = 0.1
dt = 0.1
time_steps = 20

# generate the data again
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

print(f"Raw data shape: {all_simulations.shape}")

# create the dataset
# a Dataset tells PyTorch how to get each training sample
class HeatEquationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        # total number of training examples
        # 10 simulations x 20 time steps = 200 examples
        return self.data.shape[0] * self.data.shape[1] 
    
    def __getitem__(self, idx):
        # convert the flat index into a simulation index and time step
        sim_idx = idx // self.data.shape[1]  # Which simulation (0 to 9)
        t_idx = idx % self.data.shape[1]     # Which time step (0 to 19)
        
        # input is the heat at the current time step
        x = self.data[sim_idx, t_idx, :, :, :] 
        
        # target is the heat at the next time step
        # for the last time step, we use the same state as the target
        if t_idx < self.data.shape[1] - 1:
            y = self.data[sim_idx, t_idx + 1, :, :, :]
        else:
            y = self.data[sim_idx, t_idx, :, :, :] 
            
        return x, y

# create the dataloader
# it shuffles the data and groups the samples into batches
full_dataset = HeatEquationDataset(all_simulations)
dataloader = DataLoader(full_dataset, batch_size=16, shuffle=True)

print(f"Total training samples: {len(full_dataset)}")
print(f"Batch size: 16, Number of batches: {len(dataloader)}")

# test the dataloader
# grab one batch and check its shape
for batch_x, batch_y in dataloader:
    print(f"\nInput batch shape: {batch_x.shape}")   # Should be (16, 1, 64, 64)
    print(f"Target batch shape: {batch_y.shape}")   # Should be (16, 1, 64, 64)
    
    # visualize the first sample
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(batch_x[0, 0].cpu().numpy(), cmap='hot')
    plt.title("Input (t)")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(batch_y[0, 0].cpu().numpy(), cmap='hot')
    plt.title("Target (t+1)")
    plt.colorbar()
    plt.show()
    break  # We only test the first batch
