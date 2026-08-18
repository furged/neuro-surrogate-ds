import torch
import matplotlib.pyplot as plt

# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# parameters
grid_size = 64          # 64x64 grid
num_simulations = 10    # Reduced to 10 for faster CPU runs
diffusion_coefficient = 0.1  
dt = 0.1                
time_steps = 20         # Reduced to 20 time steps

# create the initial conditions
initial_heat = torch.zeros((num_simulations, 1, grid_size, grid_size), device=device)

for i in range(num_simulations):
    num_spots = torch.randint(1, 4, (1,)).item()
    for _ in range(num_spots):
        x, y = torch.randint(0, grid_size, (2,)).tolist()
        initial_heat[i, 0, x, y] = 100.0

# run the physics solver
all_simulations = torch.zeros((num_simulations, time_steps, 1, grid_size, grid_size), device=device)
all_simulations[:, 0, :, :, :] = initial_heat

laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=device)

current_state = initial_heat.clone()

for t in range(1, time_steps):
    laplacian = torch.nn.functional.conv2d(current_state, laplacian_kernel, padding=1)
    current_state = current_state + dt * (diffusion_coefficient * laplacian)
    all_simulations[:, t, :, :, :] = current_state

print(f"Dataset generated! Shape: {all_simulations.shape}")

# visualize the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(all_simulations[0, 0, 0].cpu().numpy(), cmap='hot')
plt.title("Initial Heat (t=0)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(all_simulations[0, -1, 0].cpu().numpy(), cmap='hot')
plt.title("Heat after 20 steps (t=20)")
plt.colorbar()

plt.show()
