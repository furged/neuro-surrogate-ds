import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

# setup and data generation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

grid_size = 64
num_simulations = 10
diffusion_coefficient = 0.1
dt = 0.1
time_steps = 20

# generate the data
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

# prepare the dataset
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

# create the dataloader
full_dataset = HeatEquationDataset(all_simulations)
# use a batch size of 16
dataloader = DataLoader(full_dataset, batch_size=16, shuffle=True)

# define the CNN model
class HeatCNN(nn.Module):
    def __init__(self):
        super(HeatCNN, self).__init__()
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 64x64 -> 64x64
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), # 64x64 -> 64x64
            nn.ReLU(),
        )
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)   # Output 1 channel (heat map)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# initialize the model, loss function, and optimizer
model = HeatCNN().to(device)
criterion = nn.MSELoss()  # Mean Squared Error - standard for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

# training loop
num_epochs = 50  # We'll train for 50 rounds
loss_history = []

print("\nStarting training...")
start_time = time.time()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    
    for batch_x, batch_y in dataloader:
        # move the data to the device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # forward pass
        predictions = model(batch_x)
        
        # calculate the loss
        loss = criterion(predictions, batch_y)
        
        # backward pass and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds!")

# visualize the result
# pick one example and see what the model predicts
model.eval()  # Put model in evaluation mode
with torch.no_grad():
    # get a sample
    sample_input, sample_target = full_dataset[0]
    sample_input = sample_input.unsqueeze(0).to(device) # Add batch dimension (1, 1, 64, 64)
    
    # make a prediction
    sample_prediction = model(sample_input)
    
    # move the results back to CPU
    sample_input = sample_input.cpu().squeeze().numpy()
    sample_target = sample_target.cpu().squeeze().numpy()
    sample_prediction = sample_prediction.cpu().squeeze().numpy()

# plot the results
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(sample_input, cmap='hot')
plt.title("Input (t)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(sample_target, cmap='hot')
plt.title("True Target (t+1)")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(sample_prediction, cmap='hot')
plt.title("CNN Prediction (t+1)")
plt.colorbar()

plt.suptitle(f"Final Training Loss: {loss_history[-1]:.6f}")
plt.tight_layout()
plt.show()

# plot the loss
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()
