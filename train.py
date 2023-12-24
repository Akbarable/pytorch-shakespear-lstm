import torch
from torch.utils.data import DataLoader
from utils.dataset import ShakespeareDataset
from models.LSTM_model import LSTMModel
import matplotlib.pyplot as plt

#dataset
dataset = ShakespeareDataset("data/input.txt", sequence_length=30)

# set hyperparameters
input_size = len(dataset.vocab)  # Use the size of your vocabulary
hidden_size = 256
output_size = len(dataset.vocab)  # Use the size of your vocabulary
num_layers = 2
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# dataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store training loss for each epoch
train_loss_history = []

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = len(dataloader)
    
    for i, (batch_sequence, batch_target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(batch_sequence)
        loss = criterion(output, batch_target.long())  # Convert to torch.long
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

        # Print progress every 100 batches
        if (i + 1) % 100 == 0 or (i + 1) == num_batches:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{num_batches}], Loss: {loss.item()}')

    # Average loss for the epoch
    avg_epoch_loss = epoch_loss / num_batches
    train_loss_history.append(avg_epoch_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss}')

# Plotting the training loss over epochs
plt.plot(train_loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'shakespeare_lstm_model.pth')
