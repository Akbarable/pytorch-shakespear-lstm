import torch
from torch.utils.data import DataLoader
from utils.dataset import ShakespeareDataset
from models.LSTM_model import LSTMModel

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

# Training loop
for epoch in range(num_epochs):
    for batch_sequence, batch_target in dataloader:
        optimizer.zero_grad()
        output = model(batch_sequence)
        loss = criterion(output, batch_target.long())  # Convert to torch.long
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


# Save the trained model
torch.save(model.state_dict(), 'shakespeare_lstm_model.pth')
