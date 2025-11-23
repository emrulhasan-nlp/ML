import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(29)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Synthetic Data for training and validation
X = torch.randn(1000, 20)
y = torch.randn(1000, 1)
dataset=TensorDataset(X, y)

# split data into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define a simple Deep Neural Network
class SimpleDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleDNN, self).__init__()

        self.input_layer=nn.Linear(input_dim, hidden_dim)
        self.hidden_layer=nn.Linear(hidden_dim, hidden_dim)
        self.output_layer=nn.Linear(hidden_dim, output_dim)
        #self.layers=nn.ModuleList([input_layer, hidden_layer, output_layer])
        self.activation=nn.ReLU()
        self.dropout=nn.Dropout(p=0.5)


    def forward(self, x):
        x=self.input_layer(x)
        x=self.activation(x)
        x=self.dropout(x)
        x=self.hidden_layer(x)  
        x=self.activation(x)
        x=self.dropout(x)
        x=self.output_layer(x)
        return x
        
# Instantiate the model, define loss function and optimizer
model=SimpleDNN(input_dim=20, hidden_dim=64, output_dim=1).to(device)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=0.001)


best_val_loss = float("inf")
save_path = "outputs/dnn_best_model.pth"
os.makedirs("outputs", exist_ok=True)

# Training loop
num_epochs=50
for epoch in range(num_epochs):
    model.train()
    training_loss=0.0
    for inputs, targets in train_loader:
        inputs, targets=inputs.to(device), targets.to(device)
        #print(inputs.shape, targets.shape)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        training_loss += loss.item() * inputs.size(0)
    
    avg_training_loss=training_loss / len(train_loader.dataset)
    # Validation loop
    model.eval()
    validation_loss=0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets=inputs.to(device), targets.to(device)
            outputs=model(inputs)
            loss=criterion(outputs, targets)
            validation_loss += loss.item() * inputs.size(0)
    avg_validation_loss=validation_loss / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}],Training Loss: {avg_training_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}")
    
    # Saving the best model based on validation loss can be added here
    if avg_validation_loss < best_val_loss:
        best_val_loss = avg_validation_loss
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")


# Load the best model and test it on a new test set
print("Testing the best saved model on new test data...")
model.load_state_dict(torch.load(save_path))
model.eval()

#Test the model on a new synthetic test set
X_test = torch.randn(200, 20).to(device)
y_test = torch.randn(200, 1).to(device) 

with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f"Test MSE Loss: {test_loss.item():.4f}")
