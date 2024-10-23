# %%
# Load preprocessed data
train_df = pd.read_csv("train_data_final.csv")
test_df = pd.read_csv("test_data_final.csv")

# %%
# Inspect the first few rows
print("Training Data:")
print(train_df.head())

print("\nTest Data:")
print(test_df.head())

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import ast  # For safely evaluating strings containing Python literals
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report



def index_to_one_hot(index, vocab_size):
    """ Convert an index to a one-hot encoded vector """
    one_hot = torch.zeros(vocab_size)
    one_hot[index] = 1
    return one_hot

class SentimentDataset(Dataset):
    """ Custom dataset class for sentiment analysis data """
    def __init__(self, dataframe):
        """
        Args:
            dataframe (pd.DataFrame): Dataframe containing 'padded_review' and 'label'.
        """
        self.reviews = dataframe['padded_review'].apply(ast.literal_eval)  # Ensure it's a list
        self.labels = dataframe['label']

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        """
        Returns:
            X (torch.Tensor): The one-hot encoded review.
            y (int): The label.
        """
        indices = self.reviews.iloc[idx]
        # Convert all indices in the review to one-hot vectors and stack them
        one_hot_encoded = torch.stack([index_to_one_hot(index, vocab_size) for index in indices])
        return one_hot_encoded, self.labels.iloc[idx]

# train_data = pd.read_csv("train_data_final.csv")
# # Create dataset and DataLoader
# dataset = SentimentDataset(train_data)
# data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# # Printing one batch
# for X_batch, y_batch in data_loader:
#     print("Batch X shape:", X_batch.shape)  # Shape of the batch
#     print("Batch Y shape:", y_batch.shape)  # Shape of the labels
#     print("Batch X content:\n", X_batch)    # Content of one-hot encoded vectors
#     print("Batch Y content:", y_batch)      # Content of labels
#     break  # Break after printing the first batch to avoid printing multiple batches
# Load your dataset (Make sure to replace 'your_dataset.csv' with the actual file path)

train_df = pd.read_csv('train_data_final.csv')
train_df = train_df.sample(n=1000, random_state=42)  # Sampling 1000 examples due to dataset size

test_df = pd.read_csv('train_data_final.csv')
test_df = test_df.sample(n=200, random_state=42)  # Sampling 1000 examples due to dataset size


# Splitting data into train, dev, and test sets
# train_data, test_data = train_test_split(test_df, test_size=0.2, random_state=42)
train_data, dev_data = train_test_split(train_df, test_size=0.1, random_state=42)

test_data = test_df

# Create datasets
train_dataset = SentimentDataset(train_data)
dev_dataset = SentimentDataset(dev_data)
test_dataset = SentimentDataset(test_data)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Print data loader counts
print(f"Number of batches in training loader: {len(train_loader)}")
print(f"Number of batches in dev loader: {len(dev_loader)}")
print(f"Number of batches in test loader: {len(test_loader)}")


# %%
# Feed-Forward Neural Network Definition
class FFN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor for FFN input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model
input_size = vocab_size
hidden1 = 256
hidden2 = 128
output_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FFN(input_size, hidden1, hidden2, output_size)
model = model.to(device)

# %% 

from tqdm import tqdm

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
best_dev_accuracy = 0.0
best_model_path = 'best_ffn_model.pth'

for epoch in range(num_epochs):
    model.train()
    train_losses, train_correct, train_total = [], 0, 0

    for reviews, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
        reviews = reviews.to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(reviews)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        predicted = (outputs > 0.5).float()
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_accuracy = train_correct / train_total
    print(f'Train Loss: {np.mean(train_losses):.4f}, Train Acc: {train_accuracy * 100:.2f}%')

    # Validation phase
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for reviews, labels in tqdm(dev_loader, desc=f'Validation Epoch {epoch+1}'):
            reviews = reviews.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(reviews)
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    print(f'Validation Acc: {val_accuracy * 100:.2f}%')
    if val_accuracy > best_dev_accuracy:
        best_dev_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print('Saved Best Model')

# Load and evaluate the best model
best_model = FFN(vocab_size, 256, 128, 1)
best_model.load_state_dict(torch.load(best_model_path))
best_model = best_model.to(device)
best_model.eval()

# Test evaluation
test_correct, test_total = 0, 0
with torch.no_grad():
    for reviews, labels in test_loader:
        reviews = reviews.to(device)
        labels = labels.to(device).view(-1, 1)
        outputs = best_model(reviews)
        predicted = (outputs > 0.5).float()
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')