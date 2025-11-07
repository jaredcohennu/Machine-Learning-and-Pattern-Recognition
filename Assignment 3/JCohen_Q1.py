import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# Set rng seeds
np.random.seed(42)
torch.manual_seed(42)

# Define and generate the data
C = 4
priors = np.array([0.25, 0.25, 0.25, 0.25])
mean1 = np.array([2.34, 2.34, 0.0])
mean2 = np.array([2.34, -2.34, 0.0])
mean3 = np.array([-2.34, 2.34, 0.0])
mean4 = np.array([-2.34, -2.34, 0.0])
means = [mean1, mean2, mean3, mean4]
cov_scale = 3.0
covariances = [cov_scale * np.eye(3) for _ in range(C)]
distributions = [multivariate_normal(means[i], covariances[i]) for i in range(C)]

#Function to help generate the data given the means, cov, priors, and number of samples
def generate_data(means, covariances, priors, n_samples):
    n_classes = len(priors)
    dim = len(means[0])
    X = np.zeros((n_samples, dim))
    y = np.zeros(n_samples, dtype=int)
    
    # Generate the labels
    cumulative_priors = np.cumsum(priors)
    random_vals = np.random.rand(n_samples)
    
    #Loop through and determine class and generate the sample
    for i in range(n_samples):
        class_idx = np.searchsorted(cumulative_priors, random_vals[i])
        y[i] = class_idx
        X[i] = np.random.multivariate_normal(means[class_idx], covariances[class_idx])
    
    return X, y

# Generate the different training and test datasets
training_sizes = [100, 500, 1000, 5000, 10000]
test_size = 100000
training_datasets = []
training_labels = []

for size in training_sizes:
    X_train, y_train = generate_data(means, covariances, priors, size)
    training_datasets.append(X_train)
    training_labels.append(y_train)

X_test, y_test = generate_data(means, covariances, priors, test_size)

def plot_training_data(training_datasets, training_labels, training_sizes):
    """Plot all training datasets in 3D subplots"""
    n_datasets = len(training_datasets)
    n_cols = 2
    n_rows = (n_datasets + 1) // 2
    
    fig = plt.figure(figsize=(14, 5 * n_rows))
    
    for idx, (X, y, size) in enumerate(zip(training_datasets, training_labels, training_sizes)):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        # Plot each class with different color
        colors = ['red', 'blue', 'green', 'orange']
        for class_idx in range(4):
            mask = y == class_idx
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], 
                      c=colors[class_idx], label=f'Class {class_idx}', 
                      alpha=0.6, s=20)
        
        ax.set_xlabel('X₁', fontsize=10)
        ax.set_ylabel('X₂', fontsize=10)
        ax.set_zlabel('X₃', fontsize=10)
        ax.set_title(f'Training Data (N={size} samples)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_data_2d_projections(training_datasets, training_labels, training_sizes):
    """Plot 2D projections (X1-X2, X1-X3, X2-X3) for each training set"""
    n_datasets = len(training_datasets)
    
    fig, axes = plt.subplots(n_datasets, 3, figsize=(15, 3 * n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['red', 'blue', 'green', 'orange']
    projections = [('X₁', 'X₂', 0, 1), ('X₁', 'X₃', 0, 2), ('X₂', 'X₃', 1, 2)]
    
    for row_idx, (X, y, size) in enumerate(zip(training_datasets, training_labels, training_sizes)):
        for col_idx, (xlabel, ylabel, dim1, dim2) in enumerate(projections):
            ax = axes[row_idx, col_idx]
            
            # Plot each class
            for class_idx in range(4):
                mask = y == class_idx
                ax.scatter(X[mask, dim1], X[mask, dim2], 
                          c=colors[class_idx], label=f'Class {class_idx}',
                          alpha=0.6, s=15)
            
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if col_idx == 0:
                ax.set_ylabel(f'N={size}\n{ylabel}', fontsize=10, fontweight='bold')
            
            if row_idx == 0:
                ax.set_title(f'{xlabel} vs {ylabel}', fontsize=11, fontweight='bold')
            
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()

# Plot the data and projections
plot_training_data(training_datasets, training_labels, training_sizes)
plot_data_2d_projections(training_datasets, training_labels, training_sizes)

#Function to get the theoretical best MAP classifier using the true distribution
def theoretical_classifier(X, distributions, priors):
    n_samples = X.shape[0]
    n_classes = len(distributions)
    predictions = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        posteriors = np.zeros(n_classes)
        for c in range(n_classes):
            posteriors[c] = distributions[c].pdf(X[i]) * priors[c]
        predictions[i] = np.argmax(posteriors)
    
    return predictions

# Evaluate theoretical optimal classifier
y_pred_optimal = theoretical_classifier(X_test, distributions, priors)
optimal_accuracy = np.mean(y_pred_optimal == y_test)
optimal_error = 1 - optimal_accuracy
print(f"Theoretical Optimal Classifier Performance:")
print(f"  Accuracy: {optimal_accuracy:.4f}")
print(f"  Probability of Error: {optimal_error:.4f} ({optimal_error*100:.2f}%)")

#Define a class for the MLP class which inherits from nn.Module
class MLP(nn.Module):
    
    def __init__(self, input_dim, n_classes, n_perceptrons):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_perceptrons) #Create the first layer
        self.activation = nn.ELU()  # ELU activation
        self.fc2 = nn.Linear(n_perceptrons, n_classes) #Create the second layer
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x  # No softmax neded since CrossEntropyLoss includes it
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

#Create a custom dataset so that pytorch can use it
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# train epoch function which is used to train a model for one epoch
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

#Function to evaluate a model on a dataset
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss


#Perform the 10-fold cross validation with model order selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
n_folds = 10
n_epochs = 250
batch_size = 32
learning_rate = 0.001
perceptron_candidates = [2, 4, 8, 16, 32]

criterion = nn.CrossEntropyLoss()
results = {}

#For each training dataset size iterate through and test each perceptron count
for train_idx, train_size in enumerate(training_sizes):
    print(f"Training Size: {train_size} samples ({train_idx+1}/{len(training_sizes)})")
    
    X_train = training_datasets[train_idx]
    y_train = training_labels[train_idx]
    
    dataset = CustomDataset(X_train, y_train)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    best_n_perceptrons = None
    best_cv_error = float('inf')
    cv_errors = []
    
    # Iterate through the perceptron counts
    for n_perceptrons in perceptron_candidates:
        print(f"\n  Testing {n_perceptrons} perceptrons...")
        
        fold_accuracies = []
        
        # K-fold cross-validation
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            # Create data loaders for this fold
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
            
            # Initialize model
            model = MLP(input_dim=3, n_classes=C, n_perceptrons=n_perceptrons)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train each model for the number of eopochs
            for epoch in range(n_epochs):
                train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Evaluate on validation set
            val_accuracy, _ = evaluate(model, val_loader, criterion, device)
            fold_accuracies.append(val_accuracy)
        
        # Average across folds
        mean_cv_accuracy = np.mean(fold_accuracies)
        mean_cv_error = 1 - mean_cv_accuracy
        cv_errors.append(mean_cv_error)
        
        print(f"    CV Error: {mean_cv_error:.4f} (Accuracy: {mean_cv_accuracy:.4f})")
        
        # Track best model
        if mean_cv_error < best_cv_error:
            best_cv_error = mean_cv_error
            best_n_perceptrons = n_perceptrons
    
    print(f"\n  Best number of perceptrons: {best_n_perceptrons}")
    print(f"  Best CV error: {best_cv_error:.4f}")
    
    # Store results
    results[train_size] = {
        'perceptron_counts': perceptron_candidates,
        'cv_errors': cv_errors,
        'best_n_perceptrons': best_n_perceptrons,
        'best_cv_error': best_cv_error
    }

#Train the final models with the best perceptron counts
print("TRAINING FINAL MODELS WITH BEST PERCEPTRON COUNTS")

final_models = {}
test_errors = []
test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

for train_idx, train_size in enumerate(training_sizes):
    X_train = training_datasets[train_idx]
    y_train = training_labels[train_idx]
    
    # Select the best number of perceptrons for each model and test it on the test dataset
    best_n_perceptrons = results[train_size]['best_n_perceptrons']
    
    print(f"\nTraining final model for size {train_size} with {best_n_perceptrons} perceptrons...")
    
    dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train multiple times and keep best (to avoid local optima)
    best_model = None
    best_train_accuracy = 0
    n_restarts = 5
    
    for restart in range(n_restarts):
        model = MLP(input_dim=3, n_classes=C, n_perceptrons=best_n_perceptrons)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(n_epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
        
        train_accuracy, _ = evaluate(model, train_loader, criterion, device)
        
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            best_model = model
    
    # Evaluate on test set
    test_accuracy, test_loss = evaluate(best_model, test_loader, criterion, device)
    test_error = 1 - test_accuracy
    test_errors.append(test_error)
    
    final_models[train_size] = best_model
    
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Error: {test_error:.4f}")

#Report results and graph them

#Get optimal error on test set
y_pred_optimal_test = theoretical_classifier(X_test, distributions, priors)
optimal_test_accuracy = np.mean(y_pred_optimal_test == y_test)
optimal_test_error = 1 - optimal_test_accuracy
print(f"\nTheoretical Optimal Classifier (on {test_size} test samples):")
print(f"  Accuracy: {optimal_test_accuracy:.6f}")
print(f"  Error:    {optimal_test_error:.6f} ({optimal_test_error*100:.3f}%)")

# Error vs Number of Perceptrons for each training size
plt.figure(figsize=(12, 8))
for train_size in training_sizes:
    perceptron_counts = results[train_size]['perceptron_counts']
    cv_errors = results[train_size]['cv_errors']
    plt.plot(perceptron_counts, cv_errors, marker='o', label=f'{train_size} samples')

plt.axhline(y=optimal_test_error, color='r', linestyle='--', linewidth=2, label='Theoretical Optimal')
#plt.xscale('log')
plt.xlabel('Number of Perceptrons', fontsize=12)
plt.ylabel('Probability of Error', fontsize=12)
plt.title('CV Error vs Number of Perceptrons', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Test Error vs Training Size
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, test_errors, marker='o', markersize=10, linewidth=2, label='MLP Test Error')
plt.axhline(y=optimal_error, color='r', linestyle='--', linewidth=2, label='Theoretical Optimal')
plt.xscale('log')
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Probability of Error', fontsize=12)
plt.title('Test Set Performance vs Training Set Size', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Print a final summary

print("FINAL SUMMARY")

print(f"\nTheoretical Optimal Classifier:")
print(f"  Probability of Error: {optimal_error:.4f} ({optimal_error*100:.2f}%)")

print(f"\nMLP Classifier Results:")
for i, train_size in enumerate(training_sizes):
    best_n = results[train_size]['best_n_perceptrons']
    test_err = test_errors[i]
    print(f"  Training Size {train_size:5d}: {best_n:4d} perceptrons, Test Error: {test_err:.4f} ({test_err*100:.2f}%)")
