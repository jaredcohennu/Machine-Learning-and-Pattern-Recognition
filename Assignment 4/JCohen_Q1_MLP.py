import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import math


np.random.seed(42)
torch.manual_seed(42)

# Parameters
r = np.array([2, 4])
sigma = 1
mean = np.array([0, 0])
class_labels = [-1, 1]
priors = [0.5, 0.5]


#function to generate the dataset
def generate_dataset(n_samples):
    labels = np.random.choice(class_labels, size=n_samples, p=priors)
    radii = np.where(labels == -1, r[0], r[1])
    thetas = np.random.uniform(-math.pi, math.pi, n_samples)
    noise = np.random.multivariate_normal(mean, sigma**2 * np.eye(2), n_samples)
    
    # create the samples = r * [cos(theta), sin(theta)] + noise
    x = np.column_stack([
        radii * np.cos(thetas),
        radii * np.sin(thetas)
    ]) + noise
    
    # convert labels for pytorch compatability
    labels_binary = np.where(labels == -1, 0, 1)
    
    return x, labels_binary

#Create a custom pytorch class for the circular data
class CircularDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#Create custom class for two layer  MLP with the following:
# 2 layer input
#Hidden layer with n perceptrons and activation
#Outut layer with 2 units and softmax
class MLP(nn.Module):
    def __init__(self, n_perceptrons, activation='tanh'):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(2, n_perceptrons)
        self.fc2 = nn.Linear(n_perceptrons, 2)
        
        # Activation function selection so I can try out a bunch of different ones
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.n_perceptrons = n_perceptrons
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x  # CrossEntropyLoss includes softmax
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

#Function to train the model for one epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

#helper function to evaluate the model on a dataset
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

#function to perform k fold validation given an MLP and n number of perceptrons
def cross_validate_mlp(x_train, y_train, n_perceptrons, n_folds=10, 
                       n_epochs=1000, batch_size=64, lr=0.001, activation='tanh'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = CircularDataset(x_train, y_train)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(x_train)):
        # create the necessary dataloaders
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        # Initialize model and train for epochs
        model = MLP(n_perceptrons, activation=activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(n_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # evaluate the model
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        fold_accuracies.append(val_acc)
    
    avg_accuracy = np.mean(fold_accuracies)
    return avg_accuracy, fold_accuracies

#perform the selection to find the optimal number of perceptrons
def model_selection(x_train, y_train, perceptron_range, n_folds=10, 
                    n_epochs=1000, activation='tanh'):
    
    results = []
    
    for n_perceptrons in perceptron_range:
        avg_acc, fold_accs = cross_validate_mlp(
            x_train, y_train, n_perceptrons, 
            n_folds=n_folds, n_epochs=n_epochs, activation=activation
        )
        
        results.append({
            'n_perceptrons': n_perceptrons,
            'avg_accuracy': avg_acc,
            'fold_accuracies': fold_accs,
            'std_accuracy': np.std(fold_accs)
        })
        
        print(f"  Average accuracy: {avg_acc*100:.2f}% (±{np.std(fold_accs)*100:.2f}%)")
    
    # Find best model
    best_result = max(results, key=lambda x: x['avg_accuracy'])
    
    return results, best_result

#Functiion to plot the model results
def plot_model_selection(results):
    n_perceptrons = [r['n_perceptrons'] for r in results]
    avg_accuracies = [r['avg_accuracy'] for r in results]
    std_accuracies = [r['std_accuracy'] for r in results]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.errorbar(n_perceptrons, avg_accuracies, yerr=std_accuracies, 
                marker='o', capsize=5, capthick=2, linewidth=2)
    
    # Mark best model
    best_idx = np.argmax(avg_accuracies)
    ax.plot(n_perceptrons[best_idx], avg_accuracies[best_idx], 
            'r*', markersize=20, 
            label=f'Best: n={n_perceptrons[best_idx]}, '
                  f'Accuracy={avg_accuracies[best_idx]*100:.2f}%')
    
    ax.set_xlabel('Number of Perceptrons', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('MLP Model Selection: Number of Perceptrons vs Accuracy', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('mlp_model_selection.png', dpi=150)
    plt.show()

# Function to train the final model on the full training set
def train_final_mlp(x_train, y_train, n_perceptrons, n_epochs=1000, 
                    batch_size=64, lr=0.001, activation='tanh'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = CircularDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MLP(n_perceptrons, activation=activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, dataloader, criterion, optimizer, device)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Loss={train_loss:.4f}, Accuracy={train_acc*100:.2f}%")
    
    return model

#Helper function to plot the decision boundary for the MLP
def plot_decision_boundary(model, x_test, y_test, title='MLP Decision Boundary'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
    y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    # Get the predictions for the mesh
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    
    with torch.no_grad():
        outputs = model(grid_points)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    
    Z = probs.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    contour = ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')
    y_test_original = np.where(y_test == 0, -1, 1)  # Convert back to -1, 1
    for label in [-1, 1]:
        mask = (y_test_original == label)
        ax.scatter(x_test[mask, 0], x_test[mask, 1], 
                  label=f'Class {label}', s=1, alpha=0.3, edgecolors='k')
    
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('P(Class = +1)', fontsize=12)
    
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlp_decision_boundary.png', dpi=150)
    plt.show()

#Evaluates the model on the test set
def evaluate_test_set(model, x_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = CircularDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512)
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    
    # Per-class accuracy
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_test).to(device)
        outputs = model(x_tensor)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()
    
    for class_idx in [0, 1]:
        mask = (y_test == class_idx)
        class_acc = np.mean(predictions[mask] == y_test[mask])
        class_label = -1 if class_idx == 0 else 1
        print(f"Class {class_label} Accuracy: {class_acc*100:.2f}%")
    
    print(f"\nOverall Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Error Rate: {(1-test_accuracy)*100:.2f}%")
    
    return test_accuracy


def main():
 
    x_train, y_train = generate_dataset(1000)
    x_test, y_test = generate_dataset(10000)
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    perceptron_range = [1, 2, 3, 4, 5, 7, 9, 13, 19, 27, 38, 55, 79, 113]
    
    results, best_result = model_selection(
        x_train, y_train, perceptron_range, 
        n_folds=10, n_epochs=1000, activation='tanh'
    )
    
    plot_model_selection(results)
    
    best_n_perceptrons = best_result['n_perceptrons']
    final_model = train_final_mlp(
        x_train, y_train, best_n_perceptrons, 
        n_epochs=1000, activation='tanh'
    )
    
    test_accuracy = evaluate_test_set(final_model, x_test, y_test)
    
    plot_decision_boundary(
        final_model, x_test, y_test,
        f'MLP Decision Boundary (n={best_n_perceptrons}, Test Accuracy: {test_accuracy*100:.2f}%)'
    )


if __name__ == "__main__":
    main()