import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold
import math


np.random.seed(42)

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
    
    return x, labels

#Helper function to plot the dataset
def plot_dataset(x_train, y_train, x_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for (x, y, title), ax in zip([
        (x_train, y_train, f'Training Data (N={len(x_train)})'),
        (x_test, y_test, f'Test Data (N={len(x_test)})')
    ], axes):
        for label in class_labels:
            mask = (y == label)
            ax.scatter(x[mask, 0], x[mask, 1], label=f'Class {label}', 
                      s=1, alpha=0.6)
        
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('dataset_visualization.png', dpi=150)
    plt.show()

#Function that performs the grid search with k-fold CV to find the optimal parameters
def svm_grid_search_cv(x_train, y_train, k_folds=10):

    #Grid setup
    c_values = np.logspace(-2, 2, 50)
    kernel_width_values = np.logspace(-2, 2, 50)
    gamma_values = 1 / (2 * kernel_width_values**2)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # results storage
    grid_results = np.zeros((len(gamma_values), len(c_values)))
    best_accuracy = 0
    best_c = None
    best_gamma = None
    
    iteration = 0
    total_iterations = len(c_values) * len(gamma_values) * k_folds
    
    for i, gamma in enumerate(gamma_values):
        for j, c in enumerate(c_values):
            fold_accuracies = []
            
            # Perform CV
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(x_train)):
                x_fold_train, x_fold_val = x_train[train_idx], x_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Train the SVM and evaluate the results
                clf = svm.SVC(kernel='rbf', gamma=gamma, C=c)
                clf.fit(x_fold_train, y_fold_train)
                accuracy = clf.score(x_fold_val, y_fold_val)
                fold_accuracies.append(accuracy)
                
                iteration += 1
                if iteration % 500 == 0:
                    print(f"Progress: {iteration}/{total_iterations} "
                          f"({100*iteration/total_iterations:.1f}%)")
            
            avg_accuracy = np.mean(fold_accuracies)
            grid_results[i, j] = avg_accuracy
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_c = c
                best_gamma = gamma
    
    best_kernel_width = np.sqrt(1 / (2 * best_gamma))
    
    print(f"\nGrid Search Complete!")
    print(f"Best Parameters:")
    print(f"  C (Box Constraint): {best_c:.6f}")
    print(f"  Kernel Width (σ): {best_kernel_width:.6f}")
    print(f"  Validation Accuracy: {best_accuracy*100:.2f}%")
    
    best_params = {
        'C': best_c,
        'gamma': best_gamma,
        'kernel_width': best_kernel_width,
        'validation_accuracy': best_accuracy
    }
    
    return best_params, grid_results, c_values, kernel_width_values


#Helper function to plot the results of the grid search
def plot_grid_search_results(grid_results, c_values, kernel_width_values, best_params):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    contour = ax.contourf(c_values, kernel_width_values, grid_results, 
                          levels=20, cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Validation Accuracy', fontsize=12)
    
    # Mark best parameters
    ax.plot(best_params['C'], best_params['kernel_width'], 
            'r*', markersize=20, 
            label=f"Best: C={best_params['C']:.2f}, "
                  f"σ={best_params['kernel_width']:.2f}\n"
                  f"Accuracy={best_params['validation_accuracy']*100:.2f}%")
    
    ax.set_xlabel('Regularization Parameter (C)', fontsize=12)
    ax.set_ylabel('Kernel Width (σ)', fontsize=12)
    ax.set_title('SVM Grid Search: 10-Fold Cross-Validation Results', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svm_grid_search.png', dpi=150)
    plt.show()

#Trains the final svm on the ideal parameters
def train_final_svm(x_train, y_train, best_params):
    clf = svm.SVC(kernel='rbf', gamma=best_params['gamma'], C=best_params['C'])
    clf.fit(x_train, y_train)
    return clf

#Helper function to plot the decision boundary
def plot_decision_boundary(model, x_test, y_test, title='SVM Decision Boundary'):
    # Create mesh grid
    x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
    y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    # Get decision function values
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    for label in class_labels:
        mask = (y_test == label)
        ax.scatter(x_test[mask, 0], x_test[mask, 1], 
                  label=f'Class {label}', s=1, alpha=0.3, edgecolors='k')
    
    # Plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
              s=50, linewidth=1, facecolors='none', edgecolors='yellow',
              label=f'Support Vectors (n={len(model.support_vectors_)})')
    
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svm_decision_boundary.png', dpi=150)
    plt.show()

# function to evaluate the model on the test
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = np.mean(y_pred == y_test)
    
    # Calculate error by class
    for label in class_labels:
        mask = (y_test == label)
        class_accuracy = np.mean(y_pred[mask] == y_test[mask])
        print(f"Class {label} Accuracy: {class_accuracy*100:.2f}%")
    
    print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Error Rate: {(1-accuracy)*100:.2f}%")
    
    return accuracy


def main():
    x_train, y_train = generate_dataset(1000)
    x_test, y_test = generate_dataset(10000)
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    plot_dataset(x_train, y_train, x_test, y_test)
    
    best_params, grid_results, c_values, kernel_width_values = \
        svm_grid_search_cv(x_train, y_train, k_folds=10)
    
    plot_grid_search_results(grid_results, c_values, kernel_width_values, best_params)
    
    final_model = train_final_svm(x_train, y_train, best_params)
    
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(final_model, x_test, y_test)
    
    plot_decision_boundary(final_model, x_test, y_test, 
                          f'SVM Decision Boundary (Test Accuracy: {test_accuracy*100:.2f}%)')

if __name__ == "__main__":
    main()