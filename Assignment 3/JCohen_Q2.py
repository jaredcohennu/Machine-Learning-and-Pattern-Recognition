import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import seaborn as sns

# Set rng seeds
np.random.seed(42)

# Define GMM
n_components_true = 4
n_dims = 2
mean1 = np.array([2.0, 2.0])
mean2 = np.array([2.5, 2.5])  # overlaps with component 1
mean3 = np.array([-2.0, -2.0])
mean4 = np.array([-2.5, 2.5])
means_true = np.array([mean1, mean2, mean3, mean4])
cov1 = np.array([[1.0, 0.3], [0.3, 1.0]])
cov2 = np.array([[1.2, -0.2], [-0.2, 0.8]])
cov3 = np.array([[0.9, 0.4], [0.4, 1.1]])
cov4 = np.array([[1.1, -0.3], [-0.3, 0.9]])
covariances_true = np.array([cov1, cov2, cov3, cov4])
weights_true = np.array([0.3, 0.2, 0.35, 0.15])

# Calculate distances between means to verify overlap
print(f"\nDistances between mean vectors:")
for i in range(n_components_true):
    for j in range(i+1, n_components_true):
        dist = np.linalg.norm(means_true[i] - means_true[j])
        avg_eigenval = (np.mean(np.linalg.eigvals(covariances_true[i])) + 
                       np.mean(np.linalg.eigvals(covariances_true[j]))) / 2
        print(f"  Components {i+1}-{j+1}: {dist:.3f} (avg eigenvalue sum: {2*avg_eigenval:.3f})")

#Function to help generate the data given all the parameters needed
def generate_gmm_data(means, covariances, weights, n_samples):
    """Generate samples from a GMM"""
    n_components = len(weights)
    n_dims = means.shape[1]
    
    # Generate assignments
    component_assignments = np.random.choice(n_components, size=n_samples, p=weights)
    
    # Generate samples
    X = np.zeros((n_samples, n_dims))
    for i in range(n_samples):
        component = component_assignments[i]
        X[i] = np.random.multivariate_normal(means[component], covariances[component])
    
    return X, component_assignments

# Dataset sizes
dataset_sizes = [10, 100, 1000]
datasets = {}

for size in dataset_sizes:
    X, labels = generate_gmm_data(means_true, covariances_true, weights_true, size)
    datasets[size] = {'X': X, 'labels': labels}

# Plots for the datasets
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = ['red', 'blue', 'green', 'orange']

for idx, size in enumerate(dataset_sizes):
    ax = axes[idx]
    X = datasets[size]['X']
    labels = datasets[size]['labels']
    
    for comp in range(n_components_true):
        mask = labels == comp
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[comp], 
                  label=f'Comp {comp+1}', alpha=0.6, s=30)
        
    ax.scatter(means_true[:, 0], means_true[:, 1], 
              c='black', marker='X', s=200, edgecolors='white', linewidths=2,
              label='True means', zorder=5)
    
    ax.set_xlabel('X₁', fontsize=11)
    ax.set_ylabel('X₂', fontsize=11)
    ax.set_title(f'Dataset: N={size} samples', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('datasets_visualization.png', dpi=150, bbox_inches='tight')
plt.show()



candidate_orders = list(range(1, 11))
n_folds = 10

#Fit GMM using EM algorithm using sklearn
def fit_gmm_with_em(X_train, n_components, max_iter=200):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=max_iter,
        n_init=5,  # Multiple random initializations
        random_state=None  # Allow different initializations
    )
    gmm.fit(X_train)
    return gmm

#Helper function to get the log likelihood
def compute_log_likelihood(gmm, X):
    return gmm.score(X)

#Perform the k fold validation on the GMM
def cross_validate_gmm(X, n_components, n_folds=10):
    n_samples = X.shape[0]
    
    # Must skip if the components are more than samples
    min_train_samples = n_samples * (n_folds - 1) // n_folds
    if n_components > min_train_samples:
        # Return very poor log-likelihood to discourage selection
        return -np.inf, 0.0
    
    # Adjust n_folds if dataset is too small
    actual_folds = min(n_folds, n_samples // (n_components + 1))
    if actual_folds < 2:
        actual_folds = 2  # Minimum for CV
    
    kfold = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
    
    fold_log_likelihoods = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        
        # check to see if we have enough samples
        if X_train_fold.shape[0] < n_components:
            continue
        
        try:
            # Fit and evaluate GMM on training fold
            gmm = fit_gmm_with_em(X_train_fold, n_components)
            val_log_likelihood = compute_log_likelihood(gmm, X_val_fold)
            fold_log_likelihoods.append(val_log_likelihood)
        except:
            # If fitting fails, skip this fold
            continue
    
    if len(fold_log_likelihoods) == 0:
        return -np.inf, 0.0
    
    return np.mean(fold_log_likelihoods), np.std(fold_log_likelihoods)

# Perform CV for each dataset and each model order
cv_results = {}

for size in dataset_sizes:
    print(f"Cross-Validation for Dataset Size: {size}")
    
    X = datasets[size]['X']
    cv_results[size] = {
        'orders': candidate_orders,
        'mean_log_likelihoods': [],
        'std_log_likelihoods': []
    }
    
    for n_comp in candidate_orders:
        print(f"\n  Testing {n_comp} components...", end='')
        mean_ll, std_ll = cross_validate_gmm(X, n_comp, n_folds)
        if np.isinf(mean_ll):
            print(f" SKIPPED (insufficient samples)")
        else:
            print(f" Mean Log-Likelihood: {mean_ll:.4f} ± {std_ll:.4f}")
        cv_results[size]['mean_log_likelihoods'].append(mean_ll)
        cv_results[size]['std_log_likelihoods'].append(std_ll)
    
    # Select best model order, ignoring -inf
    valid_lls = [ll if not np.isinf(ll) else -1e10 for ll in cv_results[size]['mean_log_likelihoods']]
    best_idx = np.argmax(valid_lls)
    best_order = candidate_orders[best_idx]
    best_ll = cv_results[size]['mean_log_likelihoods'][best_idx]
    
    cv_results[size]['best_order'] = best_order
    cv_results[size]['best_log_likelihood'] = best_ll
    
    print(f"\n  *** Best Model Order: {best_order} components ***")
    print(f"  *** Best CV Log-Likelihood: {best_ll:.4f} ***")

print("Repeating Experiment 100 Times")

n_experiments = 100
selection_results = {size: np.zeros(len(candidate_orders)) for size in dataset_sizes}

for exp_idx in range(n_experiments):
    if (exp_idx + 1) % 20 == 0:
        print(f"Completed {exp_idx + 1}/{n_experiments} experiments...")
    
    for size in dataset_sizes:
        # Generate new dataset
        X, _ = generate_gmm_data(means_true, covariances_true, weights_true, size)
        
        #evaluate on all of the model orders
        log_likelihoods = []
        for n_comp in candidate_orders:
            mean_ll, _ = cross_validate_gmm(X, n_comp, n_folds)
            log_likelihoods.append(mean_ll)
        
        #select best order (ignoring -inf values)
        valid_lls = [ll if not np.isinf(ll) else -1e10 for ll in log_likelihoods]
        best_idx = np.argmax(valid_lls)
        best_order = candidate_orders[best_idx]
        
        selection_results[size][best_order - 1] += 1

selection_rates = {}
for size in dataset_sizes:
    selection_rates[size] = selection_results[size] / n_experiments


#Summarize and create the graphs
print("FINAL RESULTS SUMMARY")

# Print selection rates table
print("Model Selection Rates (over 100 experiments)")
print(f"\n{'Model Order':<12}", end='')
for size in dataset_sizes:
    print(f"N={size:<8}", end='')
print()

for order_idx, order in enumerate(candidate_orders):
    print(f"{order:<12}", end='')
    for size in dataset_sizes:
        rate = selection_rates[size][order_idx]
        print(f"{rate:.3f}    ", end='')
    print()

print("MOST FREQUENTLY SELECTED MODEL ORDERS")
for size in dataset_sizes:
    most_selected_idx = np.argmax(selection_rates[size])
    most_selected_order = candidate_orders[most_selected_idx]
    selection_rate = selection_rates[size][most_selected_idx]
    print(f"  N={size:4d}: {most_selected_order} components (selected {selection_rate*100:.1f}% of the time)")

print(f"\n  True model order: {n_components_true} components")

# CV Log-Likelihood curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, size in enumerate(dataset_sizes):
    ax = axes[idx]
    mean_lls = cv_results[size]['mean_log_likelihoods']
    std_lls = cv_results[size]['std_log_likelihoods']
    
    ax.plot(candidate_orders, mean_lls, 'o-', linewidth=2, markersize=8, color='blue')
    ax.fill_between(candidate_orders, 
                    np.array(mean_lls) - np.array(std_lls),
                    np.array(mean_lls) + np.array(std_lls),
                    alpha=0.3, color='blue')
    
    best_order = cv_results[size]['best_order']
    best_ll = cv_results[size]['best_log_likelihood']
    ax.axvline(x=best_order, color='red', linestyle='--', linewidth=2, 
              label=f'Selected: {best_order}')
    ax.axvline(x=n_components_true, color='green', linestyle='--', linewidth=2,
              label=f'True: {n_components_true}')
    
    ax.set_xlabel('Number of Components', fontsize=11)
    ax.set_ylabel('Mean Log-Likelihood', fontsize=11)
    ax.set_title(f'N={size} samples', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(candidate_orders)

plt.suptitle('Cross-Validation Log-Likelihood vs Model Order', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('cv_log_likelihood.png', dpi=150, bbox_inches='tight')
plt.show()

# Bar chart of selection rates
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, size in enumerate(dataset_sizes):
    ax = axes[idx]
    rates = selection_rates[size]
    
    colors = ['green' if order == n_components_true else 'steelblue' 
             for order in candidate_orders]
    
    bars = ax.bar(candidate_orders, rates, color=colors, alpha=0.7, edgecolor='black')
    
    # Highlight the true order
    ax.axvline(x=n_components_true, color='red', linestyle='--', 
              linewidth=2, label=f'True order: {n_components_true}')
    
    ax.set_xlabel('Number of Components', fontsize=11)
    ax.set_ylabel('Selection Rate', fontsize=11)
    ax.set_title(f'N={size} samples', fontsize=12, fontweight='bold')
    ax.set_xticks(candidate_orders)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Model Selection Frequency (100 experiments)', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('selection_frequency_bars.png', dpi=150, bbox_inches='tight')
plt.show()