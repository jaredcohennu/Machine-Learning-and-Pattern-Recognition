import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold


#Helper function to load an image from file
def load_image_from_file(filepath):
    img_array = cv2.imread(filepath)
    if img_array is None:
        raise FileNotFoundError(f"Could not load image from {filepath}")
    return img_array

#helper function that converts an image to the feature vectors used
def image_to_feature_vectors(image):
    height, width = image.shape[0], image.shape[1]
    num_pixels = height * width
    
    # Initialize feature vector array
    feature_vectors = np.zeros((num_pixels, 5))
    
    idx = 0
    for row in range(height):
        for col in range(width):
            b, g, r = image[row, col]
            # Create normalized feature vector
            feature_vectors[idx] = [row/height, col/width, r/255.0, g/255.0, b/255.0]
            idx += 1
    
    return feature_vectors

#Function that fits the GMM with different components using Kfold CV
def fit_gmm_with_cross_validation(feature_vectors, model_orders, k_folds=10):

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    all_scores = {}
    best_bic = np.inf
    best_gmm = None
    best_num_components = 0
    
    print(f"Running {k_folds}-fold cross-validation...")
    
    for num_components in model_orders:
        print(f"  Testing {num_components} components...", end=" ")
        
        bic_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(feature_vectors)):
            train_data = feature_vectors[train_idx]
            val_data = feature_vectors[val_idx]
            
            # fit the GMM on test data
            gmm = GaussianMixture(
                n_components=num_components,
                covariance_type='full',
                max_iter=100,
                random_state=42
            )
            gmm.fit(train_data)
            
            # Evaluate the performance
            bic = gmm.bic(val_data)
            bic_scores.append(bic)
        
        # Compute average BIC across all folds
        avg_bic = np.mean(bic_scores)
        all_scores[num_components] = avg_bic
        
        print(f"Average BIC: {avg_bic:.2f}")
        
        if avg_bic < best_bic:
            best_bic = avg_bic
            best_num_components = num_components
            
            # Train on the full dataset
            best_gmm = GaussianMixture(
                n_components=num_components,
                covariance_type='full', 
                max_iter=100,
                random_state=42
            )
            best_gmm.fit(feature_vectors)
    
    print(f"\nBest model: {best_num_components} components (BIC: {best_bic:.2f})")
    
    return best_gmm, best_num_components, all_scores

#Helper function that segments an image by assigning a label to each pixel
def segment_image(image, gmm):
    # convert to feature vectors and predict which component it shoul be
    feature_vectors = image_to_feature_vectors(image)
    labels = gmm.predict(feature_vectors)
    
    # Reshape to original image dimensions
    height, width = image.shape[0], image.shape[1]
    segmented_image = labels.reshape(height, width)

    num_components = gmm.n_components
    unique_labels = np.unique(labels)
    
    # Create the grayscale image
    label_to_gray = {}
    for i, label in enumerate(sorted(unique_labels)):
        label_to_gray[label] = int(255 * i / (len(unique_labels) - 1)) if len(unique_labels) > 1 else 128
    
    # Create grayscale label image
    label_image = np.zeros((height, width), dtype=np.uint8)
    for label, gray_value in label_to_gray.items():
        label_image[segmented_image == label] = gray_value
    
    return segmented_image, label_image

#Create the plots with the segmentation
def visualize_results(original_image, segmented_image, label_image, num_components):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Segmented image with color labels
    axes[1].imshow(segmented_image, cmap='tab20')
    axes[1].set_title(f'Segmentation (K={num_components} components)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Grayscale label image
    axes[2].imshow(label_image, cmap='gray')
    axes[2].set_title('Grayscale Labels', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

#helper function to plot the bic scores based on the number of components
def plot_bic_scores(all_scores):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    model_orders = sorted(all_scores.keys())
    bic_values = [all_scores[k] for k in model_orders]
    
    ax.plot(model_orders, bic_values, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of GMM Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average BIC Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Order Selection via Cross-Validation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark best model
    best_k = min(all_scores, key=all_scores.get)
    best_bic = all_scores[best_k]
    ax.plot(best_k, best_bic, 'r*', markersize=20, label=f'Best: K={best_k}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    
    image_path = "Assignment 4/horse.jpg"
    image = load_image_from_file(image_path)
    
    # Downsize the image because it is a little too large
    max_dimension = 200
    height, width = image.shape[0], image.shape[1]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    feature_vectors = image_to_feature_vectors(image)
    
    model_orders = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    best_gmm, best_k, all_scores = fit_gmm_with_cross_validation(
        feature_vectors, 
        model_orders,
        k_folds=5
    )
    
    segmented_image, label_image = segment_image(image, best_gmm)
    
    fig1 = visualize_results(image, segmented_image, label_image, best_k)
    fig2 = plot_bic_scores(all_scores)
    
    plt.show()