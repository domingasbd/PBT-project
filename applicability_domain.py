import numpy as np
from sklearn.decomposition import PCA

def analyze_applicability_domain(train_data, test_data, variance_threshold=0.90, 
                               threshold_percentile=99):
    """Analyze applicability domain using PCA and Mahalanobis distance."""
    # Initialize and fit PCA
    pca_full = PCA()
    pca_full.fit(train_data)
    
    # Find number of components for target variance
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= variance_threshold) + 1
    print(f"\nUsing {n_components} components to explain {variance_threshold*100}% of variance")
    
    # Perform PCA with determined components
    pca = PCA(n_components=n_components)
    pca_train = pca.fit_transform(train_data)
    pca_test = pca.transform(test_data)
    
    # Calculate Mahalanobis distances
    train_mean = np.mean(pca_train, axis=0)
    train_cov = np.cov(pca_train.T)
    
    def calculate_distance(point):
        diff = point - train_mean
        return np.sqrt(diff.dot(np.linalg.inv(train_cov)).dot(diff.T))
    
    train_distances = np.array([calculate_distance(point) for point in pca_train])
    test_distances = np.array([calculate_distance(point) for point in pca_test])
    
    # Define threshold and classify compounds
    threshold = np.percentile(train_distances, threshold_percentile)
    train_out_indices = train_distances > threshold
    test_out_indices = test_distances > threshold
    
    # Print detailed training distances information
    print(f"\nTraining distances statistics:")
    print(f"99th percentile threshold: {threshold:.3f} (This means 99% of training compounds have distances below this value)")
    print(f"Training set - Min distance: {np.min(train_distances):.3f}")
    print(f"Training set - Max distance: {np.max(train_distances):.3f}")
    print(f"Training set - Mean distance: {np.mean(train_distances):.3f}")
    print(f"Training set - Median distance: {np.median(train_distances):.3f}")
    
    print(f"\nTest set distances statistics:")
    print(f"Test set - Min distance: {np.min(test_distances):.3f}")
    print(f"Test set - Max distance: {np.max(test_distances):.3f}")
    print(f"Test set - Mean distance: {np.mean(test_distances):.3f}")
    print(f"Test set - Median distance: {np.median(test_distances):.3f}")
    
    # Print counts above threshold
    n_train_above = sum(train_distances > threshold)
    n_test_above = sum(test_distances > threshold)
    print(f"\nCompounds above threshold:")
    print(f"Training set: {n_train_above} ({(n_train_above/len(train_distances)*100):.1f}% of training set)")
    print(f"Test set: {n_test_above} ({(n_test_above/len(test_distances)*100):.1f}% of test set)")
    
    
    # Print summary statistics
    print("\nApplicability Domain Analysis Results")
    print("-" * 50)
    print(f"Training set: {sum(train_out_indices)} compounds out of {len(train_data)} ({sum(train_out_indices)/len(train_data)*100:.1f}%) out of domain")
    print(f"Test set: {sum(test_out_indices)} compounds out of {len(test_data)} ({sum(test_out_indices)/len(test_data)*100:.1f}%) out of domain")

    
    return {
        'train_out_indices': train_out_indices,
        'test_out_indices': test_out_indices,
        'threshold': threshold,
        'pca': pca,
        'explained_variance': sum(pca.explained_variance_ratio_),
        'n_components': n_components
    }