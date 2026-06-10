#%% Load data

import scipy.io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd

mat = scipy.io.loadmat('studentData.mat')

InputData = mat['input']
OutputData = mat['output']

GLOBAL_SEED = 36544
np.random.seed(GLOBAL_SEED)

c_value = 4
m_value = 2 
max_iterations = 40

tau = 0.95

# ==================================================
# Method with random sorting for train/test split and normalization
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    InputData, 
    OutputData, 
    test_size=0.20, 
    random_state=GLOBAL_SEED,  
    shuffle=True
)

y_min_train = y_train.min(axis=0)
y_max_train = y_train.max(axis=0)
y_train_norm = (y_train - y_min_train) / (y_max_train - y_min_train)
y_test_norm = (y_test- y_min_train) / (y_max_train - y_min_train)

x_min_train = X_train.min(axis=0)
x_max_train = X_train.max(axis=0)
X_train_norm = (X_train - x_min_train) / (x_max_train - x_min_train)
X_test_norm = (X_test - x_min_train) / (x_max_train - x_min_train)


# tukej se je uprabila drigačna metoda skaliranja.
# --- Scaling Target Variables (y) ---
# y_mean_train = y_train.mean(axis=0)
# y_std_train = y_train.std(axis=0)

# # Prevent division by zero just in case standard deviation is 0
# y_std_train[y_std_train == 0] = 1.0

# y_train_norm = (y_train - y_mean_train) / y_std_train
# y_test_norm = (y_test - y_mean_train) / y_std_train


# # --- Scaling Feature Variables (X) ---
# x_mean_train = X_train.mean(axis=0)
# x_std_train = X_train.std(axis=0)

# # Prevent division by zero
# x_std_train[x_std_train == 0] = 1.0

# X_train_norm = (X_train - x_mean_train) / x_std_train
# X_test_norm = (X_test - x_mean_train) / x_std_train


plot = plt.scatter(X_train[:,1], X_train[:,5])
plt.xlabel('Paramether 6')
plt.ylabel('Paramether 2')
plt.title('Model A, Normal Split: Paramether 6 vs Paramether 2')
plt.grid(True)
plt.show()

plot = plt.scatter(X_train_norm[:,1], X_train_norm[:,5])
plt.xlabel('Normalized Paramether 6')
plt.ylabel('Normalized Paramether 2')
plt.title('Model A, Normal Split: Normalized Paramether 6 vs Normalized Paramether 2')
plt.grid(True)
plt.show()

#FCM classification

def FCM(X, c, m=2, max_iter=100):
    N, features = X.shape
    # Standard normal or uniform initialization works for any features count
    centers = np.random.rand(c, features) 
    J_history = []

    PC, XB, SIL = [], [], []
    
    for iteration in range(max_iter):
        # 1. Compute distances (Works for any number of features via axis=2)
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        distances = np.maximum(distances, 1e-10)  
        
        # 2. VECTORIZED Membership Update (Fixes the speed bottleneck)
        # distances[:, :, None] has shape (N, c, 1)
        # distances[:, None, :] has shape (N, 1, c)
        denom = np.sum((distances[:, :, np.newaxis] / distances[:, np.newaxis, :]) ** (2 / (m - 1)), axis=2)
        U = 1.0 / denom.T
        
        # 3. VECTORIZED Centers Update
        um = U ** m
        centers = np.dot(um, X) / np.sum(um, axis=1)[:, np.newaxis]
        
        # 4. Objective Function
        J = np.sum((U**m) * (distances.T**2))
        J_history.append(J)

        # Validation Indices
        PC.append(partition_coefficient(N, U))
        XB.append(xie_beni_index(J, centers, N))
        
        # Highly recommended to move SIL out of this loop if N is large!
        SIL.append(SIL_calculation(X, U)) 
        
    return PC, XB, SIL, U, centers, J_history

def partition_coefficient(N, U):
    return np.sum(U**2) / N

def xie_beni_index(J, centers, N):  
    
    # Calculate minimum center separation (excluding zero diagonal)
    center_distances = np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)

    np.fill_diagonal(center_distances, np.inf) # Set diagonal to infinity to exclude zero distances

    min_center_dist = np.min(center_distances)
    denominator = N * (min_center_dist ** 2)
    
    return J / denominator

def SIL_calculation(X, U):
    # Hard assignment: assuming U has shape (num_clusters, N)
    X_labels = np.argmax(U, axis=0) 
    unique_clusters = np.unique(X_labels)
    
    N = X.shape[0]
    a = np.zeros(N)
    b = np.zeros(N)

    for sample in range(N):
        current_cluster = X_labels[sample]
        
        # --- Calculate a_k (Same Cluster) ---
        a_mask = (X_labels == current_cluster)
        a_mask[sample] = False  # Exclude the sample itself
        
        a_points = X[a_mask]
        
        # FIX: Check if there are other points in the same cluster
        if len(a_points) > 0:
            a_distances = np.linalg.norm(X[sample] - a_points, axis=1)
            a[sample] = np.mean(a_distances)
        else:
            a[sample] = 0.0  # Lone point in cluster defaults to 0
        
        # --- Calculate b_k (Nearest Other Cluster) ---
        mean_distances_to_other_clusters = []
        
        for other_cluster in unique_clusters:
            if other_cluster == current_cluster:
                continue  # Skip its own cluster
            
            # Mask for the neighboring cluster
            b_mask = (X_labels == other_cluster)
            b_points = X[b_mask]
            
            # FIX: Only calculate if the other cluster actually has points
            if len(b_points) > 0:
                b_distances = np.linalg.norm(X[sample] - b_points, axis=1)
                mean_distances_to_other_clusters.append(np.mean(b_distances))
        
        # b_k is the minimum of the mean distances to the other clusters
        if mean_distances_to_other_clusters:
            b[sample] = np.min(mean_distances_to_other_clusters)
        else:
            b[sample] = 0.0

    # --- Calculate Final Silhouette Coefficient (SIL) ---
    max_ab = np.maximum(a, b)
    
    s = np.zeros(N)
    valid_denom = max_ab > 0
    s[valid_denom] = (b[valid_denom] - a[valid_denom]) / max_ab[valid_denom]
    
    # Global mean SIL score
    sil_score = np.mean(s)
    
    return sil_score

def predict_FCM(X, centers, m=2):
    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    distances = np.maximum(distances, 1e-10)  
        
    N = X.shape[0]
    c = centers.shape[0]

    U = np.zeros((c, N))
    for sample in range(N):
        for cluster in range(c):
            d = distances[sample, cluster]
            denom = np.sum((d / distances[sample]) ** (2/(m-1)))
            U[cluster, sample] = 1 / denom
    return U

# PCA,  LS

def PCA_LS(X, y, tau):
    # Center the data
    X_train_mean =  np.mean(X, axis=0)
    X_centered = X - X_train_mean
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]  
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    sorted_eigenvalues = eigenvalues[sorted_indices]  # Also sort eigenvalues

    for i in range(len(eigenvalues)):
        EVR = np.sum(sorted_eigenvalues[:i+1]) / np.sum(eigenvalues)
        #print(f"Number of components: {i+1}, Explained Variance Ratio: {EVR:.4f}")
        if EVR >= tau:
            n_components = i + 1  # Note: i+1, not i, since you want count of components
            print(f"Selected {n_components} principal components to retain {tau*100}% of variance.")
            break

    # Select top n_components eigenvectors
    W = sorted_eigenvectors[:, :n_components]
    
    # Project data onto selected eigenvectors
    X_reduced = X_centered @ W
    

    # theta calcuation using least squares
    N = X_centered.shape[0]

    X =np.column_stack([np.ones(N), X_reduced])
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    
    return theta, X_train_mean,W

def predict_PSA_LS(X, theta, X_train_mean, W): 
    X_centered = X - X_train_mean

    X_reduced = X_centered @ W
    N = X_reduced.shape[0]
    X_augmented = np.column_stack([np.ones(N), X_reduced])
    predictions = X_augmented @ theta

    return predictions

def LS(X, y):
    N = X.shape[0]
    X_augmented = np.column_stack([np.ones(N), X])
    theta = np.linalg.pinv(X_augmented.T @ X_augmented) @ X_augmented.T @ y
    return theta

def predict_LS(X, theta):
    N = X.shape[0]
    X_augmented = np.column_stack([np.ones(N), X])
    predictions = X_augmented @ theta
    return predictions

def RMSE_mean_calculation(y_hat,y_mean_for_samples):
    return np.sqrt(np.mean((y_hat.ravel() - y_mean_for_samples)**2))

def print_rmse_mean_per_class(y_hat, y_true, labels, c_value):
    """
    Calculate RMSE_mean per class (comparing predictions to cluster means)
    """
    # Flatten arrays
    y_hat = y_hat.reshape(-1)
    y_true = y_true.reshape(-1)
    labels = labels.reshape(-1)
    
    print(f"\n{'Class':<8} {'Count':<8} {'RMSE_mean':<12}")
    print(f"{'-'*30}")
    
    for cluster in range(c_value):
        mask = labels == cluster
        if np.sum(mask) > 0:
            # Get predictions for this class
            y_hat_class = y_hat[mask]
            
            # Calculate mean of true values for this class
            y_mean_class = np.mean(y_true[mask])
            
            # Calculate RMSE between predictions and class mean
            rmse_mean = np.sqrt(np.mean((y_hat_class - y_mean_class)**2))
            
            print(f"{cluster:<8} {np.sum(mask):<8} {rmse_mean:<12.6f}")

#%% FCM clustering

# ==================================================================
# Model A: Using only parameters 2 and 6 for clustering and PCA + LS
# ==================================================================

print("================================")
print("Model A: Using only parameters 2 and 6 for clustering and PCA + LS, Normal Split")
print("================================")
# matlab notation is {2,6} in python it is {1,5}
X = X_train_norm[:, [1,5]]

print("\nRunning FCM clustering, Normal Split...\n")


PC, XB,SIL, U, centers, J_history = FCM(X=X, c = c_value, m = m_value, max_iter = max_iterations)


print("Partition Coefficient (PC):", PC[-1])
print("Xie-Beni Index (XB):", XB[-1])
print("Mean silhouette coefficien (SLI):", SIL[-1])

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Top-Left: Partition Coefficient (PC)
axs[0, 0].plot(PC)
axs[0, 0].set_xlabel('Iteration')
axs[0, 0].set_ylabel('Partition Coefficient (PC)')
axs[0, 0].set_title('FCM Partition Coefficient Convergence, Normal Split')
axs[0, 0].grid(True)

# Top-Right: Xie-Beni Index (XB)
axs[0, 1].plot(XB)
axs[0, 1].set_xlabel('Iteration')
axs[0, 1].set_ylabel('Xie-Beni Index (XB)')
axs[0, 1].set_title('FCM Xie-Beni Index Convergence, Normal Split')
axs[0, 1].grid(True)

# Bottom-Left: Mean Silhouette Coefficient (SIL)
axs[1, 0].plot(SIL)
axs[1, 0].set_xlabel('Iteration')
axs[1, 0].set_ylabel('Mean silhouette coefficient (SIL)')
axs[1, 0].set_title('FCM Mean silhouette coefficient, Normal Split')
axs[1, 0].grid(True)

# Bottom-Right: Objective Function J (J_history)
axs[1, 1].plot(J_history)
axs[1, 1].set_xlabel('Iteration')
axs[1, 1].set_ylabel('Objective Function J')
axs[1, 1].set_title('FCM Objective Function Convergence, Normal Split')
axs[1, 1].grid(True)

# Automatically adjust spacing between subplots to prevent overlapping
plt.tight_layout()

# Display the combined plot
plt.show()

plot = plt.scatter(X[:,0], X[:,1], c=np.argmax(U, axis=0))
plt.scatter(centers[:,0], centers[:,1], c='red', marker='X', label='Cluster Centers')
plt.xlabel('Normalized Paramether 6')
plt.ylabel('Normalized Paramether 2')   
plt.title('Model A - FCM Clustering Results, Normal Split')
plt.grid(True)
plt.show()

print("\n")


#%% Train all PCAs and LSs models

X_train_labels=np.argmax(U, axis=0)

# Create DataFrame with all features and cluster labels
df = pd.DataFrame(X_train, columns=[f'Feature_{i+1}' for i in range(X_train.shape[1])])
df['Cluster'] = X_train_labels

# Pairplot - shows all 2D combinations
sns.pairplot(df, hue='Cluster', diag_kind='hist', palette='Set1')
plt.suptitle('Model A- [2,6] Feature ', y=1.02)
plt.show()


theats = {}
X_mean_Class = {}
W_class = {}



for i in range(c_value):
    print(f"Training PCA + LS for Class {i}...")
    X_train_class = X_train_norm[X_train_labels == i]
    y_train_class = y_train_norm[X_train_labels == i]

    theats[i],X_mean_Class[i],W_class[i] = PCA_LS(X_train_class, y_train_class,tau = tau)


y_hat = np.zeros_like(y_train_norm)

for sample in range(X_train_norm.shape[0]):
    #print(f"Sample {sample}: Assigned Class {X_train_labels[sample]}")

    classSelection = X_train_labels[sample]

    y_hat[sample] = predict_PSA_LS(X_train_norm[sample].reshape(1, -1), 
                            theats[classSelection], 
                            X_mean_Class[classSelection], 
                            W_class[classSelection])


#% Evaluate the training model for the selected class
print("\n")

#create mean cluster
y_mean_per_cluster = {}

for cluster in range(c_value):
    cluster_indices = np.where(X_train_labels == cluster)[0]
    y_mean_per_cluster[cluster] = np.mean(y_train_norm[cluster_indices])


print("================================")
print("Training: Evaluating the model for the selected class with PCA + LS...")
print("================================")
print("Root Mean Square Error:", np.sqrt(np.mean((y_hat - y_train_norm)**2)))


# RMSE_mean
# Dictionary to store mean y for each cluster
y_mean_for_samples = np.array([y_mean_per_cluster[label] for label in X_train_labels])
RMSE_mean = RMSE_mean_calculation(y_hat,y_mean_for_samples)
print(f"RMSE_mean: {RMSE_mean:.4f}")
print_rmse_mean_per_class(y_hat, y_train_norm, X_train_labels, c_value)

#Evaluating on the test set


U_Test = predict_FCM(X_test_norm[:,[1,5]], centers, m = m_value)
X_test_labels=np.argmax(U_Test, axis=0)


y_test_hat = np.zeros_like(y_test_norm)
for sample in range(X_test_norm.shape[0]):
    #print(f"Sample {sample}: Assigned Class {X_test_labels[sample]}")

    classSelection = X_test_labels[sample]

    y_test_hat[sample] = predict_PSA_LS(X_test_norm[sample].reshape(1, -1), 
                            theats[classSelection], 
                            X_mean_Class[classSelection], 
                            W_class[classSelection])

print("\n================================")
print("Test: Evaluating the model for the selected class with PCA + LS...")
print("================================")
print("Root Mean Square Error:", np.sqrt(np.mean((y_test_hat - y_test_norm)**2)))

y_mean_for_samples = np.array([y_mean_per_cluster[label] for label in X_test_labels])
RMSE_mean = RMSE_mean_calculation(y_test_hat,y_mean_for_samples)
print(f"RMSE_mean: {RMSE_mean:.4f}")
print_rmse_mean_per_class(y_test_hat, y_test_norm, X_test_labels, c_value)

#%% ================================================================
# Model B – no clustering (single global model)
# ==================================================================


print("\n=================================================================")
print("Model B: No clustering, single global PCA (with) + LS model, Normal Split")
print("=================================================================")

y_mean_per_cluster = {}

for cluster in range(c_value):
    cluster_indices = np.where(X_train_labels == cluster)[0]
    y_mean_per_cluster[cluster] = np.mean(y_train_norm[cluster_indices])



theta_global, X_train_mean_global, W_global = PCA_LS(X_train_norm, y_train_norm, tau = tau)

y_train_hat_global = predict_PSA_LS(X_train_norm, theta_global, X_train_mean_global, W_global)
rmse_global_train = np.sqrt(np.mean((y_train_hat_global - y_train_norm)**2))
print("Global Model with PCA + LS - Training RMSE:", rmse_global_train)

y_mean_for_samples = np.array([y_mean_per_cluster[label] for label in X_train_labels])
RMSE_mean = RMSE_mean_calculation(y_train_hat_global, y_mean_for_samples)
print(f"Global RMSE_mean: {RMSE_mean:.4f}")
print_rmse_mean_per_class(y_train_hat_global, y_train_norm, X_train_labels, c_value)


y_test_hat_global = predict_PSA_LS(X_test_norm, theta_global, X_train_mean_global, W_global)
rmse_global_test = np.sqrt(np.mean((y_test_hat_global - y_test_norm)**2))
print("Global Model with PCA + LS - Test RMSE:", rmse_global_test)

y_mean_for_samples = np.array([y_mean_per_cluster[label] for label in X_test_labels])
RMSE_mean = RMSE_mean_calculation(y_test_hat_global,y_mean_for_samples)
print(f"Global RMSE_mean: {RMSE_mean:.4f}")


print_rmse_mean_per_class(y_test_hat_global, y_test_norm, X_test_labels, c_value)

#%%
print("\n=================================================================")
print("Model B: No clustering, single global PCA (without) + LS model, Normal Split")
print("=================================================================")

y_mean_per_cluster = {}

for cluster in range(c_value):
    cluster_indices = np.where(X_train_labels == cluster)[0]
    y_mean_per_cluster[cluster] = np.mean(y_train_norm[cluster_indices])


theta_global = LS(X_train_norm, y_train_norm)
y_train_hat_global = predict_LS(X_train_norm, theta_global)
rmse_global_train = np.sqrt(np.mean((y_train_hat_global - y_train_norm)**2))
print("Global Model with LS - Training RMSE:", rmse_global_train)


y_mean_for_samples = np.array([y_mean_per_cluster[label] for label in X_train_labels])
RMSE_mean = RMSE_mean_calculation(y_train_hat_global, y_mean_for_samples)
print(f"Global RMSE_mean: {RMSE_mean:.4f}")
print_rmse_mean_per_class(y_train_hat_global, y_train_norm, X_train_labels, c_value)

y_test_hat_global = predict_LS(X_test_norm, theta_global)
rmse_global_test = np.sqrt(np.mean((y_test_hat_global - y_test_norm)**2))
print("Global Model with LS - Test RMSE:", rmse_global_test)

y_mean_for_samples = np.array([y_mean_per_cluster[label] for label in X_test_labels])
RMSE_mean = RMSE_mean_calculation(y_test_hat_global,y_mean_for_samples)
print(f"Global RMSE_mean: {RMSE_mean:.4f}")
print_rmse_mean_per_class(y_test_hat_global, y_test_norm, X_test_labels, c_value)


#%% ================================================================
# Model C – GK on all p dimensions
# ==================================================================

print("\n=================================================================")
print("Model C – GK on all p dimensions, Normal Split")
print("=================================================================")

print("\nRunning FCM clustering, Normal Split...\n")

PC, XB, SIL, U, centers, J_history = FCM(X_train_norm, c = c_value, m = m_value, max_iter = max_iterations)

print("Partition Coefficient (PC):", PC[-1])
print("Xie-Beni Index (XB):", XB[-1])
print("Mean silhouette coefficien (SLI):", SIL[-1])

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Top-Left: Partition Coefficient (PC)
axs[0, 0].plot(PC)
axs[0, 0].set_xlabel('Iteration')
axs[0, 0].set_ylabel('Partition Coefficient (PC)')
axs[0, 0].set_title('FCM Partition Coefficient Convergence, Normal Split')
axs[0, 0].grid(True)

# Top-Right: Xie-Beni Index (XB)
axs[0, 1].plot(XB)
axs[0, 1].set_xlabel('Iteration')
axs[0, 1].set_ylabel('Xie-Beni Index (XB)')
axs[0, 1].set_title('FCM Xie-Beni Index Convergence, Normal Split')
axs[0, 1].grid(True)

# Bottom-Left: Mean Silhouette Coefficient (SIL)
axs[1, 0].plot(SIL)
axs[1, 0].set_xlabel('Iteration')
axs[1, 0].set_ylabel('Mean silhouette coefficient (SIL)')
axs[1, 0].set_title('FCM Mean silhouette coefficient, Normal Split')
axs[1, 0].grid(True)

# Bottom-Right: Objective Function J (J_history)
axs[1, 1].plot(J_history)
axs[1, 1].set_xlabel('Iteration')
axs[1, 1].set_ylabel('Objective Function J')
axs[1, 1].set_title('FCM Objective Function Convergence, Normal Split')
axs[1, 1].grid(True)

# Automatically adjust spacing between subplots to prevent overlapping
plt.tight_layout()

# Display the combined plot
plt.show()

X_train_labels_old = X_train_labels
X_train_labels=np.argmax(U, axis=0)

# For X_train_labels_old
print("\n=== X_train_labels_old ===")
unique_old, counts_old = np.unique(X_train_labels_old, return_counts=True)
for val, count in zip(unique_old, counts_old):
    print(f"Value {val}: {count} times")

print("\n=== X_train_labels ===")
unique_new, counts_new = np.unique(X_train_labels, return_counts=True)
for val, count in zip(unique_new, counts_new):
    print(f"Value {val}: {count} times")

print("")


plot = plt.scatter(X_train_norm[:,1], X_train_norm[:,5], c=np.argmax(U, axis=0))
plt.scatter(centers[:,1], centers[:,5], c='red', marker='X', label='Cluster Centers')
plt.xlabel('Normalized Paramether 6')
plt.ylabel('Normalized Paramether 2')   
plt.title('Model C - FCM Clustering Results, Normal Split')
plt.grid(True)
plt.show()


theats = {}
X_mean_Class = {}
W_class = {}


for i in range(c_value):
    print(f"Training PCA + LS for Class {i}...")
    X_train_class = X_train_norm[X_train_labels == i]
    y_train_class = y_train_norm[X_train_labels == i]

    theats[i],X_mean_Class[i],W_class[i] = PCA_LS(X_train_class, y_train_class,tau = tau)


y_hat = np.zeros_like(y_train_norm)

for sample in range(X_train_norm.shape[0]):
    #print(f"Sample {sample}: Assigned Class {X_train_labels[sample]}")

    classSelection = X_train_labels[sample]

    y_hat[sample] = predict_PSA_LS(X_train_norm[sample].reshape(1, -1), 
                            theats[classSelection], 
                            X_mean_Class[classSelection], 
                            W_class[classSelection])



y_mean_per_cluster = {}

for cluster in range(c_value):
    cluster_indices = np.where(X_train_labels == cluster)[0]
    y_mean_per_cluster[cluster] = np.mean(y_train_norm[cluster_indices])


print("\n================================")
print("Training: Evaluating the model for the selected class with PCA + LS...")
print("================================")
print("Root Mean Square Error:", np.sqrt(np.mean((y_hat - y_train_norm)**2)))

y_mean_for_samples = np.array([y_mean_per_cluster[label] for label in X_train_labels])
RMSE_mean = RMSE_mean_calculation(y_hat,y_mean_for_samples)
print(f"RMSE_mean: {RMSE_mean:.4f}")
print_rmse_mean_per_class(y_hat, y_train_norm, X_train_labels, c_value)



U_Test = predict_FCM(X_test_norm, centers, m = m_value)
X_test_labels=np.argmax(U_Test, axis=0)


y_test_hat = np.zeros_like(y_test_norm)
for sample in range(X_test_norm.shape[0]):
    #print(f"Sample {sample}: Assigned Class {X_test_labels[sample]}")

    classSelection = X_test_labels[sample]

    y_test_hat[sample] = predict_PSA_LS(X_test_norm[sample].reshape(1, -1), 
                            theats[classSelection], 
                            X_mean_Class[classSelection], 
                            W_class[classSelection])

print("\n================================")
print("Test: Evaluating the model for the selected class with PCA + LS...")
print("================================")
print("Root Mean Square Error:", np.sqrt(np.mean((y_test_hat - y_test_norm)**2)))

y_mean_for_samples = np.array([y_mean_per_cluster[label] for label in X_test_labels])
RMSE_mean = RMSE_mean_calculation(y_test_hat,y_mean_for_samples)
print(f"RMSE_mean: {RMSE_mean:.4f}")
print_rmse_mean_per_class(y_test_hat, y_test_norm, X_test_labels, c_value)


# Create DataFrame with all features and cluster labels
df = pd.DataFrame(X_train, columns=[f'Feature_{i+1}' for i in range(X_train.shape[1])])
df['Cluster'] = X_train_labels

# Pairplot - shows all 2D combinations
sns.pairplot(df, hue='Cluster', diag_kind='hist', palette='Set1')
plt.suptitle('Model C - All Feature Pairs by Cluster', y=1.02)
plt.show()

# Create DataFrame with all features and cluster labels
df = pd.DataFrame(X_test, columns=[f'Feature_{i+1}' for i in range(X_train.shape[1])])
df['Cluster'] = X_test_labels

# Pairplot - shows all 2D combinations
sns.pairplot(df, hue='Cluster', diag_kind='hist', palette='Set1')
plt.suptitle('Model C - All Feature Pairs by Cluster', y=1.02)
plt.show()
# %%
