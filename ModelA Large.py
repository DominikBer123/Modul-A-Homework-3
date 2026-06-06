#%% Load data

import scipy.io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#%% FCM classification

def FCM(X, c, m=2, max_iter=100, epsilon=1e-4):
    N, features = X.shape
    centers = np.random.rand(c, features)
    U = np.zeros((c, N))
    J_history = []
    centers_old = []


    PC = []
    XB = []
    
    for iteration in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        distances = np.maximum(distances, 1e-10)  
        
        # 2. Update membership matrix
        for sample in range(N):
            for cluster in range(c):
                d = distances[sample, cluster]
                denom = np.sum((d / distances[sample]) ** (2/(m-1)))
                U[cluster, sample] = 1 / denom
        
        
        #centers_old.append(centers.copy())

        # 3. Update centers
        for cluster in range(c):
            um = U[cluster] ** m
            numerator = np.sum(um[:, np.newaxis] * X, axis=0)
            denominator = np.sum(um)
            centers[cluster] = numerator / denominator
        
        # 4. Calculate objective function
        J = np.sum((U**m) * (distances.T**2))
        J_history.append(J)

        PC.append(partition_coefficient(N, U))
        XB.append(xie_beni_index(J, centers, N))
        
    
    return PC, XB, U, centers, J_history


# Partition coefficient (PC)
def partition_coefficient(N, U):
    return np.sum(U**2) / N


def xie_beni_index(J, centers, N):  
    
    # Calculate minimum center separation (excluding zero diagonal)
    center_distances = np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)

    np.fill_diagonal(center_distances, np.inf) # Set diagonal to infinity to exclude zero distances

    min_center_dist = np.min(center_distances)
    denominator = N * (min_center_dist ** 2)
    
    return J / denominator


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

#%% PCA + LS

def PCA_LS(X, y, tau = 0.95):
    # Center the data
    X_train_mean =  np.mean(X, axis=0)
    X_centered = X - X_train_mean
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Fixed: removed "dece"
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

def predict(X, theta, X_train_mean, W): 
    X_centered = X - X_train_mean

    X_reduced = X_centered @ W
    N = X_reduced.shape[0]
    X_augmented = np.column_stack([np.ones(N), X_reduced])
    predictions = X_augmented @ theta

    return predictions

#%% Model A

mat = scipy.io.loadmat('studentDataLarge.mat')

InputData = mat['inputLarge']
OutputData = mat['outputLarge']

X_train, X_test, y_train, y_test = train_test_split(
    InputData, 
    OutputData, 
    test_size=0.30, 
    random_state=42,  # Crucial to keep the shuffling synced
    shuffle=True
)

# matlab notation is {2,6} in python it is {1,5}
X = X_train[:, [1,5]]

# Normalization
x_min_train = X.min(axis=0)
x_max_train = X.max(axis=0)
XNormalized = (X - x_min_train) / (x_max_train - x_min_train)

plot = plt.scatter(X[:,0], X[:,1])
plt.xlabel('Paramether 6')
plt.ylabel('Paramether 2')
plt.title('Model A: Paramether 6 vs Paramether 2')
plt.grid(True)
plt.show()

plot = plt.scatter(XNormalized[:,0], XNormalized[:,1])
plt.xlabel('Normalized Paramether 6')
plt.ylabel('Normalized Paramether 2')
plt.title('Model A: Normalized Paramether 6 vs Normalized Paramether 2')
plt.grid(True)
plt.show()

#%% FCM clustering

print("================================")
print("Running FCM clustering...")

c_value = 4
m_value = 2 
max_iterations = 40

PC, XB, U, centers, J_history = FCM(XNormalized, c = c_value, m = m_value, max_iter = max_iterations)


print("Partition Coefficient (PC):", PC[-1])
print("Xie-Beni Index (XB):", XB[-1])

plot = plt.plot(PC)
plt.xlabel('Iteration')
plt.ylabel('Partition Coefficient (PC)')
plt.title('FCM Partition Coefficient Convergence')
plt.grid(True)
plt.show()

plot = plt.plot(XB)
plt.xlabel('Iteration')
plt.ylabel('Xie-Beni Index (XB)')
plt.title('FCM Xie-Beni Index Convergence')
plt.grid(True)
plt.show()


plt.plot(J_history)
plt.xlabel('Iteration')
plt.ylabel('Objective Function J')
plt.title('FCM Objective Function Convergence')
plt.grid(True)
plt.show()

plot = plt.scatter(XNormalized[:,0], XNormalized[:,1], c=np.argmax(U, axis=0))
plt.scatter(centers[:,0], centers[:,1], c='red', marker='X', label='Cluster Centers')
plt.xlabel('Normalized Paramether 6')
plt.ylabel('Normalized Paramether 2')   
plt.title('FCM Clustering Results')
plt.grid(True)
plt.show()

Centers_unnormalized = centers * (x_max_train - x_min_train) + x_min_train
print("Cluster Centers (Unnormalized):")
print(Centers_unnormalized)


#%% Train all PCAs and LSs models

X_train_labels=np.argmax(U, axis=0)


theats = {}
X_mean_Class = {}
W_class = {}

tau = 0.95

for i in range(c_value):
    print(f"Training PCA + LS for Class {i}...")
    X_train_class = X_train[X_train_labels == i]
    y_train_class = y_train[X_train_labels == i]

    theats[i],X_mean_Class[i],W_class[i] = PCA_LS(X_train_class, y_train_class,tau = tau)


y_hat = np.zeros_like(y_train)

for sample in range(X_train.shape[0]):
    #print(f"Sample {sample}: Assigned Class {X_train_labels[sample]}")

    classSelection = X_train_labels[sample]

    y_hat[sample] = predict(X_train[sample].reshape(1, -1), 
                            theats[classSelection], 
                            X_mean_Class[classSelection], 
                            W_class[classSelection])


#%% Evaluate the training model for the selected class
print("================================")
print("Evaluating the model for the selected class...")
print("================================")
print("\n")
print("Root Mean Square Error:", np.sqrt(np.mean((y_hat - y_train)**2)))



#%% Evaluating on the test set

# matlab notation is {2,6} in python it is {1,5}
X_test_Evaluation = X_test[:, [1,5]] 
X_test_Raw = X_test

# Normalization
XNormalized_Test = (X_test_Evaluation - x_min_train) / (x_max_train - x_min_train)

U_Test = predict_FCM(XNormalized_Test, centers, m = m_value)
X_test_labels=np.argmax(U_Test, axis=0)


y_test_hat = np.zeros_like(y_test)
for sample in range(X_test_Evaluation.shape[0]):
    #print(f"Sample {sample}: Assigned Class {X_test_labels[sample]}")

    classSelection = X_test_labels[sample]

    y_test_hat[sample] = predict(X_test_Raw[sample].reshape(1, -1), 
                            theats[classSelection], 
                            X_mean_Class[classSelection], 
                            W_class[classSelection])

print("================================")
print("Evaluating the model for the selected class...")
print("================================")
print("\n")
print("Root Mean Square Error:", np.sqrt(np.mean((y_test_hat - y_test)**2)))




