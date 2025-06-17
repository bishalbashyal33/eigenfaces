import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Directory paths
landmarks_dir = './data/landmarks'
output_dir = './figures'
os.makedirs(output_dir, exist_ok=True)  # Create figures directory if it doesn't exist

# Get all landmark file paths (assuming .mat format)
landmark_files = [os.path.join(landmarks_dir, f) for f in os.listdir(landmarks_dir) if f.endswith('.mat')]

# Verify we have at least 1000 landmark files
if len(landmark_files) < 1000:
    print(f"Warning: Found only {len(landmark_files)} landmark files. At least 1000 are expected.")
else:
    print(f"Found {len(landmark_files)} landmark files. Proceeding with train-test split...")

# Function to load landmarks from .mat files (assuming 68 points with x, y coordinates)
def load_landmarks(file_path):
    mat_data = loadmat(file_path)
    # Debug: Print available keys to verify
    print(f"Keys in {file_path}: {mat_data.keys()}")
    landmarks = mat_data.get('lms')  # Use 'lms' as the key based on debug output
    if landmarks is None:
        raise ValueError(f"No 'lms' data found in {file_path}")
    # Handle potential nested structure or different shape
    if landmarks.shape == (68, 2):  # Direct (68, 2) array
        landmarks_flipped = landmarks.copy()
        landmarks_flipped[:, 1] = 128 - landmarks_flipped[:, 1]  # Flip y-coordinates
        return landmarks_flipped.flatten()
    elif landmarks.shape[0] == 136 and landmarks.ndim == 1:  # Flattened (136,) array
        landmarks_reshaped = landmarks.reshape(68, 2)
        landmarks_flipped = landmarks_reshaped.copy()
        landmarks_flipped[:, 1] = 128 - landmarks_flipped[:, 1]  # Flip y-coordinates
        return landmarks_flipped.flatten()
    elif landmarks.shape[0] == 68 and landmarks.shape[1] == 2:  # Transposed or nested
        landmarks_flipped = landmarks.copy()
        landmarks_flipped[:, 1] = 128 - landmarks_flipped[:, 1]  # Flip y-coordinates
        return landmarks_flipped.T.flatten()
    else:
        raise ValueError(f"Invalid landmark data in {file_path}: expected (68, 2) or (136,) array, got {landmarks.shape}")

# Load all landmarks
print("Loading all landmarks...")
landmarks_all = np.array([load_landmarks(f) for f in landmark_files]).T  # Shape: (136, n_files)

# Randomly shuffle and split into train (800) and test (200) sets
np.random.seed(42)  # For reproducibility
indices = np.random.permutation(landmarks_all.shape[1])
train_indices = indices[:800]
test_indices = indices[800:1000]

landmarks_train = landmarks_all[:, train_indices]  # Shape: (136, 800)
landmarks_test = landmarks_all[:, test_indices]    # Shape: (136, 200)

# Compute the mean landmark position from training data
mean_landmarks = np.mean(landmarks_train, axis=1)  # Shape: (136,)

# Center the data by subtracting the mean landmarks
landmarks_train_centered = landmarks_train - mean_landmarks[:, np.newaxis]
landmarks_test_centered = landmarks_test - mean_landmarks[:, np.newaxis]

# Perform SVD on centered training data to get eigen-warpings
print("Computing SVD for eigen-warpings...")
U, S, Vt = np.linalg.svd(landmarks_train_centered, full_matrices=False)  # U: (136, 800), eigen-warpings are columns of U

# Function to reconstruct landmarks using top K eigen-warpings
def reconstruct_landmarks(X_centered, U, K):
    coeffs = U[:, :K].T @ X_centered              # Project onto top K eigen-warpings
    X_reconstructed = U[:, :K] @ coeffs           # Reconstruct
    return X_reconstructed

# Display the first 10 eigen-warpings (add mean for meaningful visualization)
plt.figure(figsize=(10, 5))
for i in range(10):
    eigen_warping = U[:, i].reshape(68, 2) + mean_landmarks.reshape(68, 2)  # Add mean to eigen-warping
    plt.subplot(2, 5, i + 1)
    plt.scatter(eigen_warping[:, 0], eigen_warping[:, 1], s=10, c='b')
    plt.title(f'Eigen-Warping {i + 1}')
    plt.axis('equal')
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eigen_warpings.png'))

# Reconstruct test landmarks with K=50
# Reconstruct test landmarks with K=50
K = 50
landmarks_test_reconstructed = reconstruct_landmarks(landmarks_test_centered, U, K) + mean_landmarks[:, np.newaxis]

# Create a single plot with 10 original and 10 reconstructed landmarks
plt.figure(figsize=(15, 20))  # Wider and taller to accommodate 20 subplots
for i in range(10):  # First 10 test samples (indices 0 to 9)
    # Original landmarks
    orig_landmarks = landmarks_test[:, i].reshape(68, 2)
    plt.subplot(4, 5, i + 1)  # Top row (1 to 10)
    plt.scatter(orig_landmarks[:, 0], orig_landmarks[:, 1], s=10, c='b', label='Original')
    plt.title(f'Test {i + 801} Original')
    plt.axis('equal')
    plt.grid(True)
    if i == 0:
        plt.legend()

    # Reconstructed landmarks
    recon_landmarks = landmarks_test_reconstructed[:, i].reshape(68, 2)
    plt.subplot(4, 5, i + 11)  # Bottom row (11 to 20)
    plt.scatter(recon_landmarks[:, 0], recon_landmarks[:, 1], s=10, c='r', label='Reconstructed')
    plt.title(f'Test {i + 801} Reconstructed')
    plt.axis('equal')
    plt.grid(True)
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'test_landmarks_801_810.png'))
plt.close()


# Compute and plot reconstruction error for K = 1, 5, 10, ..., 50
K_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
errors = []
print("Computing reconstruction errors...")
for K in K_values:
    landmarks_test_reconstructed_k = reconstruct_landmarks(landmarks_test_centered, U, K) + mean_landmarks[:, np.newaxis]
    error = np.mean(np.sum((landmarks_test - landmarks_test_reconstructed_k) ** 2, axis=0))  # Average Euclidean distance
    errors.append(error)

plt.figure(figsize=(8, 6))
plt.plot(K_values, errors, marker='o')
plt.xlabel('Number of Eigen-Warpings (K)')
plt.ylabel('Average Reconstruction Error (Distance)')
plt.title('Reconstruction Error vs. Number of Eigen-Warpings')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'reconstruction_error.png'))