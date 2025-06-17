import os
import numpy as np
from skimage import io, color
from scipy.io import loadmat
import matplotlib.pyplot as plt
import mywarper

# Define directories
data_dir = './data'
male_images_dir = os.path.join(data_dir, 'male_images')
male_landmarks_dir = os.path.join(data_dir, 'male_landmarks')
female_images_dir = os.path.join(data_dir, 'female_images')
female_landmarks_dir = os.path.join(data_dir, 'female_landmarks')
output_dir = './results/3_2'
os.makedirs(output_dir, exist_ok=True)

# Load male data
male_base_names = [os.path.splitext(f)[0] for f in os.listdir(male_images_dir) if f.endswith('.jpg')]
male_images = []
male_landmarks = []
for base_name in male_base_names:
    img_path = os.path.join(male_images_dir, base_name + '.jpg')
    landmark_path = os.path.join(male_landmarks_dir, base_name + '.mat')
    img = io.imread(img_path) / 255.0
    img_hsv = color.rgb2hsv(img)
    v_channel = img_hsv[:, :, 2]
    img_3ch = np.stack([v_channel] * 3, axis=-1)
    landmarks = loadmat(landmark_path)['lms'].reshape(68, 2)
    landmarks[:, 1] = 128 - landmarks[:, 1]  # Flip y-coordinates
    male_images.append(img_3ch)
    male_landmarks.append(landmarks)  # Keep as (68, 2)

# Load female data
female_base_names = [os.path.splitext(f)[0] for f in os.listdir(female_images_dir) if f.endswith('.jpg')]
female_images = []
female_landmarks = []
for base_name in female_base_names:
    img_path = os.path.join(female_images_dir, base_name + '.jpg')
    landmark_path = os.path.join(female_landmarks_dir, base_name + '.mat')
    img = io.imread(img_path) / 255.0
    img_hsv = color.rgb2hsv(img)
    v_channel = img_hsv[:, :, 2]
    img_3ch = np.stack([v_channel] * 3, axis=-1)
    landmarks = loadmat(landmark_path)['lms'].reshape(68, 2)
    landmarks[:, 1] = 128 - landmarks[:, 1]  # Flip y-coordinates
    female_images.append(img_3ch)
    female_landmarks.append(landmarks)  # Keep as (68, 2)

# Combine data
images_all = np.array(male_images + female_images)
landmarks_all = np.array(male_landmarks + female_landmarks)
labels_all = np.array([0] * len(male_images) + [1] * len(female_images))

# Shuffle and split into training (800) and testing (200)
np.random.seed(42)
indices = np.random.permutation(len(images_all))
train_indices = indices[:800]
test_indices = indices[800:]
images_train = images_all[train_indices]
landmarks_train = landmarks_all[train_indices]
labels_train = labels_all[train_indices]
images_test = images_all[test_indices]
landmarks_test = landmarks_all[test_indices]
labels_test = labels_all[test_indices]

# PCA for landmarks (reduce to 10 dimensions)
mean_landmarks = np.mean(landmarks_train, axis=0)
landmarks_train_centered = landmarks_train - mean_landmarks
U_warp, S_warp, _ = np.linalg.svd(landmarks_train_centered.reshape(800, 136).T, full_matrices=False)
eigen_warpings = U_warp[:, :10]
landmarks_train_pca = eigen_warpings.T @ landmarks_train_centered.reshape(800, 136).T  # (10, 800)
landmarks_test_centered = landmarks_test - mean_landmarks
landmarks_test_pca = eigen_warpings.T @ landmarks_test_centered.reshape(200, 136).T  # (10, 200)

# Align images to mean landmarks for appearance
aligned_images_train = np.array([mywarper.warp(img, landmarks_train[i], mean_landmarks) for i, img in enumerate(images_train)])
aligned_images_test = np.array([mywarper.warp(img, landmarks_test[i], mean_landmarks) for i, img in enumerate(images_test)])

# PCA for images (reduce to 50 dimensions)
images_train_flat = aligned_images_train.reshape(800, -1)  # (800, 49152)
mean_face = np.mean(images_train_flat, axis=0)
images_train_centered = images_train_flat - mean_face
U_face, S_face, _ = np.linalg.svd(images_train_centered.T, full_matrices=False)
eigen_faces = U_face[:, :50]
images_train_pca = eigen_faces.T @ images_train_centered.T  # (50, 800)
images_test_flat = aligned_images_test.reshape(200, -1)
images_test_centered = images_test_flat - mean_face
images_test_pca = eigen_faces.T @ images_test_centered.T  # (50, 200)

# FLD for landmarks (geometry)
mu_land_0 = np.mean(landmarks_train_pca[:, labels_train == 0], axis=1)
mu_land_1 = np.mean(landmarks_train_pca[:, labels_train == 1], axis=1)
S_w_land = np.zeros((10, 10))
for i in range(800):
    if labels_train[i] == 0:
        S_w_land += np.outer(landmarks_train_pca[:, i] - mu_land_0, landmarks_train_pca[:, i] - mu_land_0)
    else:
        S_w_land += np.outer(landmarks_train_pca[:, i] - mu_land_1, landmarks_train_pca[:, i] - mu_land_1)
W_land = np.linalg.solve(S_w_land + 1e-5 * np.eye(10), mu_land_1 - mu_land_0)
W_land = W_land / np.linalg.norm(W_land)
W_land_2d = np.column_stack([W_land.real, np.zeros(10)])  # First direction, second is zero for 2D

# FLD for images (appearance)
mu_face_0 = np.mean(images_train_pca[:, labels_train == 0], axis=1)
mu_face_1 = np.mean(images_train_pca[:, labels_train == 1], axis=1)
S_w_face = np.zeros((50, 50))
for i in range(800):
    if labels_train[i] == 0:
        S_w_face += np.outer(images_train_pca[:, i] - mu_face_0, images_train_pca[:, i] - mu_face_0)
    else:
        S_w_face += np.outer(images_train_pca[:, i] - mu_face_1, images_train_pca[:, i] - mu_face_1)
W_face = np.linalg.solve(S_w_face + 1e-5 * np.eye(50), mu_face_1 - mu_face_0)
W_face = W_face / np.linalg.norm(W_face)
evals, evecs = np.linalg.eigh(np.outer(W_face, W_face))  # Get eigenvalues and eigenvectors
idx = np.argsort(evals)[::-1]  # Sort in descending order
W_face_2d = evecs[:, idx[:2]]  # Take top 2 eigenvectors

# Project to 2D feature space
proj_land_train = W_land_2d.T @ landmarks_train_pca
proj_land_test = W_land_2d.T @ landmarks_test_pca
proj_face_train = W_face_2d.T @ images_train_pca
proj_face_test = W_face_2d.T @ images_test_pca

# Combine training and test data for plotting
proj_land_all = np.hstack([proj_land_train, proj_land_test])
proj_face_all = np.hstack([proj_face_train, proj_face_test])
labels_all = np.hstack([labels_train, labels_test])

# Visualize separability
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(proj_land_all[0, labels_all == 0], proj_land_all[1, labels_all == 0], c='blue', alpha=0.5, label='Male')
plt.scatter(proj_land_all[0, labels_all == 1], proj_land_all[1, labels_all == 1], c='red', alpha=0.5, label='Female')
plt.xlabel('FLD Direction 1 (Geometry)')
plt.ylabel('FLD Direction 2 (Geometry)')
plt.title('2D Projection (Geometry)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(proj_face_all[0, labels_all == 0], proj_face_all[1, labels_all == 0], c='blue', alpha=0.5, label='Male')
plt.scatter(proj_face_all[0, labels_all == 1], proj_face_all[1, labels_all == 1], c='red', alpha=0.5, label='Female')
plt.xlabel('FLD Direction 1 (Appearance)')
plt.ylabel('FLD Direction 2 (Appearance)')
plt.title('2D Projection (Appearance)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fld_2d_projections.png'))
plt.close()