import numpy as np
import matplotlib.pyplot as plt
import sys


def generate_dataset(d=2, k=5, n=500, cube_size=1.0, cluster_radius=0.05, seed=None):
    """
    Generates a dataset with k clusters in a d-dimensional cube.

    Parameters:
    - d (int): Dimensionality of the cube.
    - k (int): Number of clusters.
    - n (int): Total number of points.
    - cube_size (float): Size of the cube in each dimension.
    - cluster_radius (float): Radius around each base point to generate clustered points.
    - seed (int): Random seed for reproducibility.

    Returns:
    - data (np.ndarray): Generated dataset of shape (n, d).
    - labels (np.ndarray): Cluster labels for each point.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate points per cluster
    points_per_cluster = []
    total_assigned = 0
    for i in range(1, k + 1):
        num_points = int(i * (2 * n) / ((k + 1) * k))
        points_per_cluster.append(num_points)
        total_assigned += num_points
    
    # Adjust the last cluster to ensure total points = n
    if total_assigned < n:
        points_per_cluster[-1] += n - total_assigned
    elif total_assigned > n:
        points_per_cluster[-1] -= total_assigned - n
    
    # Generate base points
    base_points = np.random.uniform(0, cube_size, size=(k, d))

    data = []
    labels = []
    
    for idx, (base, num_points) in enumerate(zip(base_points, points_per_cluster)):
        # Generate points around the base point
        # Uniformly within a hypercube of side length 2*cluster_radius centered at base
        points = np.random.uniform(-cluster_radius, cluster_radius, size=(num_points, d)) + base
        print(points)
        # Ensure points are within the cube
        points = np.clip(points, 0, cube_size)
        data.append(points)
        labels.extend([idx] * num_points)
    
    data = np.vstack(data)
    labels = np.array(labels)
    
    return data, labels

# Example Usage
if __name__ == "__main__":
    # Parameters
    dimensionality = 2
    num_clusters = 16
    total_points = 500
    cube_dimension_size = 1.0
    radius = 0.05
    random_seed = 42  # For reproducibility

    # Generate dataset
    dataset, cluster_labels = generate_dataset(
        d=dimensionality,
        k=num_clusters,
        n=total_points,
        cube_size=cube_dimension_size,
        cluster_radius=radius,
        seed=random_seed
    )

    # Visualization (for 2D)
    if dimensionality == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(dataset[:, 0], dataset[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        plt.title('Generated Dataset with Non-Uniform Clusters')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.colorbar(scatter, label='Cluster Label')
        plt.show()