import numpy as np
import matplotlib.pyplot as plt
import time

class KDNode:
    def __init__(self, points, depth=0):
        self.points = points
        self.depth = depth
        self.left = None
        self.right = None
        self.split_dim = depth % points.shape[1]
        self.split_value = np.median(points[:, self.split_dim])
        self.n_points = len(points)
        self.sum = np.sum(points, axis=0)
        self.sum_sq = np.sum(points ** 2, axis=0)

    def build_tree(self):
        if len(self.points) <= 64:  # Leaf size
            return

        left_mask = self.points[:, self.split_dim] < self.split_value
        right_mask = ~left_mask

        self.left = KDNode(self.points[left_mask], self.depth + 1)
        self.right = KDNode(self.points[right_mask], self.depth + 1)

        self.left.build_tree()
        self.right.build_tree()

        self.points = None  # Free memory

def build_kd_tree(points):
    root = KDNode(points)
    root.build_tree()
    return root

def min_max_dist(centroid, node):
    min_dist = np.sum((np.maximum(centroid, node.sum / node.n_points - node.split_value) -
                       np.minimum(centroid, node.sum / node.n_points + node.split_value)) ** 2)
    max_corner = np.where(centroid > node.sum / node.n_points, 
                          node.sum / node.n_points + node.split_value, 
                          node.sum / node.n_points - node.split_value)
    max_dist = np.sum((centroid - max_corner) ** 2)
    return np.sqrt(min_dist), np.sqrt(max_dist)

def traverse_tree(node, centroids, cluster_stats):
    if node.points is not None:
        distances = np.sqrt(((node.points[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        for i in range(len(centroids)):
            mask = labels == i
            cluster_stats[i]['n'] += np.sum(mask)
            cluster_stats[i]['sum'] += np.sum(node.points[mask], axis=0)
            cluster_stats[i]['sum_sq'] += np.sum(node.points[mask] ** 2, axis=0)
        return

    min_dists, max_dists = zip(*[min_max_dist(centroid, node) for centroid in centroids])
    min_max = min(max_dists)
    candidates = [i for i, min_dist in enumerate(min_dists) if min_dist <= min_max]

    if len(candidates) == 1:
        i = candidates[0]
        cluster_stats[i]['n'] += node.n_points
        cluster_stats[i]['sum'] += node.sum
        cluster_stats[i]['sum_sq'] += node.sum_sq
    else:
        traverse_tree(node.left, centroids[candidates], [cluster_stats[i] for i in candidates])
        traverse_tree(node.right, centroids[candidates], [cluster_stats[i] for i in candidates])

def efficient_kmeans(X, k, max_iters=100, tol=1e-4):
    n, d = X.shape
    
    # Build k-d tree
    root = build_kd_tree(X)
    
    # Initialize centroids randomly
    centroids = X[np.random.choice(n, k, replace=False)]
    
    for _ in range(max_iters):
        cluster_stats = [{'n': 0, 'sum': np.zeros(d), 'sum_sq': np.zeros(d)} for _ in range(k)]
        
        # Traverse tree and update cluster statistics
        traverse_tree(root, centroids, cluster_stats)
        
        # Update centroids
        new_centroids = np.array([stats['sum'] / stats['n'] for stats in cluster_stats])
        
        # Check for convergence
        if np.allclose(centroids, new_centroids, rtol=tol):
            break
        
        centroids = new_centroids
    
    # Calculate error
    error = sum(np.sum(stats['sum_sq']) - np.sum(stats['sum']**2) / stats['n'] for stats in cluster_stats)
    
    # Assign labels to points
    distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
    labels = np.argmin(distances, axis=1)
    
    return centroids, labels, error


def direct_kmeans(X, k, max_iters=100, tol=1e-4):
    n, d = X.shape
    
    # Initialize centroids randomly
    centroids = X[np.random.choice(n, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.allclose(centroids, new_centroids, rtol=tol):
            break
        
        centroids = new_centroids
    
    # Calculate error
    error = np.sum((X - centroids[labels]) ** 2)
    
    return centroids, labels, error

def visualize_clusters(X, centroids, labels, title):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(ticks=range(len(centroids)), label='Cluster')
    plt.show()

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(10000, 2)
    k = 20

    # Direct K-Means
    start = time.time()
    direct_centroids, direct_labels, direct_error = direct_kmeans(X, k)
    end = time.time()
    print("Direct K-Means Error:", direct_error, "Time elapsed:", end - start)
    visualize_clusters(X, direct_centroids, direct_labels, "Direct K-Means Clustering")

    # Efficient K-Means
    start = time.time()
    efficient_centroids, efficient_labels, efficient_error = efficient_kmeans(X, k)
    end = time.time()
    print("Efficient K-Means Error:", efficient_error, "Time elapsed:", end - start)
    visualize_clusters(X, efficient_centroids, efficient_labels, "Efficient K-Means Clustering")