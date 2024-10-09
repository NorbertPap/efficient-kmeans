import matplotlib.pyplot as plt
import numpy as np
import math
import time
import random

DIMENSION_SELECT = 0 # 0 for round robin, 1 for max variance
SPLIT_VALUE_SELECT = 0 # 0 for midpoint, 1 for median
LEAF_SIZE = 64
D = 6 # Dimensionality

class KDNode():
    def __init__(self, points, ranges, depth=0, axis = None):
        self.points = points # N x D, where N is number of points and D is dimensionality
        self.depth = depth
        self.left = None
        self.right = None
        self.axis = axis
        self.split_value = None
        self.ranges = ranges # D x 2, where D is dimensionality and 2 is min and max

        self.n_points = len(points)
        self.lin_sum = np.zeros(D)
        for i in range(len(points)):
            self.lin_sum += points[i]
        self.squared_sum = np.zeros(D)
        for i in range(len(points)):
            self.squared_sum += points[i] ** 2


    def build_tree(self):
        if (len(self.points) < LEAF_SIZE):
            return
        
        # Round-robin
        if (DIMENSION_SELECT == 0):
            self.axis = self.depth % self.points.shape[1]
        # Max variance
        elif (DIMENSION_SELECT == 1):
            max_variance = 0
            for i in range(len(self.points.shape[0])):
                variance = self.points[i][1] - self.points[i][0]
                if variance > max_variance:
                    max_variance = variance
                    self.axis = i

        # Midpoint
        if (SPLIT_VALUE_SELECT == 0):
            self.split_value = (self.ranges[self.axis][0] + self.ranges[self.axis][1]) / 2
        # Median
        elif(SPLIT_VALUE_SELECT == 1):
            sorted = quicksort(self.points, self.axis)
            self.split_value = sorted[len(sorted) // 2][self.axis]

        # Split the data into left and right groups
        left_points = np.array([point for point in self.points if point[self.axis] <= self.split_value])
        left_ranges = np.array(self.ranges)
        left_ranges[self.axis][1] = self.split_value
        right_points = np.array([point for point in self.points if point[self.axis] > self.split_value])
        right_ranges = np.array(self.ranges)
        right_ranges[self.axis][0] = self.split_value

        self.left = KDNode(left_points, left_ranges, self.depth + 1)
        self.right = KDNode(right_points, right_ranges, self.depth + 1)

        self.left.build_tree()
        self.right.build_tree()

        self.point = None


def build_kd_tree(points, ranges):
    root = KDNode(points=points, ranges=ranges)
    root.build_tree()
    return root


def quicksort(arr, axis):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) * np.random.rand()][axis]
    left = [x for x in arr if x[axis] < pivot[axis]]
    middle = [x for x in arr if x[axis] == pivot[axis]]
    right = [x for x in arr if x[axis] > pivot[axis]]
    return quicksort(left, axis) + middle + quicksort(right, axis)


def min_max_dist(centroid, node):
    min_dist = sum((max(0, max(node.ranges[i][0] - centroid[i], centroid[i] - node.ranges[i][1])) ** 2 for i in range(len(centroid)))) ** 0.5
    # Maximum distance calculation
    # calculate which corner is the furthest distance away from the centroid
    max_corner = [node.ranges[i][0] if abs(centroid[i] - node.ranges[i][0]) > abs(centroid[i] - node.ranges[i][1]) else node.ranges[i][1] for i in range(len(centroid))]
    max_dist = sum(
        (centroid[i] - max_corner[i]) ** 2 for i in range(len(centroid))
    ) ** 0.5
    return min_dist, max_dist


def traverse_tree(node, centroids, cluster_stats, points_per_centroid):
    if node.points is not None: # Leaf node
        distances = [[math.sqrt(sum((p[i] - c[i]) ** 2 for i in range(len(p)))) for c in centroids] for p in node.points]
        labels = [dist.index(min(dist)) for dist in distances]
        for i in range(len(centroids)):
            mask = [j for j, label in enumerate(labels) if label == i]
            cluster_stats[i]['n'] += len(mask)
            cluster_stats[i]['sum'] = [cluster_stats[i]['sum'][d] + sum(node.points[m][d] for m in mask) for d in range(len(centroids[0]))]
            cluster_stats[i]['sum_sq'] = [cluster_stats[i]['sum_sq'][d] + sum(node.points[m][d] ** 2 for m in mask) for d in range(len(centroids[0]))]
            points_per_centroid[i].extend(node.points[m] for m in mask)
        return

    min_dists, max_dists = zip(*[min_max_dist(centroid, node) for centroid in centroids])
    min_max = min(max_dists)
    candidates = [i for i, min_dist in enumerate(min_dists) if min_dist <= min_max]

    if len(candidates) == 1:
        i = candidates[0]
        cluster_stats[i]['n'] += node.n_points
        cluster_stats[i]['sum'] = [cluster_stats[i]['sum'][d] + node.lin_sum[d] for d in range(len(centroids[0]))]
        cluster_stats[i]['sum_sq'] = [cluster_stats[i]['sum_sq'][d] + node.squared_sum[d] for d in range(len(centroids[0]))]
        points_per_centroid[i].extend(node.points)
    else:
        traverse_tree(node.left, centroids, cluster_stats, points_per_centroid)
        traverse_tree(node.right, centroids, cluster_stats, points_per_centroid)


def efficient_kmeans(X, k, max_iters=100, tol=1e-4):
    start_time = time.time()
    n, d = X.shape
    
    # Build k-d tree
    root = build_kd_tree(X, np.array([[0.0, 1.0]] * d))
    
    # Initialize centroids randomly
    centroids = X[np.random.choice(n, k, replace=False)]
    
    for _ in range(max_iters):
        cluster_stats = [{'n': 0, 'sum': [0.0] * d, 'sum_sq': [0.0] * d} for _ in range(k)]
        points_per_centroid = [[] for _ in range(k)]
        
        # Traverse tree and update cluster statistics
        traverse_tree(root, centroids, cluster_stats, points_per_centroid)
        
        # Update centroids
        new_centroids = []
        for stats in cluster_stats:
            if stats['n'] > 0:
                new_centroids.append([s / stats['n'] for s in stats['sum']])
            else:
                new_centroids.append([0.0] * d)
        
        # Check for convergence
        if all(math.isclose(centroids[i][j], new_centroids[i][j], rel_tol=tol) for i in range(k) for j in range(d)):
            break
        
        centroids = new_centroids
    
    # Calculate error
    error = sum(sum(stats['sum_sq'][j] - (stats['sum'][j] ** 2) / stats['n'] for j in range(d)) for stats in cluster_stats if stats['n'] > 0)

    print("Time to run efficient kmeans:", time.time() - start_time)
    
    start_time = time.time()
    # Assign labels to points: the points per centroid are a list of clusters corresponding to each centroid
    labels = np.zeros(n, dtype=int)
    for i, points in enumerate(points_per_centroid):
        for point in points:
            labels[np.where((X == point).all(axis=1))] = i

    print("Time to assign labels:", time.time() - start_time)
    
    
    return centroids, labels, error
    



def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def mean(points):
    n = len(points)
    if n == 0:
        return []
    d = len(points[0])
    return [sum(point[i] for point in points) / n for i in range(d)]

def direct_kmeans(X, k, max_iters=100, tol=1e-4):
    n = len(X)
    d = len(X[0])

    start_time = time.time()
    
    # Initialize centroids randomly
    centroids = X[np.random.choice(n, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        labels = []
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            labels.append(distances.index(min(distances)))
        
        # Update centroids
        new_centroids = []
        for i in range(k):
            cluster_points = [X[j] for j in range(n) if labels[j] == i]
            new_centroids.append(mean(cluster_points))
        
        # Check for convergence
        if all(euclidean_distance(centroids[i], new_centroids[i]) < tol for i in range(k)):
            break
        
        centroids = new_centroids
    
    print("Time to run direct kmeans:", time.time() - start_time)
    # Calculate error
    error = sum(euclidean_distance(X[i], centroids[labels[i]]) ** 2 for i in range(n))

    return centroids, labels, error


if __name__ == '__main__':
    k = 10
    # generate 1000 random points in 2D
    X = np.random.rand(100000, D)
    centroids, labels, error = efficient_kmeans(X, k)
    print("Error:", error)
    plt.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.7)
    plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], c='red', marker='x', s=200, linewidths=3)
    plt.title("Efficient K-Means Clustering")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(ticks=range(k), label='Cluster')
    plt.show()

    centroids, labels, error = direct_kmeans(X, k)
    print("Error:", error)
    plt.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.7)
    plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], c='red', marker='x', s=200, linewidths=3)
    plt.title("Direct K-Means Clustering")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(ticks=range(k), label='Cluster')
    plt.show()