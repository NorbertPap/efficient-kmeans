import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_grid_data(K, nl, nh, rl, rh, kg, rn, order='randomized'):
    # Initialize variables
    centers = []
    points = []

    # Calculate center positions for the grid
    grid_size = int(np.sqrt(K))
    distance = kg * (rl + rh) / 2
    for i in range(grid_size):
        for j in range(grid_size):
            x = distance * i
            y = distance * j
            centers.append((x, y))
    
    # Generate points for each cluster
    for center in centers:
        n = np.random.randint(nl, nh + 1)  # Random number of points in the range [nl, nh]
        r = np.random.uniform(rl, rh)  # Random radius in the range [rl, rh]
        cluster_points = np.random.normal(loc=center, scale=r / np.sqrt(2), size=(n, 2))
        points.append(cluster_points)

    # Combine all points
    all_points = np.vstack(points)

    # Add noise points if necessary
    if rn > 0:
        noise_count = int(rn / 100 * len(all_points))
        noise = np.random.uniform(0, grid_size * distance, size=(noise_count, 2))
        all_points = np.vstack((all_points, noise))

    # Randomize order of points if specified
    if order == 'randomized':
        np.random.shuffle(all_points)

    return all_points, np.array(centers)


def generate_sine_data(K, nl, nh, rl, rh, nc, rn, order='randomized'):
    # Initialize variables
    centers = []
    points = []

    # Calculate center positions for sine pattern
    for i in range(K):
        x = 2 * np.pi * i
        y = (K / nc) * np.sin(2 * np.pi * i / (K / nc))
        centers.append((x, y))

    # Generate points for each cluster
    for center in centers:
        n = np.random.randint(nl, nh + 1)  # Random number of points in the range [nl, nh]
        r = np.random.uniform(rl, rh)  # Random radius in the range [rl, rh]
        cluster_points = np.random.normal(loc=center, scale=r / np.sqrt(2), size=(n, 2))
        points.append(cluster_points)

    # Combine all points
    all_points = np.vstack(points)

    # Add noise points if necessary
    if rn > 0:
        noise_count = int(rn / 100 * len(all_points))
        noise = np.random.uniform(0, 2 * np.pi * K, size=(noise_count, 2))
        all_points = np.vstack((all_points, noise))

    # Randomize order of points if specified
    if order == 'randomized':
        np.random.shuffle(all_points)

    return all_points, np.array(centers)


def generate_random_data(K, nl, nh, rl, rh, rn, order='randomized'):
    # Generate random cluster centers
    centers = np.random.uniform(0, K, size=(K, 2))
    points = []

    # Generate points for each cluster
    for center in centers:
        n = np.random.randint(nl, nh + 1)  # Random number of points in the range [nl, nh]
        r = np.random.uniform(rl, rh)  # Random radius in the range [rl, rh]
        cluster_points = np.random.normal(loc=center, scale=r / np.sqrt(2), size=(n, 2))
        points.append(cluster_points)

    # Combine all points
    all_points = np.vstack(points)

    # Add noise points if necessary
    if rn > 0:
        noise_count = int(rn / 100 * len(all_points))
        noise = np.random.uniform(0, 4, size=(noise_count, 2))
        all_points = np.vstack((all_points, noise))

    # Randomize order of points if specified
    if order == 'randomized':
        np.random.shuffle(all_points)

    return all_points, centers


def save_dataset(filename, dataset):
    df = pd.DataFrame(dataset, columns=['x', 'y'])
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")


if __name__ == '__main__':

    # Parameters for the datasets
    K = 16
    nl = 1000
    nh = 1000
    rl = np.sqrt(2)
    rh = np.sqrt(2)
    kg = 4
    nc = 4
    rn = 0  # Noise rate

    # Generate datasets
    ds1, centers1 = generate_grid_data(K, nl, nh, rl, rh, kg, rn)
    ds2, centers2 = generate_sine_data(K, nl, nh, rl, rh, nc, rn)
    ds3, centers3 = generate_random_data(K, 0, 2000, 0, 4, rn)

    # Save datasets
    save_dataset('ds1_grid.csv', ds1)
    save_dataset('ds2_sine.csv', ds2)
    save_dataset('ds3_random.csv', ds3)

    # Optional: Plot the datasets
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(ds1[:, 0], ds1[:, 1], s=1)
    plt.title('Grid Dataset')

    plt.subplot(1, 3, 2)
    plt.scatter(ds2[:, 0], ds2[:, 1], s=1)
    plt.title('Sine Dataset')

    plt.subplot(1, 3, 3)
    plt.scatter(ds3[:, 0], ds3[:, 1], s=1)
    plt.title('Random Dataset')

    plt.tight_layout()
    plt.show()
