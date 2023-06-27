import time
import random
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation
import clipperpy
import math

# Function to generate random 2D points between 0 and 10 and append to array called points
def generate_random_points(num_points):
    points = []
    for _ in range(num_points):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        points.append((x, y))
    return points

# Plots points with covariance ellipses, a box at the center point, and X markers at random
# points inside the covariance ellipses indicating the actual map points
def plot_points(points, center):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    map_points = []

    for point in points:
        x, y = point

        # Calculate distance from the center point
        distance = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5

        # Calculate the width and height of the ellipse based on the distance
        width = distance * 0.5
        height = distance * 0.01

        # Calculate the angle between the point and the center point
        angle = np.arctan2(y - center[1], x - center[0])

        ellipse = Ellipse((x, y), width, height, angle=np.degrees(angle), color='blue', alpha=0.2)
        ax.add_patch(ellipse)

        # Plot a small box at the center point
        center_box = plt.Rectangle((center[0] - 0.1, center[1] - 0.1), 0.2, 0.2, color='red')
        ax.add_patch(center_box)

        # Random distances for X marker to be in bounds of covariance ellipse
        distWidth = random.normalvariate(0, 0.25*width)
        distHeight = random.normalvariate(0, 0.25*height)

        # Apply the rotation transformation
        rotated_W = distWidth * math.cos(angle) - distHeight * math.sin(angle)
        rotated_H = distWidth * math.sin(angle) + distHeight * math.cos(angle)

        # Map points
        mapX = rotated_W + x
        mapY = rotated_H + y

        # Plot an 'X' marker at the map point
        plt.scatter(mapX+110, mapY, marker='x',color='red')

        # Append map point to the matrix
        map_points.append((mapX+110, mapY))

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.scatter(x, y, label='Measured points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Random 2D Points Map')
    plt.legend()
    plt.show()
    return map_points

# Function to calculate Euclidean distances between matched points in vehicle map and global map
# Returns difference between distances
# To change to a non-match, add one to index1 in line 81 (defining point1)
def calculate_distances(points, map_points, index1, index2):
    # Extract the points of interest from the arrays
    point1 = np.array(points[index1])
    point2 = np.array(points[index2])
    map_point1 = np.array(map_points[index1])
    map_point2 = np.array(map_points[index2])

    # Calculate the distance between the points
    distance_points = np.linalg.norm(point2 - point1)
    distance_map_points = np.linalg.norm(map_point2 - map_point1)

    # Print the distances
    #print("Distance between points:")
    #print(f"Point1: {point1}")
    #print(f"Point2: {point2}")
    #print(f"Distance: {distance_points}")
    #print()
    #print("Distance between map points:")
    #print(f"Map Point1: {map_point1}")
    #print(f"Map Point2: {map_point2}")
    #print(f"Distance: {distance_map_points}")
    return abs(distance_points-distance_map_points)
    #print(f"Difference: {distance_map_points-distance_points}")

def get_init_associations(n1, n2o, outrat, m):
    
    n2 = n1 + n2o # number of points in view 2
    noa = round(m * outrat) # number of outlier associations
    nia = m - noa # number of inlier associations

    # Correct associations to draw from
    Agood = np.tile(np.arange(n1).reshape(-1,1),(1,2))

    # Incorrect association to draw from
    Abad = np.zeros((n1*n2 - n1, 2))
    itr = 0
    for i in range(n1):
        for j in range(n2):
            if i == j:
                continue
            Abad[itr,:] = [i, j]
            itr += 1

    # Sample good and bad associations to satisfy total
    # num of associations with the requested outlier ratio
    IAgood = np.random.choice(Agood.shape[0], nia, replace=False)
    IAbad = np.random.choice(Abad.shape[0], noa, replace=False)

    A = np.concatenate((Agood[IAgood,:],Abad[IAbad,:])).astype(np.int32)

    # Ground truth associations
    Agt = Agood[IAgood,:]
    #print(f"Associations are: {A}")
    
    return (Agt, A)

def plot_initial_associations(A, Ain, points, map_points):
    
    # Convert lists to NumPy arrays if needed
    points1 = np.array(points)
    points2 = np.array(map_points)

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot the points from the first dataset
    ax.scatter(points1[:, 0], points1[:, 1], color='red', label='Dataset 1')

    # Plot the points from the second dataset
    ax.scatter(points2[:, 0], points2[:, 1], color='blue', label='Dataset 2')

    for line in A:
        start_idx, end_idx = line
        start_point = points1[start_idx]
        end_point = points2[end_idx]
        if (start_idx == end_idx):
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='green')
        else:
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')

    for line in Ain:
        start_idx, end_idx = line
        start_point = points1[start_idx]
        end_point = points2[end_idx]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], linestyle=(0,(5,10)), color='blue')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Points')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


def clipper(D1, D2, A, Agt):
    iparams = clipperpy.invariants.EuclideanDistanceParams()
    iparams.sigma = 0.01
    iparams.epsilon = 0.02
    invariant = clipperpy.invariants.EuclideanDistance(iparams)

    params = clipperpy.Params()
    params.rounding = clipperpy.Rounding.DSD_HEU
    clipper = clipperpy.CLIPPER(invariant, params)

    #t0 = time.perf_counter()
    #points1 = np.array(D1)
    points1 = np.array(D1, dtype=np.float64)
    #print(points1.shape)
    #print(A.shape)

    #points2 = np.array(D2)
    points2 = np.array(D2, dtype=np.float64)

    A = np.array(A, dtype=np.int32)

    print(points1.shape)
    print(points2.shape)
    print(A.shape)

    # ISSUE IS HERE:
    clipper.score_pairwise_consistency(points1, points2, A.astype(np.int64))
    #t1 = time.perf_counter()
    #print(f"Affinity matrix creation took {t1-t0:.3f} seconds")

    t0 = time.perf_counter()
    clipper.solve()
    t1 = time.perf_counter()

    A = clipper.get_initial_associations()
    Ain = clipper.get_selected_associations()

    #p = np.isin(Ain, Agt)[:,0].sum() / Ain.shape[0]
    #r = np.isin(Ain, Agt)[:,0].sum() / Agt.shape[0]
    print(f"CLIPPER selected {Ain.shape[0]} inliers from {A.shape[0]} ")
    #      f"putative associations (precision {p:.2f}, recall {r:.2f}) in {t1-t0:.3f} s")

    return Ain


def main():
    num_points = 200
    points = generate_random_points(num_points)
    center_point = (10, 20)  # Change the center point here
    map_points = plot_points(points, center_point)
    #print(points)
    #print(map_points)

    n1 = 200         # number of points used on model
    n2o = 0         # number of outliers in data
    outrat = 0.8    # outlier ratio of initial association set
    m = 50          # total number of associations in problem
    Agt, A = get_init_associations(n1, n2o, outrat, m)

    #plot_initial_associations(A, points, map_points)

    Aclip = clipper(points, map_points, A, Agt)

    plot_initial_associations(A, Aclip, points, map_points)

    print(A)

    print(Aclip)

    # Counter
    counter = 0

    # Check if rows of array1 exist in rows of array2
    matching_rows = np.all(Aclip[:, None] == A, axis=2)
    rows_exist = np.any(matching_rows, axis=1)

    # Count the number of rows in array1 that exist in array2
    counter = np.count_nonzero(rows_exist)

    #print(counter)
    counter_exist = 0
    counter_same_elements = 0

    # Iterate over rows in arrayA
    for row in A:
        # Check if row exists in arrayAclip
        if np.any(np.all(row == Aclip, axis=1)):
            counter_exist += 1

        # Check if elements of the row are the same in both arrays
        if np.any(np.all(row == Aclip, axis=1)) and np.all(row == row[0]):
            counter_same_elements += 1

    print("Total Associations from CLIPPER:", counter_exist)
    print("Correct Associations from CLIPPER:", counter_same_elements)
    print("Accuracy of CLIPPER at ", outrat, " outlier ratio: ", counter_same_elements/counter_exist)



    #num_iterations = 0
    #total_sum = 0
    
    # Calculates average Euclidean distance between a match and non-match
    #for _ in range(num_iterations):
    #    points = generate_random_points(num_points)
    #    map_points = plot_points(points, center_point)
    #    result = calculate_distances(points, map_points, 0, 2)
    #    total_sum += result

    #average = total_sum / num_iterations
    #print(f"Average Euclidean difference between match and non-match: {average}")

if __name__ == '__main__':
    main()