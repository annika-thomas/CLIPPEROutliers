# Plot random 2D points with covariance ellipses radiating outwards from a defined center point
# X markers are at random points inside covariance ellipses based on normal distributions in x and y directions
# then rotated the same direction as the ellipse
#
# Written by Annika Thomas
# 6/25/2023

import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import math

# Function to generate random 2D points between 0 and 10 and append to array called points
def generate_random_points(num_points):
    points = []
    for _ in range(num_points):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
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
        height = distance * 0.05

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
        plt.scatter(mapX, mapY, marker='x',color='red')

        # Append map point to the matrix
        map_points.append((mapX, mapY))

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
    print("Distance between points:")
    print(f"Point1: {point1}")
    print(f"Point2: {point2}")
    print(f"Distance: {distance_points}")
    print()
    print("Distance between map points:")
    print(f"Map Point1: {map_point1}")
    print(f"Map Point2: {map_point2}")
    print(f"Distance: {distance_map_points}")
    return abs(distance_points-distance_map_points)
    print(f"Difference: {distance_map_points-distance_points}")

def main():
    num_points = 5
    points = generate_random_points(num_points)
    center_point = (2, 2)  # Change the center point here
    map_points = plot_points(points, center_point)
    print(points)
    print(map_points)

    num_iterations = 1
    total_sum = 0
    
    # Calculates average Euclidean distance between a match and non-match
    for _ in range(num_iterations):
        points = generate_random_points(num_points)
        map_points = plot_points(points, center_point)
        result = calculate_distances(points, map_points, 0, 2)
        total_sum += result

    average = total_sum / num_iterations
    print(f"Average Euclidean difference between match and non-match: {average}")

if __name__ == '__main__':
    main()