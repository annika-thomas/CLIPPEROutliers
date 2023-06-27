import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def generate_random_points(num_points):
    points = []
    for _ in range(num_points):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        points.append((x, y))
    return points

def plot_points(points, center):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for point in points:
        x, y = point

        # Calculate distance from the center point
        distance = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5

        # Calculate the width and height of the ellipse based on the distance
        width = distance * 0.5
        height = distance * 1.5

        # Calculate the angle between the point and the center point
        angle = np.arctan2(y - center[1], x - center[0])

        ellipse = Ellipse((x, y), width, height, angle=np.degrees(angle), color='blue', alpha=0.2)
        ax.add_patch(ellipse)

        # Plot a small box at the center point
        center_box = plt.Rectangle((center[0] - 0.1, center[1] - 0.1), 0.2, 0.2, color='red')
        ax.add_patch(center_box)

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Random 2D Points with Covariance Ellipses')
    plt.show()

def plot_random_points_inside_ellipses(points, center):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for point in points:
        x, y = point

        # Calculate distance from the center point
        distance = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5

        # Calculate the width and height of the ellipse at the center point
        center_width = distance * 0.5
        center_height = distance * 1.5

        # Calculate the angle between the point and the center point
        angle = np.arctan2(y - center[1], x - center[0])

        # Generate a random point inside the ellipse
        angle_rad = random.uniform(0, 2 * np.pi)
        radius = random.uniform(0, 1)

        # Calculate the width and height of the ellipse at the random point
        width = center_width * radius
        height = center_height * radius

        # Calculate the coordinates of the random point inside the ellipse
        x_inside = x + width * np.cos(angle_rad + angle)
        y_inside = y + height * np.sin(angle_rad + angle)

        # Plot the random point
        plt.scatter(x_inside, y_inside, marker='x', color='red')

        # Plot the ellipse at the center point
        ellipse = Ellipse((x, y), center_width, center_height, angle=np.degrees(angle), color='blue', alpha=0.2)
        ax.add_patch(ellipse)

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Random 2D Points inside Covariance Ellipses')
    plt.show()

def main():
    num_points = 3
    points = generate_random_points(num_points)
    center_point = (2, 2)  # Change the center point here
    plot_points(points, center_point)
    plot_random_points_inside_ellipses(points, center_point)

if __name__ == '__main__':
    main()

