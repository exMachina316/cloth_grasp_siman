import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import logging
import open3d as o3d
import csv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the PNG file as depth data (greyscale)


class DepthData:
    @staticmethod
    def load_png(filename):
        # Convert to grayscale ('L' mode)
        img = Image.open(filename).convert('L')
        depth_data_np = np.array(img, dtype=np.float32)
        return depth_data_np

    @staticmethod
    def get_point_cloud(depth_data, z_threshold_min=0, z_threshold_max=255):
        height, width = depth_data.shape
        x = np.arange(0, width)
        y = np.arange(0, height)
        x, y = np.meshgrid(x, y)
        z = depth_data
        mask = (z >= z_threshold_min) & (z <= z_threshold_max)
        x_pruned = x[mask]
        y_pruned = y[mask]
        z_pruned = z[mask]
        point_cloud = np.vstack((x_pruned, y_pruned, z_pruned)).T
        return point_cloud

    @staticmethod
    def plot_point_cloud(point_cloud, z_threshold_min, z_threshold_max, best_point=None):
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        scatter = ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            c=point_cloud[:, 2],
            cmap='inferno',
            marker='.',
            s=4,
            alpha=0.6,
            label='Point Cloud'
        )
        if best_point is not None:
            ax.scatter(
                best_point[0],
                best_point[1],
                best_point[2],
                color='cyan',
                marker='.',
                s=200,
                label='Best Point'
            )
            logging.info(
                f"Best Point Coordinates: X={best_point[0]}, Y={best_point[1]}, Z={best_point[2]}")

        ax.set_title(
            f"3D Point Cloud (Pruned by Z-value: [{z_threshold_min}, {z_threshold_max}])")
        ax.set_xlabel("X-axis (pixels)")
        ax.set_ylabel("Y-axis (pixels)")
        ax.set_zlabel("Depth (Z values)")
        ax.legend()
        plt.tight_layout()
        plt.show()


class SimulatedAnnealing:
    def __init__(self, point_cloud, initial_temp=1000, cooling_rate=0.99, min_temp=1, radius=2):
        self.point_cloud = point_cloud
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.current_point = random.choice(point_cloud)
        self.best_point = self.current_point
        self.radius = radius

    def calc_strain_curvature(self, local_points):
        # Step 1: Calculate the centroid of the points
        centroid = np.mean(local_points, axis=0)
        
        # Step 2: Center the points by subtracting the centroid
        centered_points = local_points - centroid
        
        # Step 3: Perform SVD to find the normal of the best-fit plane
        _, _, vh = np.linalg.svd(centered_points)
        plane_normal = vh[-1]  # The last row of vh is the normal to the plane
        
        # Step 4: Calculate the distance of each point from the plane
        # Distance of a point p to a plane with point on plane 'centroid' and normal 'plane_normal' is:
        # distance = |(p - centroid) • plane_normal| / ||plane_normal||
        distances = np.abs(np.dot(centered_points, plane_normal)) / np.linalg.norm(plane_normal)
        
        # Step 5: Calculate mean distance as strain
        strain = np.mean(distances)

        # Step 6: Calculate the variance of distances as a measure of curvature
        curvature = np.var(distances)
        
        return strain, curvature

    def objective_function(self, point):
        # Calculate Euclidean distances
        distances = np.linalg.norm(self.point_cloud - point, axis=1)
        
        # Define neighborhood
        local_points = self.point_cloud[distances < self.radius]
        if len(local_points) < 2:
            return 0
        
        # Density
        density = len(local_points) / (np.pi * self.radius ** 2)
        
        # Strain and Curvature
        strain, curvature = self.calc_strain_curvature(local_points)
        
        return curvature + density + strain, [curvature, density, strain]

    def neighbor(self):
        """Randomly choose a nearby point as a neighbor."""
        idx = random.randint(0, len(self.point_cloud) - 1)
        return self.point_cloud[idx]

    def acceptance_probability(self, current_y, neighbor_y):
        if neighbor_y > current_y:
            return 1.0
        else:
            # Avoid division by zero in temperature
            if self.temperature == 0:
                return 0
            return np.exp((neighbor_y - current_y) / self.temperature)

    def optimize(self):
        iteration = 0
        best_point_y = 0
        best_met = [0, 0, 0]

        with open('optimization_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration', 'Temperature', 'Best Fitness', 'Curvature', 'Density', 'Strain'])
            while self.temperature > self.min_temp:
                neighbor_point = self.neighbor()
                current_y, current_met = self.objective_function(self.current_point)
                neighbor_y, _ = self.objective_function(neighbor_point)

                # Decide if we should move to the neighbor
                ap = self.acceptance_probability(current_y, neighbor_y)
                if ap > random.random():
                    self.current_point = neighbor_point

                # Check if this is the best point we've found so far
                if current_y > best_point_y:
                    best_point_y = current_y
                    best_met = current_met
                    self.best_point = self.current_point

                # Cool down the system
                self.temperature *= self.cooling_rate
                iteration += 1

                if iteration % 100 == 0 or self.temperature < self.min_temp * 10:
                    logging.info(
                        f"Iteration {iteration}: Temperature={self.temperature:.2f}, Best Fittness={best_point_y}, Best Met={best_met}")
                writer.writerow([iteration, self.temperature, best_point_y, best_met[0], best_met[1], best_met[2]])

            return self.best_point, best_met

# Main execution
if __name__ == "__main__":
    try:
        # # Path to your PNG file
        png_filename = '/app/workspace/genetic_algo/2019-02-25-hanging-blender-dataset-07-raw/img-6.png0200.png'

        # # Load the .png file
        logging.info("Loading depth data from PNG...")
        depth_data = DepthData.load_png(png_filename)

        # Z-value threshold for pruning
        z_threshold_min = 0
        z_threshold_max = 100

        # # Get the point cloud from the depth data
        logging.info("Generating point cloud...")
        point_cloud = DepthData.get_point_cloud(depth_data, z_threshold_min, z_threshold_max)

        # input_file = "1-1.ply"
        # pcd = o3d.io.read_point_cloud(input_file)  # Read the point cloud

        # # Downsample the point cloud with a voxel size of your choice
        # voxel_size = 0.005  # Adjust voxel size based on desired downsampling
        # downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # # Convert open3d format to numpy array
        # # Here, you have the point cloud in numpy format.
        # point_cloud = np.asarray(downsampled_pcd.points)

        logging.info(f"Point cloud generated with {len(point_cloud)} points.")

        # Perform simulated annealing to find the maxima of y
        logging.info("Starting Simulated Annealing optimization...")
        sa = SimulatedAnnealing(point_cloud, 10000, 0.99, 1, 16)
        best_point, best_met = sa.optimize()
        logging.info(f"Best Point Found: {best_point}")
        logging.info(f"Best Point Metrics: {best_met}")

        # Plot the 3D surface with the best point
        logging.info("Plotting the point cloud with the best point...")
        DepthData.plot_point_cloud(
            point_cloud, z_threshold_min, z_threshold_max, best_point=best_point)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        exit(0)
