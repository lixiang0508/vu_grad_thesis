import sys
import random

import Obstacle
from Obstacle import Obstacle
from scipy.spatial import Delaunay

'''
class Chromosome:
    cost = sys.maxsize
    edges = []

    def __init__(self, steinerpts, bins):
        self.steinerpts = steinerpts
        self.bins = bins # k , the length of list of obstacle corners is k
'''

import sys
import numpy as np
import itertools
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.path as mpath


class Chromosome:
    def __init__(self, steinerpts, bins, terminals , obstacles):
        self.steinerpts = steinerpts  # List of tuples for Steiner points (x, y)
        self.bins = bins  # Binary string indicating obstacle corner inclusion
        self.cost = sys.maxsize  # Initially set to maximum value
        self.terminals = terminals
        self.obstacles = obstacles
        dist_matrix = distance_matrix(steinerpts + terminals, steinerpts + terminals) # to be updated
        mst = minimum_spanning_tree(dist_matrix).toarray()

        # Update the edges based on MST calculation
        self.edges = np.argwhere(mst > 0)  # List to store edges of the MST

    def flipMove(self, dis, nogen):
        # dis stands for average Euclidean distance between terminals
        spts = self.steinerpts
        bins = self.bins
        mrange = max(1 - nogen / 100, 0.01) * dis
        xmove = np.random.uniform(0, mrange)
        ymove = np.random.uniform(0, mrange)
        s = len(spts)
        k = len(bins)

        probs = np.random.binomial(1, 1 / (s + k), s + k)
        for i in range(s):
            if probs[i] == 1:
                loc = spts[i]
                flagx = 1
                flagy = 1
                if loc[0] + xmove > 1:
                    flagx = 0
                if loc[1] + ymove > 1:
                    flagy = 0
                spts[i][0] += xmove * flagx
                spts[i][1] += ymove * flagy
        # self.steinerpts = spts
        for i in range(k):
            if probs[i + s] == 1:
                if bins[i] == 0:
                    bins[i] = 1
                else:
                    bins[i] = 0
        # self.bins = bins
        ret_chromosome = Chromosome(spts, bins, self.terminals, self.obstacles)
        # ret_chromosome.edges = self.edges
        # ret_chromosome.cost = self.cost
        return ret_chromosome

    def calculate_mst(self):
        """Calculates the Weighted Minimum Spanning Tree for the chromosome."""
        ret_chromosome = Chromosome(self.steinerpts, self.bins, self.terminals, self.obstacles)

        ret_chromosome.steinerpts = self.steinerpts
        all_points = self.steinerpts
        dist_matrix = distance_matrix(all_points, all_points)
        mst = minimum_spanning_tree(dist_matrix).toarray()

        # Update the edges based on MST calculation
        self.edges = np.argwhere(mst > 0)
        ret_chromosome.edges = self.edges
        ret_chromosome.cost = self.cost
        return ret_chromosome

    def add_steiner_mutation(self, hard_obstacles):
        """Performs the addSteiner mutation on the chromosome."""
        new_chro = self.calculate_mst()

        # Find nodes with angles less than 2*pi/3
        small_angle_nodes = []  # List to store nodes with small angles
        for edge in self.edges:
            node, neighbor = edge
            for other_neighbor in (set(range(len(self.steinerpts))) - {neighbor}):
                angle = self.calculate_angle(node, neighbor, other_neighbor)
                if angle < 2 * np.pi / 3:
                    small_angle_nodes.append((node, neighbor, other_neighbor))

        if small_angle_nodes:
            # Select one small angle node set randomly and calculate Steiner point
            selected_nodes = small_angle_nodes[np.random.randint(len(small_angle_nodes))]
            new_steiner_point = self.calculate_steiner_point(selected_nodes)
            new_chro.steinerpts.append(new_steiner_point)
        else:
            # Add a random Steiner point if no small angle found
            random_point = self.generate_random_steiner_point(hard_obstacles)
            new_chro.steinerpts.append(random_point)
        return new_chro

    def removeSteiner(self):
        degrees = {node: 0 for node in range(self.steinerpts + self.terminals)}
        for edge in self.edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
        degree_2_pts = [pt for pt, degree in degrees.items() if degree == 2]

        # If there are no Steiner points with degree 2, return without doing anything
        if not degree_2_pts:
            return

        idx = random.choice(degree_2_pts)
        while idx >= len(self.steinerpts):
            idx = random.choice(degree_2_pts)

        pt_to_remove = self.steinerpts[idx]  # random would generate  an index of steiner points
        return Chromosome([pt for pt in self.steinerpts if pt != pt_to_remove], self.bins, self.terminals, self.obstacles)

    def calculate_angle(self, node, neighbor1, neighbor2):
        """Calculates the angle between three points."""
        # Fetch point coordinates
        p1 = np.array(self.steinerpts)[node]
        p2 = np.array(self.steinerpts)[neighbor1]
        p3 = np.array(self.steinerpts)[neighbor2]

        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p1

        # Calculate angle using dot product
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return angle

    def calculate_steiner_point(self, nodes):
        """Calculates the position of a new Steiner point."""
        # Here, we're just calculating the centroid of the triangle formed by the nodes.
        points = [np.array(self.steinerpts)[node] for node in nodes]
        centroid = np.mean(points, axis=0)
        return tuple(centroid)

    def generate_random_steiner_point(self, hard_obstacles):
        """Generates a random Steiner point avoiding placement inside  solid obstacles."""
        # Placeholder for random point generation. You need to implement obstacle avoidance.
        cur = np.random.random(), np.random.random()
        while self.check_inside(hard_obstacles, cur):
            cur = np.random.random(), np.random.random()

        return cur

    def check_inside(self, vertices, point):
        for i in range(len(vertices)):
            # Define the vertices of the polygon
            # vertices = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
            polygon = mpath.Path(np.array(vertices[i]))

            # Point to be tested
            # point = np.array([0.5, 0.5])

            # Check if the point is inside the polygon
            inside = polygon.contains_point(np.array(point))
            if inside == True:
                return True
        # print('Point inside polygon:', inside)
        return False


# Example usage
terminals = [(0.1, 0.1), (0.9, 0.9)]  # List of terminals (x, y)
initial_steiner_points = [(0.5, 0.5)]  # Initial list of Steiner points
bins = "110"  # Example binary string for obstacle corner inclusion
obstacles = [Obstacle(1.1, 'soft',[(0.1, 0.1), (0.9, 0.9), (0.3,0.4)]) , Obstacle(1.1, 'soft',[(0.7, 0.1), (0.9, 0.9),(0.5,0.6)]) ]

chromosome = Chromosome(initial_steiner_points, bins, terminals, obstacles)
chromosome.add_steiner_mutation([[(0.1, 0.1), (0.9, 0.9)]])
print("Steiner Points:", chromosome.steinerpts)
