import random
import sys

import numpy as np
import matplotlib.path as mpath

def round_point(point, precision=6):
    return (round(point[0], precision), round(point[1], precision))


def check_inside(vertices, point):
    for i in range(len(vertices)):
        # Define the vertices of the polygon
        #vertices = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        polygon = mpath.Path(np.array(vertices[i]))

        # Point to be tested
        #point = np.array([0.5, 0.5])

        # Check if the point is inside the polygon
        inside = polygon.contains_point(np.array(point))
        if inside==True:
            return True
    #print('Point inside polygon:', inside)
    return False
def cal_dis(x, y):
    return np.sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]))


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def line_intersection(p1, p2, p3, p4):
    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(p1, p2), det(p3, p4))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

'''
def point_inside_polygon(point, polygon):
    # A point is inside a polygon if a horizontal ray to the right intersects the polygon an odd number of times
    # Simple ray-casting algorithm
    x, y = point
    odd_intersects = False
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
                (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) +
                 polygon[i][0]):
            odd_intersects = not odd_intersects
    return odd_intersects
'''
def point_on_line(px, py, x1, y1, x2, y2, epsilon=1e-10):
    if x1 == x2:  # 垂直线段
        return min(y1, y2) <= py <= max(y1, y2) and abs(px - x1) <= epsilon
    if y1 == y2:  # 水平线段
        return min(x1, x2) <= px <= max(x1, x2) and abs(py - y1) <= epsilon
    # 一般线段
    return abs((py - y1) * (x2 - x1) - (y2 - y1) * (px - x1)) <= epsilon and min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)

def point_inside_polygon(point, polygon):
    x, y = point
    odd_intersects = False
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        x1, y1 = polygon[i]
        if x1 == x and y1==y :
          return True
        x2, y2 = polygon[j]

        if point_on_line(x, y, x1, y1, x2, y2):
            return True  # 点在边界上，直接返回True

        # 将顶点按照y坐标排序
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        # 检查射线是否与多边形的一个边相交
        if (y1 <= y < y2):
            crossX = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if x < crossX:
                odd_intersects = not odd_intersects

    return odd_intersects

def is_line_segment_inside_polygon(polygon_vertices, line_segment):
    """
    Check if the segement is inside the polygon

    :param polygon_vertices: the list of vertices of polygon，[(x1, y1), (x2, y2), ..., (xn, yn)]
    :param line_segment: the start and end point of the segement，[(x_start, y_start), (x_end, y_end)]
    :return: boolean value
    """

    polygon_path = mpath.Path(polygon_vertices + [polygon_vertices[0]])  # closed polygon

    # the midpoint of the segement
    mid_point = np.mean(line_segment, axis=0)


    return polygon_path.contains_point(mid_point)

def segment_in_obstacle_length(obstacle, segment):
    intersections = []
    for i in range(len(obstacle)):
        next_i = (i + 1) % len(obstacle)
        if intersect(obstacle[i], obstacle[next_i], segment[0], segment[1]):
            point = line_intersection(obstacle[i], obstacle[next_i], segment[0], segment[1])
            if point:
                intersections.append(round_point(tuple(point)))

    if point_inside_polygon(segment[0], obstacle):
        intersections.append(round_point(tuple(segment[0])))
    if point_inside_polygon(segment[1], obstacle):
        intersections.append(round_point(tuple(segment[1])))

    intersections = list(set(intersections))  # Remove duplicates
    intersections.sort(key=lambda x: ((x[0] - segment[0][0]) ** 2 + (x[1] - segment[0][1]) ** 2))

    length = 0
    for i in range(1, len(intersections)):
        inside = is_line_segment_inside_polygon(obstacle, [intersections[i], intersections[i - 1]])
        if inside:
            length += cal_dis(intersections[i], intersections[i - 1])

    return length


# Function to calculate the Fermat-Torricelli point
def fermat_torricelli_point(neighbours):
    A = np.array(neighbours[0])
    B = np.array(neighbours[1])
    C = np.array(neighbours[2])
    # Calculate the lengths of the sides
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)

    # Calculate the sum of the lengths
    sum_lengths = a + b + c

    # Calculate the weights
    alpha = a / sum_lengths
    beta = b / sum_lengths
    gamma = c / sum_lengths

    # Calculate the position of the Fermat-Torricelli point
    P = alpha * A + beta * B + gamma * C
    print('position of Fermat-Torricelli point is', P)
    return P
def select_random_one_index(arr):
    # Find all indices of elements that are 1
    indices_of_ones = [i for i, element in enumerate(arr) if element == 1 or element ==-1]
    if len(indices_of_ones)==0:
        return -1
    # Randomly select and return one of these indices
    return random.choice(indices_of_ones) if indices_of_ones else None

def print_out(chromosome):
    all_obs = chromosome.obstacles
    soft_obstacles = [ob for ob in all_obs if ob['weight']<sys.maxsize]
    solid_obstacles = [ob for ob in all_obs if ob['weight']==sys.maxsize]
    print('The soft obstacles are',soft_obstacles)
    print('The solid obstacles are', soft_obstacles)
    print('The steiner points are', chromosome.steinerpts)
    print('The terminals are', chromosome.terminals)
    edges = []
    for i in range(len(chromosome.mst)):
        edges.append((chromosome.nodes[chromosome.mst[i][0]], chromosome.nodes[chromosome.mst[i][1]]))
    print('The steiner edges are ', edges)
