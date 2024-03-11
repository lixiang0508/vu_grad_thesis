import random

import numpy as np
import matplotlib.path as mpath



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
    return (x, y)


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


def is_line_segment_inside_polygon(polygon_vertices, line_segment):
    """
    判断线段是否在多边形内部。

    :param polygon_vertices: 多边形的顶点坐标列表，格式为[(x1, y1), (x2, y2), ..., (xn, yn)]
    :param line_segment: 线段的起点和终点坐标，格式为[(x_start, y_start), (x_end, y_end)]
    :return: 布尔值，如果线段在多边形内部返回True，否则返回False
    """
    # 创建多边形路径
    polygon_path = mpath.Path(polygon_vertices + [polygon_vertices[0]])  # 闭合多边形

    # 计算线段的中点
    mid_point = np.mean(line_segment, axis=0)

    # 检查中点是否在多边形内部
    return polygon_path.contains_point(mid_point)

def segment_in_obstacle_length(obstacle, segment):
    intersections = []
    for i in range(len(obstacle)):
        next_i = (i + 1) % len(obstacle)
        if intersect(obstacle[i], obstacle[next_i], segment[0], segment[1]):
            point = line_intersection(obstacle[i], obstacle[next_i], segment[0], segment[1])
            if point:
                intersections.append(point)
    # Check if segment start/end points are inside the obstacle, and add them to intersections if so
    if point_inside_polygon(segment[0], obstacle):
        intersections.append(segment[0])
    if point_inside_polygon(segment[1], obstacle):
        intersections.append(segment[1])

    # Remove duplicates and sort by distance from the segment's start
    intersections = list(set(intersections))
    intersections.sort(key=lambda x: ((x[0] - segment[0][0]) ** 2 + (x[1] - segment[0][1]) ** 2))

    # Calculate the length of the segment inside the obstacle
    length = 0
    for i in range(1, len(intersections), 2):
        inside = is_line_segment_inside_polygon(obstacle, [intersections[i],intersections[i-1]])
        if i < len(intersections) and inside:
            length += cal_dis(intersections[i],intersections[i-1])
            #length += ((intersections[i][0] - intersections[i - 1][0]) ** 2 + (
            #            intersections[i][1] - intersections[i - 1][1]) ** 2) ** 0.5

    return length

