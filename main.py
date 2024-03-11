# This is a sample Python script.
import sys
import random
import utils

import numpy as np

from Chromosome import Chromosome
from Obstacle import Obstacle
from scipy.spatial import Delaunay
import matplotlib.path as mpath

def check_inside(vertices, point):
    for i in range(len(vertices)):
        # Define the vertices of the polygon
        #vertices = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        #polygon = mpath.Path(np.array(vertices[i]))
        polygon = mpath.Path(vertices[i] + [vertices[i][0]]) #todo made some changes

        # Point to be tested
        #point = np.array([0.5, 0.5])

        # Check if the point is inside the polygon
        #inside = polygon.contains_point(np.array(point))
        inside = polygon.contains_point(point)
        if inside==True:
            return True
    #print('Point inside polygon:', inside)
    return False



# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# Define a function to process the CSV file and extract obstacles and their coordinates
def process_obstacle_file(file_path):
    obstacles = []
    current_obstacle = {}

    with open(file_path, 'r') as file:
        blank_line_count = 0  # Counter for consecutive blank lines
        for line in file:
            line = line.strip()
            # print(line)
            parts = line.split(',')
            # print(parts)
            # Check for blank line
            if len(parts[0]) == 0 and len(parts[1]) == 0:
                blank_line_count += 1
                if blank_line_count == 2:  # Two consecutive blank lines signal end of file content
                    break
                continue  # Skip processing for the first blank line
            else:
                blank_line_count = 0  # Reset counter if line is not blank

            parts = line.split(',')
            print(parts)
            if len(parts[0]) != 0 and len(parts[1]) == 0:  # New obstacle detected
                if current_obstacle:  # Save the previous obstacle if it exists
                    obstacles.append(current_obstacle)
                if parts[0] == 'max':
                    current_obstacle = {'weight': float(sys.maxsize), 'coordinates': []}
                else:

                    current_obstacle = {'weight': float(parts[0]), 'coordinates': []}
            # elif len(parts) == 2:  # Coordinate line
            elif len(parts[0]) != 0 and len(parts[1]) != 0:
                current_obstacle['coordinates'].append((float(parts[0]), float(parts[1])))

    if current_obstacle:  # Add the last obstacle if it exists
        obstacles.append(current_obstacle)

    return obstacles


def crossover(chromosome1, chromosome2, terminals, corners):
    terminal_x = [x[0] for x in terminals] #todo mention
    split = random.uniform(min(terminal_x), max(terminal_x))
    c1_stpts_l = []
    c1_stpts_r = []
    c2_stpts_l = []
    c2_stpts_r = []

    c1_corners_l = ""
    c1_corners_r = ""
    c2_corners_l = ""
    c2_corners_r = ""
    for stp in chromosome1.steinerpts:
        if stp[0] <= split:
            c1_stpts_l.append(stp)
        else:
            c1_stpts_r.append(stp)
    for stp in chromosome2.steinerpts:
        if stp[0] <= split:
            c2_stpts_l.append(stp)
        else:
            c2_stpts_r.append(stp)
    for i in range(len(corners)):
        corner = corners[i]
        c1_bin = chromosome1.bins[i]
        c2_bin = chromosome2.bins[i]
        if corner[0]<= split:
            c1_corners_l += c1_bin
            c2_corners_l += c2_bin
        else:
            c1_corners_r += c1_bin
            c2_corners_r += c2_bin
    child1_stps = c1_stpts_l.extend(c2_stpts_r)
    child2_stps = c2_stpts_l.extend(c1_stpts_r)
    child1_bins = c1_corners_l + c2_corners_r
    child2_bins = c2_corners_l + c1_corners_r

    return [Chromosome(child1_stps, child1_bins, terminals, chromosome1.obstacles) , Chromosome(child2_stps,child2_bins, terminals, chromosome1.obstacles)]


def initial_tournament(chromosomes):
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(sys.maxsize)
    # print(np.random.binomial(1, 0.9, 10))
    # print_hi('PyCharm')
    # Specify the path to your CSV file
    file_path = 'SoftObstacles/obstacles1.csv'
    chromosomes = []
    # Process the file and print the results
    obstacles = process_obstacle_file(file_path)
    print('obstacles,', obstacles)
    softobs = []
    hardobs = []
    for obstacle in obstacles:
        print(
            f"Obstacle Weight: {obstacle['weight']}, Coordinates: {obstacle['coordinates']}, len:{len(obstacle['coordinates'])}")
        if obstacle['weight'] == sys.maxsize:
            hardobs.append(Obstacle(obstacle['weight'], 'hard', obstacle['coordinates']))

        else:
            softobs.append(Obstacle(obstacle['weight'], 'soft', obstacle['coordinates']))
    all_obstacles = hardobs.extend(softobs)
    n = int(input("Please input the number of terminals： "))
    # if any terminal in soft or hard obstacle then regenerate one
    terminals = []
    hard_corners = []
    for i in range(len(hardobs)):
        curob = hardobs[i]
        hard_corners.append(curob.points)
    #hard_corners = [ob['coordinates'] for ob in softobs]
    print('hard_corners', hard_corners)
    if len(hard_corners)>0:
        for i in range(n):
            cur = (np.random.random(), np.random.random())
            while check_inside(hard_corners, cur):
                cur = (np.random.random(), np.random.random())
            terminals.append(cur)
    else:
        for i in range(n):
            cur = (np.random.random(), np.random.random())
            terminals.append(cur)

    #print(terminals)
    dis = []
    for i in range(n):
        for j in range(i + 1, n):
            dis.append(utils.cal_dis(terminals[i], terminals[j]))
    print(' average distance is', sum(dis) / len(dis))

    '''
    for t in terminals:
        while t in soft or hard obstacles:
            t = np.random.rand(1,2) 
    '''
    k = sum(len(o['coordinates']) for o in obstacles)

    # initialization 1 Delaunay triangulation
    ''' initialization 1 Delaunay triangulation '''
    term_corners =[]
    corners=[]
    for o in obstacles:
        term_corners.extend(o['coordinates'])
        corners.extend(o['coordinates'])
    term_corners.extend(terminals)
    np_term_corners = np.array(term_corners)
    tri = Delaunay(term_corners)
    # Find centroids
    centroids = [np_term_corners[triangle].mean(axis=0) for triangle in tri.simplices]
    centroids = [cent for cent in centroids if not check_inside(hard_corners,cent)]
    '''
    for cent in centroids:
        if check_inside(hard_corners,cent):
            centroids.remove(cent)
    '''


    # Print centroids
    print('initialization 1 centroids' , np.array(centroids))
    chromosomes.append(Chromosome(centroids, ''.join(str(0) for _ in range(k)), terminals, all_obstacles))
    #print(len(centroids))


    '''initialization 2 , in (1,1) randomly generate n+k Steiner points'''
    # initialization 2
    # in (1,1) randomly generate n+k Steiner points


    for j in range(50):
        points2 = []
        for i in range(n+k):
            cur = (np.random.random(), np.random.random())
            while check_inside(hard_corners, cur):
                cur = (np.random.random(), np.random.random())
            points2.append(cur)
        chromosomes.append(Chromosome(points2, ''.join(str(0) for _ in range(k)), terminals, all_obstacles))
    print(len(chromosomes))


    # initialization 3
    # randomly flip some of the genes in the second binary part of the chromosome (randomly generate 0 and 1)
    for i in range(50):
        chromosomes.append(Chromosome([], ''.join(random.choice('01') for _ in range(k)), terminals,all_obstacles))
    # print(len(chromosomes))
    #Here we shall get 101 chromosomes
