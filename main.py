# This is a sample Python script.
import csv
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

def process_terminal_file(file_path):
    terminals = []
    with open(file_path, newline='') as file:
        for line in file:
            line = line.strip()
            # print(line)
            columns = line.split(',')
            if columns[0] != 'Xcoord' :
                try:
                    # Convert the second and third column to float and add as a tuple to the list
                    x, y = float(columns[0]), float(columns[1])
                    terminals.append((x, y))
                except ValueError:
                    # If conversion to float fails, it means that we encountered invalid data
                    continue
    return terminals



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


def crossover(chromosome1, chromosome2):
    corners = []
    for obs in chromosome1.obstacles:
        corners.extend(obs.points) # chromosome 1 and 2 have the same obstacles
    terminal_x = [x[0] for x in chromosome1.terminals]
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
    c1_stpts_l.extend(c2_stpts_r)
    c2_stpts_l.extend(c1_stpts_r)
    # child1_stps = c1_stpts_l.extend(c2_stpts_r)
    # child2_stps = c2_stpts_l.extend(c1_stpts_r)
    child1_bins = c1_corners_l + c2_corners_r
    child2_bins = c2_corners_l + c1_corners_r

    return Chromosome(c1_stpts_l, child1_bins, chromosome1.terminals, chromosome1.obstacles) , Chromosome(c2_stpts_l ,child2_bins, chromosome1.terminals, chromosome1.obstacles)

def tournament(chromosomes, avg_dis_tmn, threshold, freq):
    '''threshold is a small number, it stands for the improvement ; freq is the number of times which improvement is lower
    than the threshold, when the times reach freq is it deemed as converge'''
    low_imp_times = 0
    cur_opt =  min(chromosomes, key=lambda obj: obj.cost)
    no_gen = 1  # number of generations
    pmax = 0.99
    pmin = 0.60
    while True:
        # each time randomly select 5 of chromosomes from the list and pick the one with the lowest cost
        for i in range(166):
            selected_chros_0 = random.sample(chromosomes, 5)
            min_cost_chro_0 = min(selected_chros_0, key=lambda obj: obj.cost)

            selected_chros_1 = random.sample(chromosomes, 5)
            min_cost_chro_1 = min(selected_chros_1, key=lambda obj: obj.cost)
            offspring1, offspring2 = crossover(min_cost_chro_0, min_cost_chro_1)
            # crossover part is done

            # mutation part
            p_flipMove = max(pmax * (1 - no_gen / 1000), pmin)
            p_add = (1 - p_flipMove) / 2
            p_remove = (1 - p_flipMove) / 2

            probs = [p_flipMove, p_add, p_remove]
            chosen_index = np.random.choice([0, 1, 2], p=probs)  # 0 1 2
            if chosen_index == 0:
                new_offspring1 = offspring1.flipMove(avg_dis_tmn, no_gen)
                new_offspring2 = offspring2.flipMove(avg_dis_tmn, no_gen)
                chromosomes.append(new_offspring1)
                chromosomes.append(new_offspring2)
            elif chosen_index == 1:
                new_offspring1 = offspring1.add_steiner_mutation()
                new_offspring2 = offspring2.add_steiner_mutation()
                chromosomes.append(new_offspring1)
                chromosomes.append(new_offspring2)
            elif chosen_index == 2:
                new_offspring1 = offspring1.remove_steiner_mutation()
                new_offspring2 = offspring2.remove_steiner_mutation()
                chromosomes.append(new_offspring1)
                chromosomes.append(new_offspring2)
        #At the end of each generation , discard the least fit 166 chromosomes
        chromosomes.sort(key=lambda x: x.cost, reverse=True)
        top_166_chromosomes = chromosomes[:166]
        for chromosome in top_166_chromosomes:
            chromosomes.remove(chromosome)
        temp_opt = min(chromosomes, key=lambda obj: obj.cost)
        if cur_opt.cost - temp_opt.cost < threshold: # fail to improve more than threshold
            low_imp_times +=1
        else:
            low_imp_times = 0

        cur_opt = temp_opt #update current optimal chromosome
        if low_imp_times == freq:
            print('after ', no_gen+1, ' tournaments it is converged and the cost is', cur_opt.cost)
            return cur_opt
        no_gen += 1




def initial_tournament(chromosomes, avg_dis_tmn):
    no_gen = 1 # number of generations
    pmax = 0.99
    pmin = 0.60
    # each time randomly select 5 of chromosomes from the list and pick the one with the lowest cost
    for i in range(200):
        selected_chros_0 = random.sample(chromosomes, 5)
        min_cost_chro_0 = min(selected_chros_0, key=lambda obj: obj.cost)

        selected_chros_1 = random.sample(chromosomes, 5)
        min_cost_chro_1 = min(selected_chros_1, key=lambda obj: obj.cost)
        offspring1, offspring2 = crossover(min_cost_chro_0, min_cost_chro_1)
        # crossover part is done

        #mutation part
        p_flipMove = max(pmax*(1 - no_gen/1000), pmin)
        p_add = (1 - p_flipMove)/2
        p_remove = (1 - p_flipMove)/2

        probs = [p_flipMove, p_add, p_remove]
        chosen_index = np.random.choice([0, 1, 2], p=probs) # 0 1 2
        if chosen_index == 0 :
            new_offspring1 = offspring1.flipMove(avg_dis_tmn, no_gen)
            new_offspring2 = offspring2.flipMove(avg_dis_tmn, no_gen)
            chromosomes.append(new_offspring1)
            chromosomes.append(new_offspring2)
        elif chosen_index == 1:
            new_offspring1 = offspring1.add_steiner_mutation()
            new_offspring2 = offspring2.add_steiner_mutation()
            chromosomes.append(new_offspring1)
            chromosomes.append(new_offspring2)
        elif chosen_index == 2:
            new_offspring1 = offspring1.remove_steiner_mutation()
            new_offspring2 = offspring2.remove_steiner_mutation()
            chromosomes.append(new_offspring1)
            chromosomes.append(new_offspring2)
    no_gen += 1
    chromosomes.pop() #delete the last added chromosome


def main_function(obstacle_path, terminal_path):

    chromosomes = []
    # Process the file and print the results
    obstacles = process_obstacle_file(obstacle_path)
    print('obstacles,', obstacles)
    softobs = []
    hardobs = []
    for obstacle in obstacles:
        # print(
        # f"Obstacle Weight: {obstacle['weight']}, Coordinates: {obstacle['coordinates']}, len:{len(obstacle['coordinates'])}")
        if obstacle['weight'] == sys.maxsize:
            hardobs.append(Obstacle(obstacle['weight'], 'hard', obstacle['coordinates']))

        else:
            softobs.append(Obstacle(obstacle['weight'], 'soft', obstacle['coordinates']))

    all_obstacles = []
    all_obstacles.extend(softobs)
    all_obstacles.extend(hardobs)
    # print('all_obstacles', all_obstacles)

    # if any terminal in soft or hard obstacle then regenerate one
    terminals = process_terminal_file(terminal_path)
    # print('terminals ', terminals)
    hard_corners = []
    for i in range(len(hardobs)):
        curob = hardobs[i]
        hard_corners.append(curob.points)
    # hard_corners = [ob['coordinates'] for ob in softobs]
    print('hard_corners', hard_corners)

    # print(terminals)
    dis = []
    n = len(terminals)
    for i in range(n):
        for j in range(i + 1, n):
            dis.append(utils.cal_dis(terminals[i], terminals[j]))
    avg_dis = sum(dis) / len(dis)
    print(' average distance of terminals is', avg_dis)
    k = sum(len(o['coordinates']) for o in obstacles)

    ''' initialization 1 Delaunay triangulation '''
    term_corners = []
    corners = []
    for o in obstacles:
        term_corners.extend(o['coordinates'])
        corners.extend(o['coordinates'])
    term_corners.extend(terminals)
    np_term_corners = np.array(term_corners)
    tri = Delaunay(term_corners)
    # Find centroids
    centroids = [np_term_corners[triangle].mean(axis=0) for triangle in tri.simplices]
    centroids = [cent for cent in centroids if not check_inside(hard_corners, cent)]

    # Print centroids
    # print('initialization 1 centroids' , np.array(centroids))

    # print('intialization 1 obstacles', all_obstacles)
    chromosomes.append(Chromosome(centroids, ''.join(str(0) for _ in range(k)), terminals, all_obstacles))
    # print(len(centroids))

    '''initialization 2 , in (1,1) randomly generate n+k Steiner points'''

    for j in range(50):
        points2 = []
        for i in range(n + k):
            cur = (np.random.random(), np.random.random())
            while check_inside(hard_corners, cur):
                cur = (np.random.random(), np.random.random())
            points2.append(cur)
        chromosomes.append(Chromosome(points2, ''.join(str(0) for _ in range(k)), terminals, all_obstacles))
    print('After init 2 we have how many chromosomes ', len(chromosomes))

    # initialization 3
    # randomly flip some of the genes in the second binary part of the chromosome (randomly generate 0 and 1)
    for i in range(50):
        chromosomes.append(Chromosome([], ''.join(random.choice('01') for _ in range(k)), terminals, all_obstacles))
    print('After init 3 we have how many chromosomes ', len(chromosomes))

    # Here we shall get 101 chromosomes
    initial_tournament(chromosomes, avg_dis)
    print('After full initialization we have how many chromosomes ', len(chromosomes))
    print('the intial average cost is', sum([ch.cost for ch in chromosomes]) / 500)
    # print(chromosomes[-1].cost)
    '''The initialization is done'''
    opt_chro = tournament(chromosomes, avg_dis, 0.0001, 3)

    '''What ever selection, mutation'''

    alter_opt_chro = opt_chro.simpson_line()
    if alter_opt_chro.cost < opt_chro.cost:
        print('The real optimal value is ', alter_opt_chro.cost)
    return min(alter_opt_chro.cost, opt_chro.cost)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    results =[]
    obstacle_path = 'SoftObstacles/obstacles2.csv'
    terminal_path = 'soft_terminals/terminals2.csv'
    for i in range(10):
        results.append(main_function(obstacle_path,terminal_path))

    print(results)

    #The last step is simpson_line()

