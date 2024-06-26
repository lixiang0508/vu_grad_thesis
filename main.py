# This is a sample Python script.
import copy
import csv
import math
import sys
import random
import utils

import numpy as np

from Chromosome import Chromosome
from Obstacle import Obstacle
from scipy.spatial import Delaunay
from visualization import vis
import matplotlib.path as mpath


def check_inside(vertices, point):
    for i in range(len(vertices)):
        # Define the vertices of the polygon
        # vertices = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        # polygon = mpath.Path(np.array(vertices[i]))
        polygon = mpath.Path(vertices[i] + [vertices[i][0]])  # todo made some changes

        # Point to be tested
        # point = np.array([0.5, 0.5])

        # Check if the point is inside the polygon
        # inside = polygon.contains_point(np.array(point))
        inside = polygon.contains_point(point)
        if inside == True:
            return True
    # print('Point inside polygon:', inside)
    return False


def process_terminal_file(file_path):
    terminals = []
    with open(file_path, newline='') as file:
        for line in file:
            line = line.strip()
            # print(line)
            columns = line.split(',')
            if columns[0] != 'Xcoord':
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
        first_char = file.read(1)
        if not first_char:  # if the file is empty
            print('empty file')
            return obstacles
        #print('first_char', first_char, 'is it', len(first_char))
        # if not empty back to the starting point
        file.seek(0)
        blank_line_count = 0  # Counter for consecutive blank lines
        for line in file:
            line = line.strip()
            # print(line)
            parts = line.split(',')
            # print(parts)
            # Check for blank line
            if not line or len(parts[0]) == 0 and len(parts[1]) == 0:
                blank_line_count += 1
                if blank_line_count == 2:  # Two consecutive blank lines signal end of file content
                    break
                continue  # Skip processing for the first blank line
            else:
                blank_line_count = 0  # Reset counter if line is not blank

            parts = line.split(',')
            #print(parts)
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


def mutation(chromosomes, avg_dis_tmn, low, up):
    pmax = 0.99
    pmin = 0.60
    #print('mutation')
    new_chromosomes = []
    #print('we have this number of chromosomes', len(chromosomes))
    for idx, chro in enumerate(chromosomes):
        try:
            fitness = chro.calculate_fitness(low, up)
            p_flipMove = max(pmax * (1 - fitness / 2), pmin)
            p_add = (1 - p_flipMove) / 2
            p_remove = (1 - p_flipMove) / 2
            probs = [p_flipMove, p_add, p_remove]
            probs = [float(p) / sum(probs) for p in probs]  # Normalize probabilities

            offspring_no = max(round(math.pow(10, fitness)), 1)  # Ensure it's at least 1
            gene_size = max(round(math.pow(5, 1 - fitness)), 1)  # Ensure it's at least 1

            #print('idx', idx, 'fitness is', fitness, 'offspring number is', offspring_no, 'gene_size is', gene_size)
            for i in range(offspring_no):
                chosen_index = np.random.choice([0, 1, 2], p=probs)  # 0 1 2
                cur_chro = copy.deepcopy(chro)
                if chosen_index == 0:
                    #print('flipmove')
                    for j in range(gene_size):

                        cur_chro = cur_chro.flipMove(avg_dis_tmn, fitness)
                elif chosen_index == 1:
                   # print('add steiner')
                    for j in range(gene_size):
                        cur_chro = cur_chro.add_steiner_mutation()
                else:
                    #print('remove steiner')
                    for j in range(gene_size):
                        cur_chro = cur_chro.remove_steiner_mutation()
                new_chromosomes.append(cur_chro)
        except Exception as e:
            a =1
            #print(f"An error occurred in mutation for chromosome index {idx}: {e}")

   # print('mutation done')
    chromosomes.extend(new_chromosomes)


'''
def mutation(chromosomes, avg_dis_tmn, low, up):

    pmax = 0.99
    pmin = 0.60
    print('mutation')
    idx =0
    new_chromosomes = []
    print('we have this number of chromosomes ', len(chromosomes))
    for chro in chromosomes:
        fitness = chro.calculate_fitness(low, up)
        p_flipMove = max(pmax * (1 - fitness/2 ), pmin)
        p_add = (1 - p_flipMove) / 2
        p_remove = (1 - p_flipMove) / 2
        offspring_no =  round(math.pow(10, fitness))
        gene_size = round(math.pow(5,1-fitness))
        print('idx ', idx, 'fitness is ',fitness, 'offspring number is ',offspring_no, 'gene_size is ',gene_size)
        idx+=1
        for i in range(offspring_no):
            probs = [p_flipMove, p_add, p_remove]
            chosen_index = np.random.choice([0, 1, 2], p=probs)  # 0 1 2
            cur_chro = copy.deepcopy(chro)
            if chosen_index ==0 :
                # flipMove

                for j in range(gene_size):
                    cur_chro = cur_chro.flipMove(avg_dis_tmn, fitness)
                    if cur_chro is None:
                        print('Cur chro is none')
                #
            elif chosen_index == 1:
                for j in range(gene_size):
                    cur_chro = cur_chro.add_steiner_mutation()
                    if cur_chro is None:
                        print('Cur chro is none')
            else:
                for j in range(gene_size):
                    cur_chro = cur_chro.remove_steiner_mutation()
                    if cur_chro is None:
                        print('Cur chro is none')
            new_chromosomes.append(cur_chro)
    print('mutation done ')

    chromosomes.extend(new_chromosomes)
'''


def random_subset(collection):
    # Generate a random number for the size of the subset
    subset_size = random.randint(0, len(collection))
    # Return a random subset of the collection with the random size
    return random.sample(collection, subset_size)

def elite_selection(chromosomes):
    sorted_chromosomes = sorted(chromosomes, key=lambda x: x.cost)

    # The elite chromosomes are directly taken from the sorted list
    elites = sorted_chromosomes[0]

    # Perform tournament selection for the remaining spots
    selected = [] # Start with the elite chromosomes
    selected.append(elites)
    chromosomes.remove(elites)
    while len(selected) < 30:
        tournament = random.sample(chromosomes, 3)  # Randomly select for tournament

        winner = min(tournament, key=lambda x: x.cost)  # The winner is the one with the lowest cost
        chromosomes.remove(winner)
        selected.append(winner)



    return selected

def main_function(obstacle_path, terminal_path):
    low_imp_times = 0

    no_gen = 1  # number of generations
    chromosomes = []
    # Process the file and print the results
    obstacles = process_obstacle_file(obstacle_path)
    print('obstacles,', obstacles)
    softobs = []
    hardobs = []
    for obstacle in obstacles:
        # print(
        # f"Obstacle Weight: {obstacle['weight']}, Coordinates: {obstacle['coordinates']}, len:{len(obstacle['coordinates'])}")
        if obstacle['weight'] == float(sys.maxsize):
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


    #let's calculate the lower bound and upper bound of length of MST
    chro_low = Chromosome([], ''.join('0' for _ in range(k)), terminals,all_obstacles, 0.1, 5.1)  # 0.1 and 5.1 has no meanings here
    low_bound = chro_low.cost * 0.74309

    chro_high = Chromosome([], ''.join('1' for _ in range(k)), terminals,all_obstacles,0.1, 5.2)
    up_bound = chro_high.cost



    # Print centroids
    # print('initialization 1 centroids' , np.array(centroids))

    # print('intialization 1 obstacles', all_obstacles)
    #chromosomes.append(Chromosome(centroids, ''.join(str(0) for _ in range(k)), terminals, all_obstacles))


    #For now, the centroids is the steiner set
    steiner_set = centroids

    # The first part of
    # the chromosome will then contain a random number (less or equal to the size of the steiner
    # set) of steiner points from the steiner set. The second part of the chromosome will contain
    # a random string with ones and zeros.
    for i in range(30):
        chromosomes.append(Chromosome(random_subset(steiner_set),''.join(random.choice('01') for _ in range(k)),terminals,all_obstacles,low_bound,up_bound))
    ubcount = 0
    lbcount = 0

    print('After full initialization we have how many chromosomes ', len(chromosomes))
    print('the intial average cost is', sum([ch.cost for ch in chromosomes]) / len(chromosomes))
    print('the minimum cost is ', min([ch.cost for ch in chromosomes]))
    # print(chromosomes[-1].cost)
    '''The initialization is done'''
    # Here we shall get 101 chromosomes
    #initial_tournament(chromosomes, avg_dis) #todo add fitness in the signature
    cur_opt = min(chromosomes, key=lambda obj: obj.cost)
    while True:
        mutation(chromosomes, avg_dis,  low_bound, up_bound)

        '''What ever selection, mutation'''
        fitness_vals = [chro.fitness for chro in chromosomes]
        if max(fitness_vals) > 0.8:
            ubcount = lbcount+1
        if max(fitness_vals) <0.2:
            lbcount = ubcount +1
        if lbcount == 10:
            low_bound += 0.05
            lbcount = 0
        if ubcount == 10:
            up_bound = up_bound - 0.05
            ubcount =0
        #calculate the highest and lowest distance
        dis_list = [chro.cost for chro in chromosomes]
        if low_bound >= min(dis_list):
            low_bound = min(dis_list) -0.1
        if up_bound <= max(dis_list):
            up_bound = min(dis_list) +0.1

        chromosomes= elite_selection(chromosomes)
        fit_mean = np.mean(np.array([chro.fitness for chro in chromosomes]))
        if fit_mean >= 0.8:
            up_bound = up_bound - max(0.1- 0.001*no_gen, 0.01)
        if fit_mean <= 0.2 :
            low_bound = low_bound + max(0.1- 0.001*no_gen, 0.01)



        temp_opt = min(chromosomes, key=lambda obj: obj.cost)
        if cur_opt.cost - temp_opt.cost < temp_opt.cost * 0.001:  # fail to improve more than threshold
            low_imp_times += 1
        else:
            low_imp_times = 0

        cur_opt = temp_opt  # update current optimal chromosome
        if low_imp_times == 500:
            print('after ', no_gen + 1, ' tournaments it is converged and the cost is', cur_opt.cost)
            print('nodes', cur_opt.get_nodes())
            print('steiner points', cur_opt.steinerpts)
            print('mst', cur_opt.mst)
            print('bins', cur_opt.bins)
            return cur_opt, no_gen
        no_gen += 1
        #if no_gen % 50 == 0:
        print('generation ', no_gen, ' cost ', cur_opt.cost)




def outer_vis(c,softobs,hardobs,path):

    all_corners = c.get_obstacle_corners()
    corners = []
    for i in range(len(c.bins)):
        if c.bins[i] == '1':
            corners.append(all_corners[i])
    nodes = c.get_nodes()
   # print('nodes',nodes)
    mst = c.mst
   # print('mst：', mst)
    mst_edges = []
    for edge in mst:
        mst_edges.append((nodes[edge[0]], nodes[edge[1]]))

    vis(c.terminals, c.steinerpts, corners, softobs, hardobs, mst_edges,path)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''
    for i in range(2, 3):
        obstacle_path = 'SoftObstacles/obstacles' + str(i) + '.csv'
        terminal_path = 'soft_terminals/terminals' + str(i) + '.csv'
        # for i in range(10):
        obs = 'soft_obs_' + str(i)
        visualize(obstacle_path, terminal_path)


    '''
    results = {}


    for i in [301]:
        obstacle_path = 'SolidObstacles/obstacles' + str(i) + '.csv'
        terminal_path = 'solid_terminals/terminals' + str(i) + '.csv'
        # for i in range(10):
        obs = 'soft_obs_' + str(i)
        chro, no_gen = main_function(obstacle_path, terminal_path)
        results[obs] = chro.cost
        print(obstacle_path, ' ', results[obs])
        with open('output1.txt', 'a') as file:
            # 将浮点数转换为字符串并写入文件

            file.write(obstacle_path + ' ' + str(results[obs]) + 'how many steinerpts  '+ str(len(chro.steinerpts))
                       +"how many corners "+ str(len(chro.nodes) - len(chro.steinerpts) - len(chro.terminals))+ "iters"+ str(no_gen)+" \n")



