import numpy as np

from Chromosome import Chromosome
from Obstacle import Obstacle
import random
import utils
from copy import deepcopy

def crossover(chromosome1, chromosome2):
    corners = []
    for obs in chromosome1.obstacles:
        corners.extend(obs.points)  # chromosome 1 and 2 have the same obstacles
    print('length of corners is ', len(corners), ' corners', corners)
    print('length of bins 1 is ',len(chromosome1.bins))
    print('length of bins 2 is ', len(chromosome2.bins))
    terminal_x = [x[0] for x in chromosome1.terminals]
    split = random.uniform(min(terminal_x), max(terminal_x))
    print("crossover, split at ",split)
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
        if corner[0] <= split:
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

    return Chromosome(c1_stpts_l, child1_bins, chromosome1.terminals, chromosome1.obstacles), Chromosome(c2_stpts_l,
                                                                                                         child2_bins,
                                                                                                         chromosome1.terminals,
                                                                                                         chromosome1.obstacles)

# Generate complex polygonal obstacles
def generate_complex_obstacles(num_obstacles, min_points, max_points):
    obstacles = []
    for _ in range(num_obstacles):
        num_points = np.random.randint(min_points, max_points)
        points = [(np.random.random(), np.random.random()) for _ in range(num_points)]
        weight = np.random.random() * 10  # Example weight
        obstacles.append(Obstacle(weight, 'soft', points))
    return obstacles

# Create a Chromosome instance
def create_chromosome(num_steiner, num_terminals, obstacles):
    steinerpts = [(np.random.random(), np.random.random()) for _ in range(num_steiner)]
    terminals = [(np.random.random(), np.random.random()) for _ in range(num_terminals)]
    corners = []
    for obs in obstacles:
        corners.extend(obs.points)
    # Example binary string for obstacle corner inclusion, adjust size according to obstacles
    bins = ''.join(np.random.choice(['0', '1'], len(corners)))
    chromosome = Chromosome(steinerpts, bins, terminals, obstacles)
    return chromosome

# Test the mutations on the chromosome
def test_mutations(chromosome, num_iterations):
    for _ in range(num_iterations):
        # Perform flipMove mutation
        chromosome = chromosome.flipMove(0.1, 50)  # Example arguments for dis and nogen
        print(f"After flipMove: {len(chromosome.steinerpts)} Steiner points")
        print(chromosome.steinerpts)
        # Perform add_steiner_mutation
        chromosome = chromosome.add_steiner_mutation()
        print(f"After add_steiner_mutation: {len(chromosome.steinerpts)} Steiner points")
        print(chromosome.steinerpts)
        # Perform remove_steiner_mutation
        chromosome = chromosome.remove_steiner_mutation()
        print(f"After remove_steiner_mutation: {len(chromosome.steinerpts)} Steiner points")
        print(chromosome.steinerpts)
# Main function to run the test
def main():
    # Generate obstacles
    obstacles = generate_complex_obstacles(num_obstacles=2, min_points=1, max_points=9)

    # Create a Chromosome instance
    chromosome1  = create_chromosome(num_steiner=5, num_terminals=3, obstacles=obstacles)
    chromosome2 = create_chromosome(num_steiner=3, num_terminals=4, obstacles=obstacles)
    #print('chromosome1.steinerpts',chromosome1.steinerpts)
    #print('chromosome1.bins',chromosome1.bins)

    #print('chromosome2.steinerpts',chromosome2.steinerpts)
    #print('chromosome2.bins',chromosome2.bins)

    newchro1, newchro2 = crossover(chromosome1,chromosome2)

    #print('child chromosome1.steinerpts',newchro1.steinerpts)
    #print('child chromosome1.bins',newchro1.bins)

    print('child chromosome2.steinerpts', newchro2.steinerpts)
    print('child chromosome2.bins', newchro2.bins)

    print('mst',newchro2.mst)

    print('len of nodes',len(newchro2.nodes))

    print('nodes',newchro2.nodes)

    print('len of steiner points', len(newchro2.steinerpts))

    degrees = {node: 0 for node in range(len(newchro2.steinerpts))}
    for edge in newchro2.mst:
        if newchro2.nodes[edge[0]] in newchro2.steinerpts:
            degrees[edge[0]] += 1
        if newchro2.nodes[edge[1]] in newchro2.steinerpts:
            degrees[edge[1]] += 1

    degree_2_pts = [pt for pt, degree in degrees.items() if degree == 2]

    #new_chro = chromosome.simpson_line()
    #print(new_chro)

    # Test mutations on the chromosome
    #test_mutations(chromosome, num_iterations=3)



if __name__ == "__main__":
    main()
