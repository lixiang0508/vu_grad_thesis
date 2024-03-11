import numpy as np

from Chromosome import Chromosome
from Obstacle import Obstacle
import utils
from copy import deepcopy

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
    # Example binary string for obstacle corner inclusion, adjust size according to obstacles
    bins = ''.join(np.random.choice(['0', '1'], len(obstacles)*4))
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
    obstacles = generate_complex_obstacles(num_obstacles=5, min_points=3, max_points=6)

    # Create a Chromosome instance
    chromosome = create_chromosome(num_steiner=5, num_terminals=3, obstacles=obstacles)

    # Test mutations on the chromosome
    test_mutations(chromosome, num_iterations=3)

if __name__ == "__main__":
    main()
