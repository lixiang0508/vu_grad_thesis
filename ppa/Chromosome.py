import heapq
import random

import Obstacle
import copy
import utils
from Obstacle import Obstacle

import sys
import numpy as np


class Chromosome:
    def __init__(self, steinerpts, bins, terminals, obstacles,low,up):
        self.steinerpts = steinerpts  # List of tuples for Steiner points (x, y)
        self.bins = bins  # Binary string indicating obstacle corner inclusion
        # self.cost = sys.maxsize  # Initially set to maximum value
        self.terminals = terminals
        self.obstacles = obstacles
        self.nodes = self.get_nodes()  # possible nodes to generate a MST ( all steiner pts, all terminals, obstacles
        # with char 1)
        self.mst, self.cost = self.cal_mst()
        self.low = low
        self.up = up
        self.fitness = self.calculate_fitness(low,up)
        # dist_matrix = distance_matrix(steinerpts + terminals, steinerpts + terminals) #todo to be updated
        # mst = minimum_spanning_tree(dist_matrix).toarray()

        # Update the edges based on MST calculation
        # self.edges = np.argwhere(self.mst > 0)  # List to store edges of the MST
    def calculate_objective(self,low,up):
        return (self.cost - low) / (up - low)

    def calculate_fitness(self,low,up):
        objective = self.calculate_objective(low,up)
        fitness = 0.5 * (np.tanh(8 * (1 - objective) - 4)) + 0.5
        return fitness


    def get_nodes(self):  # possible nodes to generate a MST ( all steiner pts, all terminals, obstacles with char 1)
        nodes = []
        nodes.extend(self.steinerpts)
        nodes.extend(self.terminals)
        obstacle_corners = self.get_obstacle_corners()
        for i in range(len(self.bins)):
            if self.bins[i] == '1':
                nodes.append(obstacle_corners[i])
        return nodes

    def cal_mst(self):
        # possible nodes to generate a MST ( all steiner pts, all terminals, obstacles with char 1)
        nodes = self.get_nodes()
        return self.prim(nodes)

    def cal_edge_weight(self, point1, point2):
        org_len = utils.cal_dis(point1, point2)
        cross_len = 0
        weight = 0
        for i in range(len(self.obstacles)):
            cur_len = utils.segment_in_obstacle_length(self.obstacles[i].points, (point1, point2))
            cross_len += cur_len
            weight += cur_len * self.obstacles[i].weight
        return weight + (org_len - cross_len)

    def prim(self, nodes):  # used to generate the MST
        num_nodes = len(nodes)
        visited = [False] * num_nodes
        #  Choose the first node as starting node
        visited[0] = True
        # PriorityQueue used to choose the next edge with minimum weight
        edges = [(self.cal_edge_weight(nodes[0], nodes[i]), 0, i) for i in range(1, num_nodes)]
        heapq.heapify(edges)
        weight_cache = {}

        mst = []  # store the edges of MST
        total_weight = 0

        while edges:
            weight, u, v = heapq.heappop(edges)
            if not visited[v]:
                visited[v] = True
                mst.append((u, v))
                total_weight += weight

                for next_v in range(num_nodes):
                    if not visited[next_v]:
                        if (v, next_v) not in weight_cache:
                            # 计算一次并缓存
                            weight_cache[(v, next_v)] = weight_cache[(next_v, v)] = self.cal_edge_weight(nodes[v],nodes[next_v])
                        next_weight = weight_cache[(v, next_v)]
                        #next_weight = self.cal_edge_weight(nodes[v], nodes[next_v])
                        heapq.heappush(edges, (next_weight, v, next_v))

        return mst, total_weight

    def get_obstacle_corners(self):
        obstacle_corners = []
        if not self.obstacles:
            return obstacle_corners
        for obstacle in self.obstacles:
            obstacle_corners.extend(obstacle.points)
        return obstacle_corners

    def flipMove(self, dis, fitness):
        # note for this method here in PPA we only flip or move one point
        # dis stands for average Euclidean distance between terminals
        spts = copy.deepcopy(self.steinerpts)
        bins = self.bins
        mrange = max(1 - fitness, 0.01) * dis
        xmove = np.random.uniform(0, mrange)
        ymove = np.random.uniform(0, mrange)
        s = len(spts)
        k = len(bins)
        random_idx= random.randint(0, s + k - 1)
        ret_chromosome = copy.deepcopy(self)
        if 0 <= random_idx <= s-1:
            loc = spts[random_idx]
            flagx_pos = 1
            flagy_pos = 1
            flagx_neg = -1
            flagy_neg = -1
            if loc[0] + xmove > 1:
                flagx_pos = 0
            if loc[1] + ymove > 1:
                flagy_pos = 0
            if loc[0] - xmove < 0:
                flagx_neg = 0
            if loc[1] - ymove < 0:
                flagy_neg = 0
            xdirs = [flagx_pos, flagx_neg]
            ydirs = [flagy_pos, flagy_neg]
            x_idx = utils.select_random_one_index(xdirs)
            y_idx = utils.select_random_one_index(ydirs)
            if not (x_idx == -1 and y_idx == -1):
                spts[random_idx] = (spts[random_idx][0] + xmove * xdirs[x_idx], spts[random_idx][1] + ymove * ydirs[y_idx])

        else:
            new_bins = ''
            str_idx = random_idx - s
            for i in range(len(bins)):
                if i==str_idx:
                    if bins[i] == '0':
                        new_bins += '1'
                    else:
                        new_bins += '0'
                else:
                    new_bins += bins[i]
            ret_chromosome.bins = new_bins


        # self.bins = bins

        ret_chromosome.steinerpts = spts

        # ret_chromosome = Chromosome(spts, bins, self.terminals, self.obstacles)
        # ret_chromosome.edges = self.edges
        # ret_chromosome.cost = self.cost
        ret_chromosome.mst, ret_chromosome.cost = ret_chromosome.cal_mst()
        ret_chromosome.nodes = ret_chromosome.get_nodes()
        ret_chromosome.fitness = ret_chromosome.calculate_fitness(self.low, self.up)
        return ret_chromosome


    def add_steiner_mutation(self):
        hard_obstacles = [ob.points for ob in self.obstacles if
                          ob.weight == sys.maxsize]  # hard_obstacles is a list of list (each list contains some
        # coordinates)
        """Performs the addSteiner mutation on the chromosome."""
        # new_chro = Chromosome(self.steinerpts, self.bins, self.terminals, self.obstacles)
        new_chro = copy.deepcopy(self)


        random_point = self.generate_random_steiner_point(hard_obstacles)
        new_chro.steinerpts.append(random_point)
        new_chro.mst, new_chro.cost = new_chro.cal_mst()
        new_chro.nodes = new_chro.get_nodes()
        new_chro.fitness = new_chro.calculate_fitness(self.low, self.up)
        return new_chro
    '''
    def remove_steiner_mutation(self):
        ret_chromosome = copy.deepcopy(self)
        if len(self.steinerpts) ==0:
            return ret_chromosome
        degrees = {node: 0 for node in range(len(self.nodes))}
        print('degrees ', degrees)
        print('Steiner points are',self.steinerpts)
        for edge in self.mst:
            if any(np.array_equal(self.nodes[edge[0]], pt) for pt in self.steinerpts):
                degrees[edge[0]] += 1
                print('find one')

            if any(np.array_equal(self.nodes[edge[1]], pt) for pt in self.steinerpts):
                degrees[edge[1]] += 1
                print('find one')
            
          
            
        degree_2_pts = [pt for pt, degree in degrees.items() if degree < 3]
        print('degree 2 pts ',degree_2_pts)



        # If there are no Steiner points with degree 2, return without doing anything
        if not degree_2_pts or len(degree_2_pts) == 0 :
            print('No degree 2 pts')
            return ret_chromosome

        idx = random.choice(degree_2_pts)
        while idx >= len(self.steinerpts):
            print(idx)
            idx = random.choice(degree_2_pts)
        print('to generate')
        pt_to_remove = tuple(self.steinerpts[idx] ) # random would generate  an index of steiner points
        ret_chromosome.steinerpts = [pt for pt in self.steinerpts if tuple(pt) != pt_to_remove]
        ret_chromosome.mst, ret_chromosome.cost = ret_chromosome.cal_mst()
        ret_chromosome.nodes = ret_chromosome.get_nodes()
        ret_chromosome.fitness = ret_chromosome.calculate_fitness(self.low, self.up)
        print('remove chromosome done')
        return ret_chromosome
        '''

    def remove_steiner_mutation(self):
        ret_chromosome = copy.deepcopy(self)
        if len(self.steinerpts) == 0:
            return ret_chromosome  # 如果没有斯坦纳点直接返回

        # 从节点列表构建索引映射到斯坦纳点
        steiner_indices = {idx for idx, node in enumerate(self.nodes) if node in self.steinerpts}

        # 计算所有节点的度
        degrees = {node: 0 for node in range(len(self.nodes))}
        for edge in self.mst:
            if any(np.array_equal(self.nodes[edge[0]], pt) for pt in self.steinerpts):
                degrees[edge[0]] += 1

            if any(np.array_equal(self.nodes[edge[1]], pt) for pt in self.steinerpts):
                degrees[edge[1]] += 1
            #degrees[edge[0]] += 1
            #degrees[edge[1]] += 1
        #print('degrees ', degrees)
        # 筛选出度小于3的斯坦纳点索引
        degree_2_pts = [idx for idx in steiner_indices if degrees[idx] < 3]
        if not degree_2_pts:
            return ret_chromosome  # 如果没有符合条件的斯坦纳点，直接返回

        # 随机选择一个度小于3的斯坦纳点进行移除
        idx_to_remove = random.choice(degree_2_pts)
        print('idx ',idx_to_remove)
        pt_to_remove = self.nodes[idx_to_remove]
        ret_chromosome.steinerpts = [pt for pt in self.steinerpts if pt != pt_to_remove]

        # 重新计算最小生成树和成本
        ret_chromosome.mst, ret_chromosome.cost = ret_chromosome.cal_mst()
        ret_chromosome.nodes = ret_chromosome.get_nodes()
        ret_chromosome.fitness = ret_chromosome.calculate_fitness(self.low, self.up)
        return ret_chromosome

    def calculate_angle(self, node, neighbor1, neighbor2):
        """Calculates the angle between three points."""
        # Fetch point coordinates
        p1 = np.array(self.nodes)[node]
        p2 = np.array(self.nodes)[neighbor1]
        p3 = np.array(self.nodes)[neighbor2]

        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p1

        # Avoid division by zero if vectors are zero vectors
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 100  # p2 and p3 are the same point
            # Calculate the cosine of the angle
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        # Ensure the cosine value is within the range [-1, 1]
        cosine_angle = np.clip(cosine_angle, -1, 1)

        # Calculate angle using arccos
        angle = np.arccos(cosine_angle)

        # Calculate angle using dot product
        #angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return angle

    def calculate_steiner_point(self, nodes):
        #todo what if the new point is in solid obstacle
        """Calculates the position of a new Steiner point."""
        # Here, we're just calculating the centroid of the triangle formed by the nodes.
        points = [np.array(self.nodes)[node] for node in nodes]
        centroid = np.mean(points, axis=0)
        return tuple(centroid)

    def generate_random_steiner_point(self, hard_obstacles):
        """Generates a random Steiner point avoiding placement inside  solid obstacles."""
        # Placeholder for random point generation. You need to implement obstacle avoidance.
        cur = np.random.random(), np.random.random()
        while utils.check_inside(hard_obstacles, cur):
            cur = np.random.random(), np.random.random()

        return cur
    def simpson_line(self):
        hard_obstacles = [ob.points for ob in self.obstacles if
                          ob.weight == sys.maxsize]
        ret_chromosome = copy.deepcopy(self)

        new_stpts = self.steinerpts
        degrees = {node: 0 for node in range(len(self.steinerpts))}
        for edge in self.mst:
            if self.nodes[edge[0]] in self.steinerpts:
                degrees[edge[0]] += 1
            if self.nodes[edge[1]] in self.steinerpts:
                degrees[edge[1]] += 1

        degree_3_pts = [pt for pt, degree in degrees.items() if degree == 3]

        # If there are no Steiner points with degree 2, return without doing anything
        if not degree_3_pts:
            print('No neighbours')
            return ret_chromosome
        for pt in degree_3_pts:
            neighbours = []
            for edge in self.mst:
                if edge[0] == pt:
                    neighbours.append(self.nodes[edge[1]])
                elif edge[1] == pt:
                    neighbours.append(self.nodes[edge[0]])
            print('neighbours', neighbours)
            new_pt = tuple(utils.fermat_torricelli_point(neighbours))
            if not utils.check_inside(hard_obstacles, new_pt):
                new_stpts.remove(self.nodes[pt])
                new_stpts.append(new_pt)
        ret_chromosome.steinerpts = new_stpts
        ret_chromosome.mst, ret_chromosome.cost = ret_chromosome.cal_mst()
        ret_chromosome.nodes = ret_chromosome.get_nodes()

        return ret_chromosome



# Example usage
