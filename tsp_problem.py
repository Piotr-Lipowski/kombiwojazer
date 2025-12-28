import tsplib95
import numpy as np
import math

class TSPProblem:
    def __init__(self, file_path):
        self.problem = tsplib95.load(file_path)
        self.name = self.problem.name
        self.dimension = self.problem.dimension
        self.coords = self._get_coords()
        self.distance_matrix = self._create_distance_matrix()

    def _get_coords(self):
        coords = {}
        # tsplib95 uses 1-based indexing for nodes usually
        nodes = list(self.problem.get_nodes())
        for i, node in enumerate(nodes):
            coords[i] = self.problem.node_coords[node]
        return coords

    def _calculate_distance(self, node1_idx, node2_idx):
        nodes = list(self.problem.get_nodes())
        u_name = nodes[node1_idx]
        v_name = nodes[node2_idx]
        
        u = self.problem.node_coords[u_name]
        v = self.problem.node_coords[v_name]
        
        if self.problem.edge_weight_type == 'EUC_2D':
            xd = u[0] - v[0]
            yd = u[1] - v[1]
            return int(math.sqrt(xd*xd + yd*yd) + 0.5)
        
        elif self.problem.edge_weight_type == 'ATT':
            xd = u[0] - v[0]
            yd = u[1] - v[1]
            rij = math.sqrt((xd*xd + yd*yd) / 10.0)
            tij = int(rij + 0.5)
            if tij < rij:
                dij = tij + 1
            else:
                dij = tij
            return dij
            
        return self.problem.get_weight(u_name, v_name)

    def _create_distance_matrix(self):
        matrix = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    matrix[i][j] = self._calculate_distance(i, j)
        return matrix

    def get_total_distance(self, route):
        distance = 0
        for i in range(len(route)):
            distance += self.distance_matrix[route[i]][route[(i + 1) % len(route)]]
        return distance
