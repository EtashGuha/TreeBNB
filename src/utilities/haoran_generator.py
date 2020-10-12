

def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed
import os
import argparse
import numpy as np
import scipy.sparse
import sys, pickle
import networkx as nx
from itertools import combinations
sys.path.append('..')
import utilities as utils



class Graph():
    """
    Container for a graph.
    Parameters
    ----------
    number_of_nodes : int
        The number of nodes in the graph.
    edges : set of tuples (int, int)
        The edges of the graph, where the integers refer to the nodes.
    degrees : numpy array of integers
        The degrees of the nodes in the graph.
    neighbors : dictionary of type {int: set of ints}
        The neighbors of each node in the graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        """
        The number of nodes in the graph.
        """
        return self.number_of_nodes

    def greedy_clique_partition(self):
        """
        Partition the graph into cliques using a greedy algorithm.
        Returns
        -------
        list of sets
            The resulting clique partition.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def barabasi_albert(backbone, number_of_nodes, affinity, random):
        """
        Generate a Barabási-Albert random graph with a given edge probability.
        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        affinity : integer >= 1
            The number of nodes each new node will be attached to, in the sampling scheme.
        random : numpy.random.RandomState
            A random number generator.
        Returns
        -------
        Graph
            The generated graph.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges0, degrees0, neighbors0 = backbone
        edges = edges0.copy()
        degrees = np.zeros(number_of_nodes, dtype=int)
        degrees[:degrees0.shape[0]] = degrees0
        neighbors = {node: set() for node in range(number_of_nodes)}
        for n in neighbors0:
            neighbors[n] = neighbors0[n].copy()
        for new_node in range(degrees0.shape[0], number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability, random):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.
        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        edge_probability : float in [0,1]
            The probability of generating each edge.
        random : numpy.random.RandomState
            A random number generator.
        Returns
        -------
        Graph
            The generated graph.
        """
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class indset():
    def __init__(self, n0, aff, seed):
        self.n0 = n0
        self.aff = aff
        self.rng = np.random.RandomState(seed)
        backbone = f'../data/instances/indset_{n0}n_{aff}a_{seed}se/backbone.pkl'
        if os.path.isfile(backbone):
            with open(backbone, 'rb') as handle:
                self.backbone = pickle.load(handle)
        else:
            print("BLAH")
            #
            # edges = set()
            # degrees = np.zeros(n0, dtype=int)
            # neighbors = {node: set() for node in range(n0)}
            # for new_node in range(aff, n0):
            #     # first node is connected to all previous ones (star-shape)
            #     if new_node == aff:
            #         neighborhood = np.arange(new_node)
            #     # remaining nodes are picked stochastically
            #     else:
            #         neighbor_prob = degrees[:new_node] / (2 * len(edges))
            #         neighborhood = self.rng.choice(new_node, aff, replace=False, p=neighbor_prob)
            #     for node in neighborhood:
            #         edges.add((node, new_node))
            #         degrees[node] += 1
            #         degrees[new_node] += 1
            #         neighbors[node].add(new_node)
            #         neighbors[new_node].add(node)
            # self.backbone = [edges, degrees, neighbors]
            # with open(backbone, 'wb') as handle:
            #     pickle.dump(self.backbone, handle)

    def generate_indset(self, n, filename):
        """
        Generate a Maximum Independent Set (also known as Maximum Stable Set) instance
        in CPLEX LP format from a previously generated graph.
        Parameters
        ----------
        graph : Graph
            The graph from which to build the independent set problem.
        filename : str
            Path to the file to save.
        """
        graph = Graph.barabasi_albert(self.backbone, n, self.aff, self.rng)
        cliques = graph.greedy_clique_partition()
        inequalities = set(graph.edges)
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        # Put trivial inequalities for nodes that didn't appear
        # in the constraints, otherwise SCIP will complain
        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(len(graph)):
            if node not in used_nodes:
                inequalities.add((node,))

        with open(filename, 'w') as lp_file:
            lp_file.write("maximize\nOBJ:" + "".join([f" + 1 x{node+1}" for node in range(len(graph))]) + "\n")
            lp_file.write("\nsubject to\n")
            for count, group in enumerate(inequalities):
                lp_file.write(f"C{count+1}:" + "".join([f" + x{node+1}" for node in sorted(group)]) + " <= 1\n")
            lp_file.write("\nbinary\n" + " ".join([f"x{node+1}" for node in range(len(graph))]) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'fcnf', 'facility', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=utils.valid_seed,
        default=0,
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    if args.problem == 'indset':
        n0 = 400
        affinity = 4
        dir = f'../data/instances/indset_{n0}n_{affinity}a_{args.seed}se/'
        if not os.path.isdir(dir):
            os.makedirs(dir)
        indset_generator = indset(n0, affinity, args.seed)
        filenames = []
        nnodess = []

        number_of_nodes = 600
        # train instances
        n = 500
        lp_dir = os.path.join(dir, f'train_{number_of_nodes}n/')
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # validation instances
        n = 50
        lp_dir = os.path.join(dir, f'valid_{number_of_nodes}n/')
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # test instances
        n = 50
        lp_dir = os.path.join(dir, f'test_{number_of_nodes}n/')
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # transfer
        n = 2
        number_of_nodes = 1000
        lp_dir = os.path.join(dir, f'transfer_{number_of_nodes}n/')
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        n = 2
        number_of_nodes = 1500
        lp_dir = os.path.join(dir, f'transfer_{number_of_nodes}n/')
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # actually generate the instances

        for filename, nnodes in zip(filenames, nnodess):
            print(f"  generating file {filename} ...")
            indset_generator.generate_indset(nnodes, filename)

        print("done.")