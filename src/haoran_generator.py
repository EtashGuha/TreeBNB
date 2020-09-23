import os
import argparse
import numpy as np
import scipy.sparse
import sys, pickle
import networkx as nx
sys.path.append('..')

from itertools import combinations

def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed


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


class set_cover():
    def __init__(self, max_nrows, max_ncols, density, mc, seed):
        self.rng = np.random.RandomState(seed)
        backbone = f'data/instances/setcover_{density}d_{mc}mc_{seed}se/backbone.pkl'
        if os.path.isfile(backbone):
            with open(backbone, 'rb') as handle:
                self.A, self.c = pickle.load(handle)
        else:
            self.A, self.c = self.generate_A(max_nrows, max_ncols, density, mc, backbone)

    def generate_A(self, nrows, ncols, density, mc, backbone):
        nnzrs = int(nrows * ncols * density)

        assert nnzrs >= nrows  # at least 1 col per row
        assert nnzrs >= 2 * ncols  # at leats 2 rows per col

        # compute number of rows per column
        indices = rng.choice(ncols, size=nnzrs)  # random column indexes
        indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)  # force at leats 2 rows per col
        _, col_nrows = np.unique(indices, return_counts=True)

        # for each column, sample random rows
        indices[:nrows] = rng.permutation(nrows)  # force at least 1 column per row
        i = 0
        indptr = [0]
        for n in col_nrows:
            # empty column, fill with random rows
            if i >= nrows:
                indices[i:i + n] = rng.choice(nrows, size=n, replace=False)
            # partially filled column, complete with random rows among remaining ones
            elif i + n > nrows:
                remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows], assume_unique=True)
                indices[nrows:i + n] = rng.choice(remaining_rows, size=i + n - nrows, replace=False)
            i += n
            indptr.append(i)

        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(nrows, ncols)).todense()
        # objective coefficients
        c = rng.randint(mc, size=ncols) + 1
        with open(backbone, 'wb') as handle:
            pickle.dump([A, c], handle)
        return A, c

    def disturb_generator(self, ndisturb, filename):
        A = scipy.sparse.coo_matrix(self.A)
        c = self.c + self.rng.randint(2,size=self.c.shape)
        for _ in range(ndisturb):
            i, j = self.rng.permutation(A.getnnz())[:2]
            A.row[i], A.row[j] = A.row[j], A.row[i]
        A = A.tocsr()
        indices = A.indices
        indptr = A.indptr
        # write problem
        with open(filename, 'w') as file:
            file.write("minimize\nOBJ:")
            file.write("".join([f" +{c[j]} x{j + 1}" for j in range(ncols)]))

            file.write("\n\nsubject to\n")
            for i in range(nrows):
                row_cols_str = "".join([f" +1 x{j + 1}" for j in indices[indptr[i]:indptr[i + 1]]])
                file.write(f"C{i}:" + row_cols_str + f" >= 1\n")

            file.write("\nbinary\n")
            file.write("".join([f" x{j + 1}" for j in range(ncols)]))

    def sub_generator(self, nrows, ncols, filename):
        assert nrows <= self.A.shape[0]
        assert ncols <= self.A.shape[1]
        rows = self.rng.choice(np.arange(self.A.shape[0]), nrows, replace=False)
        cols = self.rng.choice(np.arange(self.A.shape[1]), ncols, replace=False)
        A = scipy.sparse.csr_matrix(self.A[rows][:,cols])
        c = self.c[cols] + self.rng.randint(2,size=ncols)
        indices = A.indices
        indptr = A.indptr
        # write problem
        with open(filename, 'w') as file:
            file.write("minimize\nOBJ:")
            file.write("".join([f" +{c[j]} x{j + 1}" for j in range(ncols)]))

            file.write("\n\nsubject to\n")
            for i in range(nrows):
                row_cols_str = "".join([f" +1 x{j + 1}" for j in indices[indptr[i]:indptr[i + 1]]])
                file.write(f"C{i}:" + row_cols_str + f" >= 1\n")

            file.write("\nbinary\n")
            file.write("".join([f" x{j + 1}" for j in range(ncols)]))


class facility():
    def __init__(self, n_c, n_f, ratio, seed):
        self.n_c = n_c
        self.n_f = n_f
        self.ratio = ratio
        self.rng = np.random.RandomState(seed)
        self.demands = self.rng.randint(10, 30, size=n_c)
        self.capacities = self.rng.randint(50, 120, size=n_f)
        backbone = f'data/instances/facility_{n_c}c_{n_f}f_{ratio}r_{seed}se/backbone.pkl'
        if os.path.isfile(backbone):
            with open(backbone, 'rb') as handle:
                self.backbone = pickle.load(handle)
        else:
            c_x = rng.rand(n_c)
            c_y = rng.rand(n_c)
            f_x = rng.rand(n_f)
            f_y = rng.rand(n_f)
            for i in range(1, n_c):
                if self.rng.rand() < 0.9:
                    k = self.rng.randint(i)
                    c_x[i] = 0.1 * c_x[i] + 0.9 * c_x[k]
                    c_y[i] = 0.1 * c_y[i] + 0.9 * c_y[k]
            self.backbone = [c_x, c_y, f_x, f_y]
            with open(backbone, 'wb') as handle:
                pickle.dump(self.backbone, handle)

    def generate(self, n_c, n_f, filename):
        assert n_c >= self.n_c
        assert n_f >= self.n_f
        c_x = np.concatenate((self.backbone[0], self.rng.rand(n_c - self.n_c)))
        c_y = np.concatenate((self.backbone[1], self.rng.rand(n_c - self.n_c)))
        f_x = np.concatenate((self.backbone[2], self.rng.rand(n_f - self.n_f)))
        f_y = np.concatenate((self.backbone[3], self.rng.rand(n_f - self.n_f)))
        for i in range(self.n_c, n_c):
            if self.rng.rand() < 0.9:
                k = self.rng.randint(i)
                c_x[i] = 0.1 * c_x[i] + 0.9 * c_x[k]
                c_y[i] = 0.1 * c_y[i] + 0.9 * c_y[k]
        demands = np.concatenate((self.demands + self.rng.randint(-5,5, size=self.n_c),
                                  self.rng.randint(5, 35 + 1, size=n_c-self.n_c)))
        capacities = np.concatenate((self.capacities+self.rng.randint(-20,20, size=self.n_f),
                                     self.rng.randint(30, 140 + 1, size=n_f-self.n_f)))
        fixed_costs = self.rng.randint(100, 110 + 1, size=n_f) * np.sqrt(capacities) + self.rng.randint(90 + 1, size=n_f)
        fixed_costs = fixed_costs.astype(int)

        total_demand = demands.sum()
        total_capacity = capacities.sum()

        # adjust capacities according to ratio
        capacities = capacities * self.ratio * total_demand / total_capacity
        capacities = capacities.astype(int)

        # transportation costs
        trans_costs = np.sqrt((c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

        # write problem
        with open(filename, 'w') as file:
            file.write("minimize\nobj:")
            file.write("".join(
                [f" +{trans_costs[i, j]} x_{i + 1}_{j + 1}" for i in range(n_c) for j in range(n_f)]))
            file.write("".join([f" +{fixed_costs[j]} y_{j + 1}" for j in range(n_f)]))

            file.write("\n\nsubject to\n")
            for i in range(n_c):
                file.write(
                    f"demand_{i + 1}:" + "".join([f" -1 x_{i + 1}_{j + 1}" for j in range(n_f)]) + f" <= -1\n")
            for j in range(n_f):
                file.write(f"capacity_{j + 1}:" + "".join([f" +{demands[i]} x_{i + 1}_{j + 1}" for i in
                                                           range(n_c)]) + f" -{capacities[j]} y_{j + 1} <= 0\n")

            # optional constraints for LP relaxation tightening
            file.write("total_capacity:" + "".join(
                [f" -{capacities[j]} y_{j + 1}" for j in range(n_f)]) + f" <= -{total_demand}\n")
            for i in range(n_c):
                for j in range(n_f):
                    file.write(f"affectation_{i + 1}_{j + 1}: +1 x_{i + 1}_{j + 1} -1 y_{j + 1} <= 0")

            file.write("\nbounds\n")
            for i in range(n_c):
                for j in range(n_f):
                    file.write(f"0 <= x_{i + 1}_{j + 1} <= 1\n")

            file.write("\nbinary\n")
            file.write("".join([f" y_{j + 1}" for j in range(n_f)]))


class FCNF():
    '''
    A Parallel Local Search Framework for the Fixed-Charge Multicommodity Network Flow Problem
    http://www.optimization-online.org/DB_FILE/2014/06/4410.pdf
    '''
    def __init__(self, n0, aff, d, u, f, q, seed):
        self.n0 = n0
        self.aff = aff
        self.d = d
        self.rng = np.random.RandomState(seed)
        backbone = f'data/instances/fcnf_{n0}n_{aff}a_{d}d_{u}u_{f}f_{q}q_{seed}se/backbone.pkl'
        if os.path.isfile(backbone):
            with open(backbone, 'rb') as handle:
                self.backbone = pickle.load(handle)
        else:
            graph = nx.barabasi_albert_graph(n0, aff)
            graph = graph.to_directed()
            attrs = {}
            for edge in graph.edges():
                attrs[edge] = {'f': self.rng.randint(f) + 1, 'u_p': self.rng.randint(int(u/2)+1, u),
                               'q': self.rng.randint(2, q)}
            nx.set_edge_attributes(graph, attrs)
            self.backbone = graph
            with open(backbone, 'wb') as handle:
                pickle.dump(self.backbone, handle)

    def generate_fcnf(self, nnodes, filename):
        assert nnodes <= self.n0
        node_list = self.rng.permutation(self.n0)[:nnodes]
        g = nx.subgraph(self.backbone, node_list)
        edge_attrs = {}
        for i, (s, t, d) in enumerate(g.edges(data=True)):
            d['id'] = i + 1
            d['q'] += self.rng.randint(-1, 2)
            edge_attrs[(s, t)] = d
        nx.set_edge_attributes(g, edge_attrs)

        degree = np.array([d[1] for d in list(g.degree)])
        degree = degree / degree.sum()
        s1, t1, s2, t2 = self.rng.choice(len(degree), 4, replace=False, p=degree)
        d2, d1 = sorted(self.rng.randint(int(self.d/2)+1, self.d, size=2))
        with open(filename, 'w') as lp_file:
            lp_file.write("minimize\nOBJ:" + "".join([f" + {d['q']} x{d['id']+1} + {d['f']} y{d['id'] + 1}" for i,j,d in range(g.edges(data=True))]) + "\n")
            lp_file.write("\nsubject to\n")
            for n in enumerate(g.nodes()):
                if n == s1:
                    demand = -d1
                elif n == t1:
                    demand = d1
                elif n == s2:
                    demand = -d2
                elif n == t2:
                    demand = d2
                else:
                    demand = 0
                lp_file.write(f"demand_{n+1}:" + "".join([f" + x{d['id']+1}" for i,j,d in g.in_edges(n, data=True)]) +
                              "".join([f"- x{d['id']+1}" for i,j,d in g.out_edges(n, data=True)]) + f" = {demand}\n")
            lp_file.write("".join([f"capacity_{i+1}_{j+1}:" + f"x{d['id']+1} <= {d['u']} y{d['id']+1}\n" for i,j,d in enumerate(g.edges(data=True))]))

            lp_file.write("\nbounds\n")
            lp_file.write("".join([f" 0<= x{d['id']+1} \n" for i,j,d in g.edges(data=True)]))
            lp_file.write("\nbinary\n")
            lp_file.write("".join([f" y{d['id']+1}" for i,j,d in g.edges(data=True)]) + "\n")


class indset():
    def __init__(self, n0, aff, seed):
        self.n0 = n0
        self.aff = aff
        self.rng = np.random.RandomState(seed)
        backbone = f'data/instances/indset_{n0}n_{aff}a_{seed}se/backbone.pkl'
        if os.path.isfile(backbone):
            with open(backbone, 'rb') as handle:
                self.backbone = pickle.load(handle)
        else:
            edges = set()
            degrees = np.zeros(n0, dtype=int)
            neighbors = {node: set() for node in range(n0)}
            for new_node in range(aff, n0):
                # first node is connected to all previous ones (star-shape)
                if new_node == affinity:
                    neighborhood = np.arange(new_node)
                # remaining nodes are picked stochastically
                else:
                    neighbor_prob = degrees[:new_node] / (2 * len(edges))
                    neighborhood = rng.choice(new_node, affinity, replace=False, p=neighbor_prob)
                for node in neighborhood:
                    edges.add((node, new_node))
                    degrees[node] += 1
                    degrees[new_node] += 1
                    neighbors[node].add(new_node)
                    neighbors[new_node].add(node)
            self.backbone = [edges, degrees, neighbors]
            with open(backbone, 'wb') as handle:
                pickle.dump(self.backbone, handle)

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


def generate_setcover(nrows, ncols, density, filename, rng, max_coef=100):
    """
    Generates a setcover instance with specified characteristics, and writes
    it to a file in the LP format.
    Approach described in:
    E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.
    Parameters
    ----------
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    filename: str
        File to which the LP will be written
    rng: numpy.random.RandomState
        Random number generator
    max_coef: int
        Maximum objective coefficient (>=1)
    """
    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    # compute number of rows per column
    indices = rng.choice(ncols, size=nnzrs)  # random column indexes
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)  # force at leats 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = rng.permutation(nrows) # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i:i+n] = rng.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows], assume_unique=True)
            indices[nrows:i+n] = rng.choice(remaining_rows, size=i+n-nrows, replace=False)

        i += n
        indptr.append(i)

    # objective coefficients
    c = rng.randint(max_coef, size=ncols) + 1

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(nrows, ncols)).tocsr()
    indices = A.indices
    indptr = A.indptr

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nOBJ:")
        file.write("".join([f" +{c[j]} x{j+1}" for j in range(ncols)]))

        file.write("\n\nsubject to\n")
        for i in range(nrows):
            row_cols_str = "".join([f" +1 x{j+1}" for j in indices[indptr[i]:indptr[i+1]]])
            file.write(f"C{i}:" + row_cols_str + f" >= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" x{j+1}" for j in range(ncols)]))


def generate_cauctions(random, filename, n_items=100, n_bids=500, min_value=1, max_value=100,
                       value_deviation=0.5, add_item_prob=0.9, max_n_sub_bids=5,
                       additivity=0.2, budget_factor=1.5, resale_factor=0.5,
                       integers=False, warnings=False):
    """
    Generate a Combinatorial Auction problem following the 'arbitrary' scheme found in section 4.3. of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.
    Saves it as a CPLEX LP file.
    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_items : int
        The number of items.
    n_bids : int
        The number of bids.
    min_value : int
        The minimum resale value for an item.
    max_value : int
        The maximum resale value for an item.
    value_deviation : int
        The deviation allowed for each bidder's private value of an item, relative from max_value.
    add_item_prob : float in [0, 1]
        The probability of adding a new item to an existing bundle.
    max_n_sub_bids : int
        The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
    additivity : float
        Additivity parameter for bundle prices. Note that additivity < 0 gives sub-additive bids, while additivity > 0 gives super-additive bids.
    budget_factor : float
        The budget factor for each bidder, relative to their initial bid's price.
    resale_factor : float
        The resale factor for each bidder, relative to their initial bid's resale value.
    integers : logical
        Should bid's prices be integral ?
    warnings : logical
        Should warnings be printed ?
    """

    assert min_value >= 0 and max_value >= min_value
    assert add_item_prob >= 0 and add_item_prob <= 1

    def choose_next_item(bundle_mask, interests, compats, add_item_prob, random):
        n_items = len(interests)
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return random.choice(n_items, p=prob)

    # common item values (resale price)
    values = min_value + (max_value - min_value) * random.rand(n_items)

    # item compatibilities
    compats = np.triu(random.rand(n_items, n_items), k=1)
    compats = compats + compats.transpose()
    compats = compats / compats.sum(1)

    bids = []
    n_dummy_items = 0

    # create bids, one bidder at a time
    while len(bids) < n_bids:

        # bidder item values (buy price) and interests
        private_interests = random.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # substitutable bids of this bidder
        bidder_bids = {}

        # generate initial bundle, choose first item according to bidder interests
        prob = private_interests / private_interests.sum()
        item = random.choice(n_items, p=prob)
        bundle_mask = np.full(n_items, 0)
        bundle_mask[item] = 1

        # add additional items, according to bidder interests and item compatibilities
        while random.rand() < add_item_prob:
            # stop when bundle full (no item left)
            if bundle_mask.sum() == n_items:
                break
            item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
            bundle_mask[item] = 1

        bundle = np.nonzero(bundle_mask)[0]

        # compute bundle price with value additivity
        price = private_values[bundle].sum() + np.power(len(bundle), 1 + additivity)
        if integers:
            price = int(price)

        # drop negativaly priced bundles
        if price < 0:
            if warnings:
                print("warning: negatively priced bundle avoided")
            continue

        # bid on initial bundle
        bidder_bids[frozenset(bundle)] = price

        # generate candidates substitutable bundles
        sub_candidates = []
        for item in bundle:

            # at least one item must be shared with initial bundle
            bundle_mask = np.full(n_items, 0)
            bundle_mask[item] = 1

            # add additional items, according to bidder interests and item compatibilities
            while bundle_mask.sum() < len(bundle):
                item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
                bundle_mask[item] = 1

            sub_bundle = np.nonzero(bundle_mask)[0]

            # compute bundle price with value additivity
            sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            if integers:
                sub_price = int(sub_price)

            sub_candidates.append((sub_bundle, sub_price))

        # filter valid candidates, higher priced candidates first
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        for bundle, price in [
            sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break

            if price < 0:
                if warnings:
                    print("warning: negatively priced substitutable bundle avoided")
                continue

            if price > budget:
                if warnings:
                    print("warning: over priced substitutable bundle avoided")
                continue

            if values[bundle].sum() < min_resale_value:
                if warnings:
                    print("warning: substitutable bundle below min resale value avoided")
                continue

            if frozenset(bundle) in bidder_bids:
                if warnings:
                    print("warning: duplicated substitutable bundle avoided")
                continue

            bidder_bids[frozenset(bundle)] = price

        # add XOR constraint if needed (dummy item)
        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        # place bids
        for bundle, price in bidder_bids.items():
            bids.append((list(bundle) + dummy_item, price))

    # generate the LP file
    with open(filename, 'w') as file:
        bids_per_item = [[] for item in range(n_items + n_dummy_items)]

        file.write("maximize\nOBJ:")
        for i, bid in enumerate(bids):
            bundle, price = bid
            file.write(f" +{price} x{i+1}")
            for item in bundle:
                bids_per_item[item].append(i)

        file.write("\n\nsubject to\n")
        for item_bids in bids_per_item:
            if item_bids:
                for i in item_bids:
                    file.write(f" +1 x{i+1}")
                file.write(f" <= 1\n")

        file.write("\nbinary\n")
        for i in range(len(bids)):
            file.write(f" x{i+1}")


def generate_capacited_facility_location(random, filename, n_customers, n_facilities, ratio):
    """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.
    Saves it as a CPLEX LP file.
    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """
    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    demands = rng.randint(5, 35+1, size=n_customers)
    capacities = rng.randint(10, 160+1, size=n_facilities)
    fixed_costs = rng.randint(100, 110+1, size=n_facilities) * np.sqrt(capacities) \
            + rng.randint(90+1, size=n_facilities)
    fixed_costs = fixed_costs.astype(int)

    total_demand = demands.sum()
    total_capacity = capacities.sum()

    # adjust capacities according to ratio
    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)
    total_capacity = capacities.sum()

    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nobj:")
        file.write("".join([f" +{trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
        file.write("".join([f" +{fixed_costs[j]} y_{j+1}" for j in range(n_facilities)]))

        file.write("\n\nsubject to\n")
        for i in range(n_customers):
            file.write(f"demand_{i+1}:" + "".join([f" -1 x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" <= -1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j+1}:" + "".join([f" +{demands[i]} x_{i+1}_{j+1}" for i in range(n_customers)]) + f" -{capacities[j]} y_{j+1} <= 0\n")

        # optional constraints for LP relaxation tightening
        file.write("total_capacity:" + "".join([f" -{capacities[j]} y_{j+1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"affectation_{i+1}_{j+1}: +1 x_{i+1}_{j+1} -1 y_{j+1} <= 0")

        file.write("\nbounds\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'fcfn', 'facility', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=valid_seed,
        default=0,
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    if args.problem == 'setcover':
        max_nrows = 1000
        max_ncols = 2000
        nrows = 500
        ncols = 1000
        dens = 0.05
        max_coef = 100

        dir = f'data/instances/setcover_{dens}d_{max_coef}mc_{args.seed}se/'
        if not os.path.isdir(dir):
            os.makedirs(dir)
        setcover_generator = set_cover(max_nrows, max_ncols, dens, max_coef, args.seed)

        filenames = []
        nrowss = []
        ncolss = []

        # train instances
        n = 500
        print(dir)
        lp_dir = os.path.join(dir, f'train_{nrows}r_{ncols}c/')
        print(lp_dir)
        print(f"{n} instances in {lp_dir}")
        os.mkdir(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)

        # validation instances
        n = 50
        lp_dir = os.path.join(dir, f'valid_{nrows}r_{ncols}c')
        print(f"{n} instances in {lp_dir}")
        os.mkdir(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)

        # test instances
        n = 30
        lp_dir = os.path.join(dir, f'test_{nrows}r_{ncols}c')
        print(f"{n} instances in {lp_dir}")
        os.mkdir(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)


        # actually generate the instances
        for filename, nrows, ncols in zip(filenames, nrowss, ncolss):
            print(f'  generating file {filename} ...')
            setcover_generator.sub_generator(nrows=nrows, ncols=ncols, filename=filename)

        # transfer instances
        n = 10
        lp_dir = os.path.join(dir, f'transfer_{max_nrows}r_{max_ncols}c')
        print(f"{n} instances in {lp_dir}")
        os.mkdir(lp_dir)
        filenames = [os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)]
        ndisturb = [100] * n

        # actually generate the instances
        for filename, n in zip(filenames, ndisturb):
            print(f'  generating file {filename} ...')
            setcover_generator.disturb_generator(ndisturb= n, filename=filename)

        print('done.')

    elif args.problem == 'indset':
        n0 = 3500
        affinity = 4
        dir = f'data/instances/indset_{n0}n_{affinity}a_{args.seed}se/'
        if not os.path.isdir(dir):
            os.makedirs(dir)
        indset_generator = indset(n0, affinity, args.seed)
        filenames = []
        nnodess = []

        number_of_nodes = 4000
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
        n = 10
        number_of_nodes = 600
        lp_dir = os.path.join(dir, f'transfer_{number_of_nodes}n/')
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        n = 10
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

    elif args.problem == 'fcfn':
        #todo: finish fcfn
        number_of_items = 100
        number_of_bids = 500
        filenames = []
        nitemss = []
        nbidss = []

        # train instances
        n = 10000
        lp_dir = f'data/instances/cauctions/train_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids ] * n)

        # validation instances
        n = 2000
        lp_dir = f'data/instances/cauctions/valid_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids ] * n)

        # small transfer instances
        n = 100
        number_of_items = 100
        number_of_bids = 500
        lp_dir = f'data/instances/cauctions/transfer_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids ] * n)

        # medium transfer instances
        n = 100
        number_of_items = 200
        number_of_bids = 1000
        lp_dir = f'data/instances/cauctions/transfer_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids ] * n)

        # big transfer instances
        n = 100
        number_of_items = 300
        number_of_bids = 1500
        lp_dir = f'data/instances/cauctions/transfer_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids ] * n)

        # test instances
        n = 2000
        number_of_items = 100
        number_of_bids = 500
        lp_dir = f'data/instances/cauctions/test_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids ] * n)

        # actually generate the instances
        for filename, nitems, nbids in zip(filenames, nitemss, nbidss):
            print(f"  generating file {filename} ...")
            generate_cauctions(rng, filename, n_items=nitems, n_bids=nbids, add_item_prob=0.7)

        print("done.")

    elif args.problem == 'facility':
        n_c0 = 100
        n_f0 = 50
        ratio = 5
        dir = f'data/instances/facility_{n_c0}c_{n_f0}f_{ratio}r_{args.seed}se'
        if not os.path.isdir(dir):
            os.makedirs(dir)
        facility_generator = facility(n_c0, n_f0, ratio, args.seed)

        n_c = 100
        n_f = 50
        filenames = []
        ncustomerss = []
        nfacilitiess = []

        # train instances
        n = 20
        lp_dir = os.path.join(dir, f'train_{n_c}c_{n_f}f')
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([n_c] * n)
        nfacilitiess.extend([n_f] * n)

        # validation instances
        n = 20
        lp_dir = os.path.join(dir, f'valid_{n_c}c_{n_f}f')
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([n_c] * n)
        nfacilitiess.extend([n_f] * n)

        # test instances
        n = 10
        lp_dir = os.path.join(dir, f'test_{n_c}c_{n_f}f')
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        ncustomerss.extend([n_c] * n)
        nfacilitiess.extend([n_f] * n)

        # transfer instances
        n = 10
        n_c = 200
        n_f = 100
        lp_dir = os.path.join(dir, f'transfer_{n_c}c_{n_f}f')
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        ncustomerss.extend([n_c] * n)
        nfacilitiess.extend([n_f] * n)

        # transfer instances
        n = 10
        n_c = 400
        n_f = 200
        lp_dir = os.path.join(dir, f'transfer_{n_c}c_{n_f}f')
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        ncustomerss.extend([n_c] * n)
        nfacilitiess.extend([n_f] * n)


        # actually generate the instances
        for filename, ncs, nfs in zip(filenames, ncustomerss, nfacilitiess):
            print(f"  generating file {filename} ...")
            facility_generator.generate(ncs, nfs, filename)

        print("done.")
