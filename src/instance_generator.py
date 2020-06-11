
import numpy as np
import scipy
import argparse
import os
import scipy.sparse

def generate_setcover(nrows, ncols, density, filename, filename_pkl, rng, max_coef=100):
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
            # indices[i:i+n] = np.concatenate((rng.choice(nrows, size=int(n*0.55), replace=False), rng.choice(int(nrows*0.1), size=n-int(n*0.55), replace=False)),axis=0)
            indices[i:i+n] = rng.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows], assume_unique=True)
            indices[nrows:i+n] = rng.choice(remaining_rows, size=i+n-nrows, replace=False)

        i += n
        indptr.append(i)

    # A = np.random.binomial(1, density - 1/ncols, (nrows, ncols))
    # row_idx = [i for i in range(nrows)]
    # col_idx = np.random.choice([i for i in range(ncols)], nrows)
    # A[row_idx, col_idx] = 1
    # objective coefficients
    if max_coef == 0:
        c = 1
    else:
        c = rng.randint(max_coef, size=ncols) + 1 #col_nrows
    #
    # # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
        (np.ones(len(indices), dtype=int), indices, indptr),
        shape=(nrows, ncols)).tocsr()
    print(c.shape)
    c = np.asarray(np.sum(A.todense(), axis=0)).squeeze(axis=0)
    print(ncols)

    print(c.shape)

    indices = A.indices
    indptr = A.indptr
    #
    # b = np.ones((nrows, 1))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        default=0,
    )

    args = parser.parse_args()
    print(args.seed)
    rng = np.random.RandomState(int(args.seed))

    if args.problem == 'setcover':
        nrows = 120
        ncols = 240
        dens = 0.05

        max_coef = 10

        filenames = []
        filenames_pkl = []
        nrowss = []
        ncolss = []
        denss = []
        lp_dirs = []

        # train instances
        n_train= 2000
        for i in range(n_train):
            lp_dir =  f'../data/instances/setcover/train_{nrows}r_{ncols}c_{dens}d_{max_coef}mc_{args.seed}se'
            try:
                os.makedirs(lp_dir)
            except:
                pass
            lp_dirs.append(lp_dir)
        filenames.extend([os.path.join(lp_dirs[i], f'instance_{i+1}.lp') for i in range(n_train)])
        filenames_pkl.extend([os.path.join(lp_dirs[i], f'instance_{i+1}.pkl') for i in range(n_train)])
        nrowss.extend([nrows] * n_train)
        ncolss.extend([ncols] * n_train)
        denss.extend([dens] * n_train)

        # validation instances
        n_valid = 200
        for i in range(n_valid):
            lp_dir = f'../data/instances/setcover/valid_{nrows}r_{ncols}c_{dens}d_{max_coef}mc_{args.seed}se'
            if not os.path.exists((lp_dir)):
                os.makedirs(lp_dir)
            lp_dirs.append(lp_dir)
        # print(f"{n} instances in {lp_dir}")
        filenames.extend([os.path.join(lp_dirs[n_train+i], f'instance_{i+1}.lp') for i in range(n_valid)])
        filenames_pkl.extend([os.path.join(lp_dirs[n_train+i], f'instance_{i+1}.pkl') for i in range(n_valid)])
        nrowss.extend([nrows] * n_valid)
        ncolss.extend([ncols] * n_valid)
        denss.extend([dens] * n_valid)

        # test instances
        n_test = 200
        for i in range(n_test):
            lp_dir = f'../data/instances/setcover/test_{nrows}r_{ncols}c_{dens}d_{max_coef}mc_{args.seed}se'
            if not os.path.exists((lp_dir)):
                os.makedirs(lp_dir)
            lp_dirs.append(lp_dir)
        # print(f"{n} instances in {lp_dir}")
        filenames.extend([os.path.join(lp_dirs[n_train+n_valid+i], f'instance_{i+1}.lp') for i in range(n_test)])
        filenames_pkl.extend([os.path.join(lp_dirs[n_train+n_valid+i], f'instance_{i+1}.pkl') for i in range(n_test)])
        nrowss.extend([nrows] * n_test)
        ncolss.extend([ncols] * n_test)
        denss.extend([dens] * n_test)

        # medium transfer instances
        n_medium = 100
        nrows = 1000
        for i in range(n_medium):
            lp_dir = f'../data/instances/setcover/test_{nrows}r_{ncols}c_{dens}d_{max_coef}mc_{args.seed}se'
            if not os.path.exists((lp_dir)):
                os.makedirs(lp_dir)
            lp_dirs.append(lp_dir)
        # print(f"{n} instances in {lp_dir}")
        filenames.extend([os.path.join(lp_dirs[n_train + n_valid + n_test + i], f'instance_{i + 1}.lp') for i in range(n_medium)])
        filenames_pkl.extend(
            [os.path.join(lp_dirs[n_train + n_valid + n_test + i], f'instance_{i + 1}.pkl') for i in range(n_medium)])
        nrowss.extend([nrows] * n_medium)
        ncolss.extend([ncols] * n_medium)
        denss.extend([dens] * n_medium)

        # big transfer instances
        n_large = 100
        nrows = 2000
        for i in range(n_large):
            lp_dir = f'../data/instances/setcover/test_{nrows}r_{ncols}c_{dens}d_{max_coef}mc_{args.seed}se'
            try:
                os.makedirs(lp_dir)
            except:
                pass
            lp_dirs.append(lp_dir)
        # print(f"{n} instances in {lp_dir}")
        filenames.extend([os.path.join(lp_dirs[n_train + n_valid + n_test + n_medium + i], f'instance_{i + 1}.lp') for i in range(n_large)])
        filenames_pkl.extend(
            [os.path.join(lp_dirs[n_train + n_valid + n_test + n_medium + i], f'instance_{i + 1}.pkl') for i in range(n_large)])
        nrowss.extend([nrows] * n_large)
        ncolss.extend([ncols] * n_large)
        denss.extend([dens] * n_large)


        # actually generate the instances
        for filename, filename_pkl, nrows, ncols, dens in zip(filenames, filenames_pkl, nrowss, ncolss, denss):
            print(f'  generating file {filename} ...')
            generate_setcover(nrows=nrows, ncols=ncols, density=dens, filename=filename, filename_pkl=filename_pkl, rng=rng, max_coef=max_coef)

        print('done.')

