from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel
from NodeSel import MyNodesel, LinNodesel, SamplerNodesel
from nodeutil import getListOptimalID, checkIsOptimal
import torch
import glob
import pickle
from utilities import init_scip_params, init_scip_params_haoran, personalize_scip

class Sampler():
    def __init__(self, time_limit = 450):
        self.dataset = []
        self.time_limit = time_limit

    def solveModel(self, problem, train=True, default=False):
        temp_features = []
        self.model = Model("setcover")

        self.nodesel = SamplerNodesel(self.model, dataset=temp_features)
        self.model.includeNodesel(self.nodesel, "nodesel", "My node selection", 999999, 999999)
        personalize_scip(self.model, 10)
        self.model.setRealParam('limits/time', self.time_limit)
        self.model.readProblem(problem)
        self.model.optimize()
        return temp_features

    def collect(self, problem_dir):
        problems = glob.glob(problem_dir + "/*.lp")
        for problem in problems:
            temp_features = self.solveModel(problem)

            optimal_node = None
            for node in self.nodesel.tree.leaves():
                if checkIsOptimal(node, self.model, self.nodesel.tree):
                    optimal_node = node
                    break

            if optimal_node is not None:

                optimal_ids = getListOptimalID(optimal_node.identifier, self.nodesel.tree)
                for i in range(len(temp_features)):
                    queue_contains_optimal = False
                    optimal_id = None
                    dgl_tree, step_ids = temp_features[i]
                    idlist = step_ids.flatten().tolist()
                    for id in idlist:
                        if id in optimal_ids:
                            queue_contains_optimal = True
                            optimal_id = id
                            break
                    if queue_contains_optimal:
                        oracle_val = (step_ids== optimal_id).type(torch.uint8).nonzero()[0][0]
                        weight = 1/len(dgl_tree)
                        self.dataset.append((dgl_tree, oracle_val, weight))

            with open("../data/instances/setcover/100_200samples/100_200.pkl", "wb") as f:
                pickle.dump(self.dataset, f)

if __name__ == "__main__":
    sampler = Sampler()
    sampler.collect("../data/instances/setcover/test_100r_200c_0.1d_5mc_10se")


