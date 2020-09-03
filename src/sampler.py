from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel
from NodeSel import MyNodesel, LinNodesel, SamplerNodesel, SamplerLinNodesel
from nodeutil import getListOptimalID, checkIsOptimal
import torch
import glob
import pickle
from utilities import init_scip_params, init_scip_params_haoran, personalize_scip
import os
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

            with open("../data/instances/setcover/train_500r_1000c_0.05d_100mc_0se/sample.pkl", "wb") as f:
                pickle.dump(self.dataset, f)


class rankSampler():
    def __init__(self, time_limit=450):
        self.dataset = []
        self.time_limit = time_limit
        self.listNNodes = []

    def solveModel(self, problem, train=True, default=False):
        temp_features = []
        self.model = Model("setcover")
        self.model.setRealParam('limits/time', self.time_limit)
        if not default:
            self.ourNodeSel = SamplerLinNodesel(self.model, dataset=temp_features)
            self.model.includeNodesel(self.ourNodeSel, "nodesel", "My node selection", 999999, 999999)
        self.model.readProblem(problem)
        self.model.optimize()
        return temp_features


    def addTreeData(self, temp_features,num_past = 1500):
        num_right = 0
        num_cases = 0
        optimal_node = None
        # ourNodeSel.tree.show(data_property="variables")
        for node in self.ourNodeSel.tree.leaves():
            if checkIsOptimal(node, self.model, self.ourNodeSel.tree):
                optimal_node = node
                print("FOUND OPTIMal")

        if optimal_node is None:
            return None, 0, 0

        optimal_ids = getListOptimalID(optimal_node.identifier, self.ourNodeSel.tree)

        for idToFeature in temp_features:
            ids = idToFeature.keys()
            for id in ids:
                if id in optimal_ids:
                    for otherid in ids:
                        if id == otherid:
                            continue
                        self.sfeature_list.append(idToFeature[id] - idToFeature[otherid])
                        self.soracle.append(torch.tensor([1], dtype=torch.float32));
                        self.sfeature_list.append(idToFeature[otherid] - idToFeature[id])
                        self.soracle.append(torch.tensor([-1], dtype=torch.float32));
            samples = list(zip(self.sfeature_list, self.soracle))[-1 * num_past:]
        return samples, num_right, num_cases

    def collect(self, problem_dir):
        problems = glob.glob(problem_dir + "/*.lp")


        for problem in problems:
            temp_features = self.solveModel(problem)
            self.listNNodes.append(self.model.getNNodes())
            print(self.listNNodes)
            samples, num_right, num_cases_predict = self.addTreeData(temp_features)
            total_samples = []
            if os.path.exists(problem_dir + "/sample_rank.pkl"):
                with open(problem_dir + "/sample_rank.pkl", "wb") as f:
                    total_samples = pickle.load(f)
            total_samples.extend(samples)
            with open(problem_dir + "/sample_rank.pkl", "wb") as f:
                pickle.dump(total_samples, f)
            total_samples = []

if __name__ == "__main__":
    # sampler = Sampler()
    #
    # sampler.collect("../data/instances/setcover/train_500r_1000c_0.05d_100mc_0se")
    rank_sampler = rankSampler()
    rank_sampler.collect("../realsingle")

