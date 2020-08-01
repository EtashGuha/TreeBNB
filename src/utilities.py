from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule
import torch
import numpy as np

def probing_features_extraction(model, idx, branch_cand, rounding_direction):
    """ get probing features """
    if rounding_direction == 'up':
        rounding = np.ceil
    elif rounding_direction == 'down':
        rounding = np.floor
    else:
        raise Exception(f'Expect rounding direction up or down but get {rounding_direction}')
    lp_objval = model.getLPObjVal()
    assert not model.inRepropagation()
    assert not model.inProbing()
    model.startProbing()
    model.fixVarProbing(branch_cand[0][idx], int(rounding(branch_cand[1][idx])))
    model.constructLP()
    model.solveProbingLP()
    score = max(model.getLPObjVal(), lp_objval)
    gain = score - lp_objval
    sol = get_solution(model)
    var_UB, var_LB = get_bound(model)
    model.endProbing()
    return gain, score, sol, var_UB, var_LB

def sorted_cols_list(model):
    cols = model.getLPColsData()
    return cols
    if model.data[4] == 'lp':
        return cols
    elif model.data[4] == 'mps':
        cols_list = [cols[i + 1] for i in range(0, len(cols) - 2)]
        cols_list.append(cols[0])
        cols_list.append(cols[-1])
        return cols_list
    else:
        raise Exception('invalid file format')

def get_solution(model):
    cols_list = sorted_cols_list(model)
    # test_lp(model)
    # print([col.getPrimsol() for col in cols_list])
    return torch.tensor([col.getPrimsol() for col in cols_list]).view(-1, 1)


def get_bound(model):
    cols_list = sorted_cols_list(model)
    l = torch.tensor([cols_list[i].getLb() for i in range(len(cols_list))]).view(-1, 1)
    u = torch.tensor([cols_list[i].getUb() for i in range(len(cols_list))]).view(-1, 1)
    return u, l


def init_scip_params(model, seed, heuristics=True, presolving=True,
                     separating_root=True, conflict=True, propagation=True, separating=True):

    seed = seed % 2147483648  # SCIP seed range

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # no restart
    model.setIntParam('presolving/maxrestarts', 0)

    # disable separation except the root
    if not separating:
        model.setIntParam('separating/maxrounds', 0)

    # if asked, disable separating (cuts)
    if not separating_root:
        model.setIntParam('separating/maxroundsroot', 0)
        # model.setSeparating(SCIP_PARAMSETTING.OFF)

    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)
        # model.setPresolve(SCIP_PARAMSETTING.OFF)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(SCIP_PARAMSETTING.OFF)

    # if asked, disable propagation heuristics
    # if not propagation:
    #     model.disablePropagation(False)
    #
    # model.setParam('branching/relpscost/maxdepth', 0)
    # model.setParam('branching/pscost/maxdepth', 0)
    # model.setParam('branching/inference/maxdepth', 0)
    # model.setParam('branching/mostinf/maxdepth', 0)
    # model.setParam('branching/allfullstrong/maxdepth', 0)
    # # model.setParam('branching/allfullstrong/priority', 99999)
    # model.setParam('branching/cloud/maxdepth', 0)
    # model.setParam('branching/fullstrong/maxdepth', 0)
    # model.setParam('branching/fullstrong/priority', 999999)
    # model.setParam('branching/leastinf/maxdepth', 0)
    # model.setParam('branching/lookahead/maxdepth', 0)
    # model.setParam('branching/multaggr/maxdepth', 0)
    # model.setParam('branching/nodereopt/maxdepth', 0)
    # model.setParam('branching/random/maxdepth', 0)
    # model.setParam('branching/distribution/maxdepth', 0)
    # model.setBoolParam('branching/vanillafullstrong/scoreall', True)
    # model.setParam('branching/vanillafullstrong/priority', 99999)
    # model.setParam('branching/vanillafullstrong/maxdepth', 0)
    # model.setParam('display/freq', 1)

    # if method == 'relp':
    #     print('The branching method is relp')
    #     model.setParam('branching/relpscost/maxdepth', -1)
    # elif method == 'FSB':
    #     print('The branching method is FSB')
    #     model.setParam('branching/fullstrong/maxdepth', -1)
    # elif method == 'vali':
    #     print('The branching method is vali')
    #     model.setParam('branching/vanillafullstrong/maxdepth', 0)
    # elif method == 'PDB':
    #     # print('The branching method is PDB')
    #     pass
    # else:
    #     raise Exception('invalid input branching method')
