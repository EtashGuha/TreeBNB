from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel

problem = "../realsingle/instance_9.lp"
model = Model("setcover")
model.setIntParam('separating/maxroundsroot', 0)
model.setBoolParam('conflict/enable', False)
model.readProblem(problem)
model.optimize()