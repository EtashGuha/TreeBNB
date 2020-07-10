from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel

problem = "../realsingle/instance_9.lp"
model = Model("setcover")
model.readProblem(problem)
model.solveConcurrent()