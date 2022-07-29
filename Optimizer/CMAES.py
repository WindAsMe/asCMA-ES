import numpy as np
from cmaes import CMA
from Optimizer import MyProblem


def CMAES_exe(Dim, max_iter, NIND, benchmark, scale_range, VarType):
    obj_trace = []
    best_cv = None
    problem = MyProblem.Problem(Dim, benchmark, scale_range, obj_trace, VarType)
    SIGMA_MAX = 1e32

    optimizer = CMA(mean=np.zeros(Dim), bounds=np.array(scale_range).T, sigma=1.3, population_size=NIND)
    for generation in range(max_iter):
        solutions = []
        Obj = []
        CV = []
        indis = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            x = individual_verify(x, VarType, scale_range)
            obj, cv = benchmark(x)
            for c in cv:
                if c > 0:
                    obj = [SIGMA_MAX]
            solutions.append((x, obj[0]))
            indis.append(x)
            Obj.append(obj)
            CV.append(cv)

        best_indi = np.argmin(Obj)
        obj_trace.append(Obj[best_indi][0])
        best_cv = CV[best_indi]
        optimizer.tell(solutions)
    return obj_trace, best_cv


def individual_verify(individual, VarType, scale_range):
    Dim = len(individual)
    for i in range(Dim):
        if VarType[i] == 1:
            individual[i] = int(individual[i])
            if individual[i] < scale_range[0][i]:
                individual[i] = scale_range[0][i]
            if individual[i] > scale_range[1][i]:
                individual[i] = scale_range[1][i]
    return individual





