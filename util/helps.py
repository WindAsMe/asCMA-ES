import random
import geatpy as ea
from Optimizer import MyProblem
import copy


def random_population(LB, UB, int_var, size):
    Population = []
    Dim = len(LB)
    int_index = Dim - int_var
    for i in range(size):
        indi = []
        for j in range(Dim):
            if j >= int_index:
                indi.append(random.randint(LB[j], UB[j]))
            else:
                indi.append(random.uniform(LB[j], UB[j]))
        Population.append(indi)
    return Population


def initial_population(Dim, NIND, func, groups, scale_range, VarType):
    initial_Population = []
    based_population = center(Dim, scale_range)
    for i in range(len(groups)):
        sub_scale = sub_scale_range(groups[i], scale_range)
        problem = MyProblem.CC_Problem(groups[i], func, sub_scale, based_population, VarType[i])   # 实例化问题对象

        Encoding = 'RI'  # 编码方式

        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
        population = ea.Population(Encoding, Field, NIND * len(groups[i]))
        population.initChrom(NIND * len(groups[i]))
        population.Phen = copy.deepcopy(population.Chrom)
        problem.aimFunc(population)
        initial_Population.append(population)
    return initial_Population


def CV_Normalize(CV):
    for i in range(len(CV)):
        if CV[i] < 0:
            CV[i] = 0
    return CV


def sub_scale_range(group, scale_range):
    sub_scale = [[], []]
    for var in group:
        sub_scale[0].append(scale_range[0][var])
        sub_scale[1].append(scale_range[1][var])
    return sub_scale


def center(Dim, scale_range):
    point = []
    for i in range(Dim):
        point.append((scale_range[0][i] + scale_range[1][i]) / 2)
    return point


def define_VarType(groups, start):
    VarTypeS = []
    for group in groups:
        vartype = []
        for var in group:
            if var >= start:
                vartype.append(1)
            else:
                vartype.append(0)
        VarTypeS.append(vartype)
    return VarTypeS