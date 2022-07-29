import numpy as np
import benchmark
from util import helps


func_num = 2
problem = benchmark.GTOPX_Problem(func_num)
obj, var, int_var, cons, LB, UB = problem.info()
func = problem.evaluate
print(func(helps.center(var, [LB, UB])))
