from UFL import *

file = "MO1"
u = UFL_Problem.readInstance(file)
u.solve()
print('eof')
