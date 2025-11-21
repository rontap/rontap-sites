from UFL import *

file = "MO4"
u = UFL_Problem.readInstance(file)
u.solve()
print('eof')
