from UFL import *

file = "MO5"
u = UFL_Problem.readInstance(file)
u.solve()
print('eof')
