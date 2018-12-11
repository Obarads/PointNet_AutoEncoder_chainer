import numpy as np

v = np.array((
    [[2,3],
    [5,6],
    [1,3],
    [4,9],
    [7,4]
]))
choice = np.random.choice(len(v), 10, replace=True)
print(v[choice,:])
print(choice)