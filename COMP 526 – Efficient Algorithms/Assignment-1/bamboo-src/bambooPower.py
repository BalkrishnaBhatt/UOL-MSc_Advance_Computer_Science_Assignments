task_id = 3
growth_rate = [96, 54, 54, 48, 24, 18, 18, 12, 6, 6, 6, 3, 3, 2, 2, 2]

# Don't change anything above this line
# =====================================

# generate your solution as a list
#queue=[(0, 1),(2, 3),(0,15),(1,14), (4, 5),(0,2), (3,13),(1,12),(0,11),(2,6),(3,0),(1,4),(7,10),(0,2),(5,1),(3,8),(0, 9)]
queue=[(0, 1),(2, 3),(0,15),(1,14), (4, 5),(0,2), (3,13),(1,12),(0,11),(2,6),(3,0),(1,4),(7,10),(0,2),(1,8),(3, 9)]

# =====================================
# Don't change anything below this line

from collections import deque

solution = deque()
# add each element to the solution
for i in queue:
    solution.append(i)

import bamboo

# records your solution
bamboo.calculate_height(growth_rate, solution, task_id)
