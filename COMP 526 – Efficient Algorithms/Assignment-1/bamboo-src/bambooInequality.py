task_id = 1
growth_rate = [98, 98, 1, 1, 1, 1]

# Don't change anything above this line
# =====================================

# generate your solution as a list
queue = [(0, 1), (2, 3),(0, 1), (4, 5)]

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
