task_id = 2
growth_rate = [100, 32, 16, 8, 4, 2, 1, 1]

# Don't change anything above this line
# =====================================

# generate your solution as a list
queue = [(0, 1), (0, 2), (0, 1), (0,3), (0, 1), (0, 4), (0, 1), (0,5), (0, 1), (0, 6)]

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
