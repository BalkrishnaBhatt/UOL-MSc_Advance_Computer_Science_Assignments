task_id = 0
growth_rate = [10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5]

# Don't change anything above this line
# =====================================

# generate your solution as a list
queue = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]

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
