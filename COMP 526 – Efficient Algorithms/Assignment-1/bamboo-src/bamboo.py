from collections import deque

csvSplitBy = ","
NUMBER_OF_ITERATIONS = 1000000
csvFile = "maxheights.csv"
numberPlots = 5


def write_results(task_id: int, max_height: int, csv_file: str):
    height_string = str(max_height)
    try:
        file = open(csvFile)
        line = file.readline().strip()
        fields = line.split(sep=csvSplitBy)
        if len(fields) != numberPlots:
            raise
        results = fields[0:numberPlots]
    except:
        # ignore, use empty default
        results = ["", "", "", "", ""]
    else:
        file.close()
    results[task_id] = height_string
    stripped = [s.strip() for s in results]
    with open(csvFile, 'w') as file:
        file.writelines(csvSplitBy.join(stripped) + '\n')


def calculate_height(growth_rate, solution: deque, task_id: int):
    n = len(growth_rate)
    max_height = 0
    count = n * [1]
    sum_of_growth_rates = sum(growth_rate)
    
    for i in range(0, NUMBER_OF_ITERATIONS):
        to_cut = solution.popleft()
        for j in range(0, n):
            print("Count= " + str(count[j]), "growth rate =" + str(growth_rate[j]), "Multi = " +str(count[j] * growth_rate[j]))
            max_height = max(max_height, count[j] * growth_rate[j])
        for j in range(0, n):
            if j in to_cut:
                count[j] = 1
            else:
                count[j] += 1
        solution.append(to_cut)
    print("max_height=" + str(max_height)
          + " sum_of_growth_rates=" + str(sum_of_growth_rates))
    try:
        write_results(task_id, max_height, csvFile)
    except:
        print("Could not save result to " + csvFile)

