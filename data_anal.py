from collections import defaultdict
import numpy as np
def read_results(path):
    data = defaultdict(dict)

    first = None
    second = None
    values = []

    def save_current(current_first,current_second,current_values):
        if current_first is not None and current_second is not None:
            data[current_first][current_second] = current_values.copy()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line =='':
                continue
            if ":" in line:
                save_current(first,second,values)
                values=[]
                first = (int)(line.split(":")[0])
                second = (int)(line.split(":")[1])
            else:
                values.append(float(line))
    save_current(first,second,values)
    return data


if __name__ == "__main__":
    results = read_results("data.txt")

    for a in results.keys():
        for b in results[a].keys():
            print("|",a,"|",b ,"|",sum(results[a][b])/len(results[a][b]),"|",np.std(results[a][b]),"|")
