import argparse
import numpy as np
from collections import defaultdict

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description = 'collect the test result')
    parser.add_argument('--path', dest='path', type = str)
    args = parser.parse_args()
    
    time = defaultdict(list)
    success = defaultdict(list)
    recovery = defaultdict(list)

    with open(args.path, "r") as f:
        for l in f.readlines():
            l = l.split(" ")
            h = int(l[0])
            if int(l[-1]):
                time[h].append(float(l[2]))
                recovery[h].append(int(l[-2]))
            success[h].append(int(l[-1]))
    avg_time = []
    avg_success = []
    for k in sorted(time.keys()):
        print("habitat: %d, time: %.4f +- %.2f, success: %.4f, (%d/%d)" %(k, np.mean(time[k]), np.std(time[k]), np.sum(success[k]) / len(success[k]), len(time[k]), len(success[k])))
        avg_time.append(np.mean(time[k]))
        avg_success.append(np.sum(success[k]) / len(success[k]))

    time_mean = [np.mean([time[k][min(i, len(time[k])-1)] for k in time.keys()]) for i in range(20)]
    print("Average time: %.4f +- %.4f, average success: %.4f +- %.4f" %(np.mean(time_mean), np.std(time_mean), np.mean(avg_success), np.std(avg_success)))
