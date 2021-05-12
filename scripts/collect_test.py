from os.path import exists, join
import json
import os
import numpy as np
import time
import pickle
import sys
import yaml
import logging
import argparse

def main(buffer_path):
    f = open(join(buffer_path, 'config.yaml'), 'r')
    config = yaml.load(f, Loader=yaml.FullLoader)
    num_trials = config["condor_config"]["num_trials"]
    ep_lengths = []
    flag = True

    save_path = join(buffer_path, "test_result.txt")
    outf =  open(save_path, "w")
    worlds = []
    bad_worlds = []
    for dirname, dirnames, filenames in os.walk(buffer_path):
        for filename in filenames:
            p = join(dirname, filename)
            if p.endswith('.pickle'):
                try:
                    with open(p, 'rb') as f:
                        traj = pickle.load(f)
                    world = traj[-1][-1]['world']
                    if isinstance(world, str):
                        world = int(world.split("_")[-1].split(".")[0])
                    ep_return = sum([t[2] for t in traj])
                    ep_length = len(traj)
                    success = int(traj[-1][-1]['success'])
                    
                    if len(filenames) == num_trials and world not in worlds:
                        outf.write("%d %d %f %d\n" %(world, ep_length, ep_return, success))
                        ep_lengths.append(float(ep_length))
                    else:
                        break
                except:
                    logging.exception("")
                    pass

        if dirname.split("/")[-1].startswith("actor"):
            if len(filenames) == num_trials:
                worlds.append(world)
            elif world not in bad_worlds:
                bad_worlds.append(world)
            else:
                print("world %s fail for all two test!" %(world))
                flag = False

    outf.close()
    if flag:
        print("Test finished!")
        print("Find the test result under %s" %(save_path))
        print("Quick report: avg time %.2f" %(sum(ep_lengths)/len(ep_lengths)))
    else:
        print("Some tests are still running")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description = 'collect the test result')
    parser.add_argument('--buffer_path', dest='buffer_path', type = str)
    buffer_path = parser.parse_args().buffer_path
    main(buffer_path)