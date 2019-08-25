import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force = True)
import numpy as np
import gym
from time import sleep
import os
from Actor_Process import actor_process
from Learner import learner_process

def run():
    mp.freeze_support()
    n_actors = 5
    path = os.path.join('./', 'memory.pt')
    model_path = os.path.join('./', 'model.pt')
    try:
        os.remove(path)
    except:
        pass
    
    processes = [mp.Process(target = learner_process,
                                    args = (path, model_path))]
    for actor_id in range(n_actors):
        processes.append(mp.Process(target = actor_process,
                                    args = (path, model_path)))
    for i in range(len(processes)):
        processes[i].start()
    for p in processes:
        p.join()
        
if __name__ == '__main__':
    run()