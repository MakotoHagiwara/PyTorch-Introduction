import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force = True)
import numpy as np
import gym
from time import sleep
import os
from Actor import actor_process
from Learner import learner_process

def run():
    mp.freeze_support()
    n_actors = 5
    path = os.path.join('./', 'memory.pt')
    model_path = os.path.join('./', 'model.pt')
    target_model_path = os.path.join('./', 'target_model.pt')
    try:
        os.remove(path)
        os.remove(model_path)
        os.remove(target_model_path)
    except:
        pass
    
    processes = [mp.Process(target = learner_process,
                                    args = (path, model_path, target_model_path))]
    for actor_id in range(n_actors):
        processes.append(mp.Process(target = actor_process,
                                    args = (path, model_path, target_model_path, actor_id)))
    for i in range(len(processes)):
        processes[i].start()
    for p in processes:
        p.join()
        
if __name__ == '__main__':
    run()