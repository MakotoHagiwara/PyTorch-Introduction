import torch
import torch.multiprocessing as mp
#プロセスを開始する方式の指定
#spawnは新たにpythonインタープリタープロセスを開始する
#子プロセスはプロセスオブジェクトのrun()メソッド実行に
#必要なリソースのみを継承する
mp.set_start_method('spawn', force = True)
import numpy as np
import gym
from time import sleep
import os

def actor_process(path):
    env = gym.make('CartPole-v0')
    obs = np.zeros((100, 4), dtype = np.float)
    env.reset()
    
    #各ステップの状態を保存する
    for i in range(100):
        ob, _, done, _ = env.step(env.action_space.sample())
        if done:
            ob = env.reset()
        obs[i] = ob
    sleep(np.random.random() * 5)
    
    while True:
        try:
            if os.path.isfile(path):
                #メモリを読み込む
                memory = torch.load(path)
                #メモリファイルの削除
                os.remove(path)
                #メモリに追加
                #vstackは一番深い層の要素同士を結合する(http://ailaby.com/vstack_hstack/)
                #vstack = concatenate(axis = 0)
                #hstack = concatenate(axis = 1)
                memory['obs'] = np.vstack(memory['osb'], obs)
                #メモリを保存
                torch.save(memory, path)
                break
            else:
                memory = dict()
                memory['obs'] = obs
                torch.save(memory, path)
                break
        except:
            #他のプロセスがファイルを開いている場合は、タイミングをずらして開く
            sleep(np.random.random() * 2 + 2)
            
def learner_process(path, n_actors):
    learner_memory = dict()
    learner_memory['obs'] = np.zeros((100 * n_actors, 4), dtype = np.float)
    idx = 0
    
    while True:
        if os.path.isfile(path):
            try:
                memory = torch.load(path)
                os.remove(path)
                for i in range(idx, memory['obs'].shape[0]):
                    learner_memory['obs'][i] = memory['obs'][i]
                idx += memory['obs'].shape[0]
                
                #全てのActorのデータを読み込んだら終了
                print('memory_index:', idx)
                if idx == 100 * n_actors:
                    return
            except:
                sleep(np.random.random() * 2 + 2)
                
def run():
    mp.freeze_support()
    
    n_actors = 8
    path = os.path.join('./', 'memory.pt')
    try:
        os.remove(path)
    except:
        pass
    
    #Learner用のプロセスを追加(targetはプロセス(関数)、argsはプロセスの引数)
    processes = [mp.Process(target = learner_process, 
                            args = (path, n_actors))]
    #Actor用のプロセスを追加
    for actor_id in range(n_actors):
        processes.append(mp.Process(target = actor_process, 
                                    args = (path, )))
        
    for i in range(len(processes)):
        processes[i].start()
        
    for p in processes:
        p.join()
        
if __name__ == '__main__':
    run()