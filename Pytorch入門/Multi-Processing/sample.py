import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import numpy as np
import gym
from time import sleep
import os


def actor_process(path):
    # 環境，メモリを定義
    env = gym.make('CartPole-v0')
    obs = np.zeros((100, 4), dtype=np.float)
    env.reset()

    # メモリに状態を保存
    for i in range(100):
        ob, _, done, _ = env.step(env.action_space.sample())
        if done:
            ob = env.reset()
        obs[i] = ob
    sleep(np.random.random()*5)
    
    # 既にファイルが存在するなら、追加して保存．無いなら新しく保存．
    while True:
        try:
            if os.path.isfile(path):
                # メモリを読み込み
                memory = torch.load(path)
                # メモリファイルを削除
                os.remove(path)
                # メモリに追加
                memory['obs'] = np.vstack(memory['obs'], obs)
                # メモリを保存
                torch.save(memory, path)
                break
            else:
                memory = dict()
                memory['obs'] = obs
                torch.save(memory, path)
                break
        except:
            # 他のプロセスがファイルを開いてたら、タイミングをずらしてまた開く
            sleep(np.random.random()*2+2)


def learner_process(path, n_actors):
    learner_memory = dict()
    learner_memory['obs'] = np.zeros((100 * n_actors, 4), dtype=np.float)
    idx = 0
    
    while True:
        if os.path.isfile(path):
            try:
                # メモリを読み込み
                memory = torch.load(path)
                # メモリファイルを削除
                os.remove(path)
                # Learnerのメモリに追加
                for i in range(idx, memory['obs'].shape[0]):
                    learner_memory['obs'][i] = memory['obs'][i]
                idx += memory['obs'].shape[0]

                # すべてのActorのデータを読み込んだら終了
                print('memory_index:', idx)
                if idx == 100 * n_actors:
                    return
            except:
                # 他のプロセスがファイルを開いてたら、タイミングをずらしてまた開く
                sleep(np.random.random()*2+2)


def run():
    mp.freeze_support()
    
    n_actors = 8
    path = os.path.join('./', 'memory.pt')
    try:
        os.remove(path)
    except:
        pass

    # Learner用のプロセスを追加（targetはプロセス（関数），argsはプロセスの引数）
    processes = [mp.Process(target=learner_process, args=(path, n_actors))]

    # Actor用のプロセスを追加
    for actor_id in range(n_actors):
        processes.append(mp.Process(target=actor_process, args=(path,)))

    # すべてのプロセスを開始
    for i in range(len(processes)):
        processes[i].start()

    # すべてのプロセスの終了を確認
    for p in processes:
        p.join()


if __name__ == '__main__':
    run()