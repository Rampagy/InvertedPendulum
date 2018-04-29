import sys
sys.path.append("../Shared")

import train_model as tm
import NeuralNetwork as nn
import gym_custominvertedpendulum as gym
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate(i):
    try:
        graph_data = open('../Model/scores.txt', 'r').read()
        lines = graph_data.split('\n')
        xs = []
        ys = []
        count = 0

        for line in lines:
            if len(line) > 1:
                y = float(line)
                xs.append(count)
                ys.append(y)
                count += 1

        ax1.clear()
        ax1.plot(xs, ys)
        plt.xlabel('Evaluation episode, 1 evaluation every 100 training episodes')
        plt.ylabel('Evaluation score')
        plt.title('Evaluation Timeseries')
    except Exception as e:
        print(e)

def live_plot():
    ani = animation.FuncAnimation(fig, animate, interval=5000)
    plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

# Setup a list of processes that we want to run
processes = [mp.Process(target=live_plot, args=()) for x in range(1)]

# start plot process
for p in processes:
    p.start()

#create env to initialize model
env = gym.make('CustomInvertedPendulum-v0')

# create model
model = nn.Control_Model(len(env.observation_space.low))

train_count, eval_score = tm.train_model(env=env, eval_episodes=25,
    trained_threshold=400, model=model, render_episodes=0)

# end plot process
for p in processes:
    p.join()
