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


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ani = animation.FuncAnimation(fig, animate, interval=5000)
plt.show()
