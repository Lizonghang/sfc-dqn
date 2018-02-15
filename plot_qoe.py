import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

smooth_rate = 0.1
N = int(1/smooth_rate)


def smooth(sequence):
    y = sequence[:N-1]
    for i in xrange(len(sequence)-N+1):
        y.append(np.mean(sequence[i: i+N]))
    return y

df = pd.read_csv('output.txt', header=None)

x = range(df.shape[0])

y1 = smooth(list(df[0]))
plt.plot(x, y1, c='b', linewidth=1)

y2 = smooth(list(df[2]))
plt.plot(x, y2, c='g', linewidth=1)

y3 = smooth(list(df[4]))
plt.plot(x, y3, c='r', linewidth=1)

y4 = smooth(list(df[6]))
plt.plot(x, y4, c='k', linewidth=1)

plt.grid()
plt.title('QoE for Random, Violent, DQN, ACO')
plt.legend(['Random', 'Violent', 'DQN', 'ACO'], loc='upper right')
plt.xlabel('Step')
plt.ylabel('QoE')
plt.show()
