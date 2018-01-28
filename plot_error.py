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

y1 = smooth(list(df[1]/100))
plt.plot(x[N-1:], y1[N-1:], c='b', linewidth=1)

y2 = smooth(list(df[3]/100))
plt.plot(x[N-1:], y2[N-1:], c='g', linewidth=1)

y3 = smooth(list(df[5]/100))
plt.plot(x[N-1:], y3[N-1:], c='r', linewidth=1)

# plt.grid()
plt.title('Error for Random, Violent, DQN')
plt.legend(['Random', 'Violent', 'DQN'], loc='upper right')
plt.xlabel('SampleID')
plt.ylabel('Error %')
plt.show()
