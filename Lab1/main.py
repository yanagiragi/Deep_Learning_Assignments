import matplotlib as mpl
mpl.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# examples calls: show_data([x1, x2], [y1, y2], [title1, title2])
def show_data(xs, ys, ts):
    # colors = (r=1,g=0,b=0) ~ (r=0,g=0,b=1)
    # N is the number of rgb quantization levels
    cm = LinearSegmentedColormap.from_list('mymap', [(1,0,0), (0,0,1)], N=2)

    n = len(xs)
    plt.figure(figsize=(5*n, 5))
    
    print(xs[0])
    print(xs[0][0,:])
    print(xs[0][:,0])

    print(ys[0].shape)

    for i, x, y, t in zip(range(n), xs, ys, ts):
        y = np.round(y)
        plt.subplot(1, n, i+1)
        plt.title(t, fontsize=18)
        plt.scatter(x[:,0], x[:,1], c=y[:,0], cmap=cm)

    plt.show()

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_xor_easy(n=11):
    inputs = []
    labels = []
    step = 1 / (n - 1)
    for i in range(n):
        inputs.append([step * i, step * i])
        labels.append(0)

        # skip middle point
        if i == int((n-1)/2):
            continue

        inputs.append([step * i, 1 - step * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n * 2 - 1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x1, y1 = generate_linear(n=10)
x2, y2 = generate_xor_easy()
show_data([x1, x2], [y1, y2], ['linear', 'xor_easy'])