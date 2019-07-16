# For Ubuntu 16.04
import matplotlib as mpl
mpl.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def show_result(x, y, pred_y):
    pred_y = np.round(pred_y)
    cm = LinearSegmentedColormap.from_list(
        'mymap', [(1, 0, 0), (0, 0, 1)], N=2)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    plt.scatter(x[:,0], x[:,1], c=y[:,0], cmap=cm)
    
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    plt.scatter(x[:,0], x[:,1], c=pred_y[:,0], cmap=cm)
    
    plt.show()

# examples calls: show_data([x1, x2], [y1, y2], [title1, title2])
def show_data(xs, ys, ts):
    # colors = (r=1,g=0,b=0) ~ (r=0,g=0,b=1)
    # N is the number of rgb quantization levels
    cm = LinearSegmentedColormap.from_list('mymap', [(1,0,0), (0,0,1)], N=2)

    n = len(xs)
    plt.figure(figsize=(5*n, 5))
    
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

# Formula: https://github.com/csielee/2019DL/blob/master/lab1/DLP_LAB1_0756110_%E6%9D%8E%E6%9D%B1%E9%9C%96_report.pdf
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1 - x)

# Mean Square Error
def loss(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def derivative_loss(y, y_hat):
    return (y - y_hat) * (2 / y.shape[0])

class layer:
    def __init__(self, input_size, output_size):
        self.w = np.random.normal(0, 1, (input_size + 1, output_size))


    def forward(self, x):
        # append 1 at the end of every col in x, since we store bias in last col of w
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        self.forward_gradient = x
        self.y = sigmoid(np.matmul(x, self.w))
        return self.y

    def backward(self, derivative_C):
        self.backward_gradient = np.multiply(derivative_sigmoid(self.y), derivative_C)
        return np.matmul(self.backward_gradient, self.w[:-1].T)
    
    def update(self, learning_rate):
        self.gradient = np.matmul(self.forward_gradient.T, self.backward_gradient)
        self.w -= learning_rate * self.gradient
        return self.gradient

class nn:
    def __init__(self, sizes, learning_rate = 0.1):
        self.learning_rate = learning_rate

        # size2 = output_size of each layer
        # size[1:] for get rid of size[0:] (input of network)
        # [0] for output of network
        sizes2 = sizes[1:] + [0]
        self.l = []
        
        for a,b in zip(sizes, sizes2):
            if (a + 1) * b == 0:
                continue
            self.l += [layer(a, b)]
        
    def forward(self, x):
        _x = x
        for l in self.l:
            _x = l.forward(_x)
        return _x
    
    def backward(self, dC):
        _dC = dC
        for l in self.l[::-1]:
            _dC = l.backward(_dC)

    def update(self):
        gradients = []
        for l in self.l:
            gradients += [l.update(self.learning_rate)]
        return gradients


x_linear, y_linear = generate_linear(n=100)
x_xor, y_xor = generate_xor_easy()

nn_linear = nn([2, 4, 4, 1], 1)
nn_xor = nn([2, 4, 4, 1], 1)
epoch_count = 10000
loss_threshold = 0.005
linear_stop = False
xor_stop = False

for i in range(epoch_count):
    if not linear_stop:
        y = nn_linear.forward(x_linear)
        loss_linear = loss(y, y_linear)
        nn_linear.backward(derivative_loss(y, y_linear))
        nn_linear.update()

        if loss_linear < loss_threshold:
            print ('Covergence: Linear')
            linear_stop = True

    if not xor_stop:
        y = nn_xor.forward(x_xor)
        loss_xor = loss(y, y_xor)
        nn_xor.backward(derivative_loss(y, y_xor))
        nn_xor.update()

        if loss_xor < loss_threshold:
            print ('Covergence: XOR')
            xor_stop = True

    if i % 200 == 0 or (linear_stop and xor_stop):
        print ('[{:4d}] linear loss : {:4f} \t XOR loss: {:.4f}'.format(i, loss_linear, loss_xor))

    if linear_stop and xor_stop:
        break

# print show input
show_data([x_linear, x_xor], [y_linear, y_xor], ['linear', 'xor_easy'])

y1 = nn_linear.forward(x_linear)
show_result(x_linear, y_linear, y1)

print ('linear loss ', loss(y1, y_linear))
print ('linear loss accuracy: ${:3.2f}%'.format(np.count_nonzero(np.round(y1) == y_linear) * 100 / len(y1)))

y2 = nn_xor.forward(x_xor)
show_result(x_xor, y_xor, y2)
print ('xor loss ', loss(y2, y_xor))
print ('xor loss accuracy: ${:3.2f}%'.format(np.count_nonzero(np.round(y2) == y_xor) * 100 / len(y2)))