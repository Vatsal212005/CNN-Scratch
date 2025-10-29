import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_mnist():
    from urllib import request
    import gzip
    import shutil
    import os

    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    os.makedirs("mnist_data", exist_ok=True)
    for key, fname in files.items():
        path = os.path.join("mnist_data", fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            request.urlretrieve(base_url + fname, path)

    def read_images(path):
        with gzip.open(path, 'rb') as f:
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 28, 28) / 255.0

    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            f.read(8)
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    X_train = read_images("mnist_data/" + files["train_images"])
    y_train = read_labels("mnist_data/" + files["train_labels"])
    X_test = read_images("mnist_data/" + files["test_images"])
    y_test = read_labels("mnist_data/" + files["test_labels"])
    return X_train, y_train, X_test, y_test

def one_hot(y, num_classes=10):
    oh = np.zeros((y.size, num_classes))
    oh[np.arange(y.size), y] = 1
    return oh

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                region = image[i:(i+3), j:(j+3)]
                yield region, i, j

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((self.num_filters, h-2, w-2))

        for f in range(self.num_filters):
            for region, i, j in self.iterate_regions(input):
                output[f, i, j] = np.sum(region * self.filters[f])
        return output

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)

        for f in range(self.num_filters):
            for region, i, j in self.iterate_regions(self.last_input):
                d_L_d_filters[f] += d_L_d_out[f, i, j] * region

        self.filters -= learn_rate * d_L_d_filters
        return None


class MaxPool2:
    def iterate_regions(self, image):
        h, w = image.shape
        new_h, new_w = h // 2, w // 2
        for i in range(new_h):
            for j in range(new_w):
                region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield region, i, j

    def forward(self, input):
        self.last_input = input
        d, h, w = input.shape
        output = np.zeros((d, h//2, w//2))
        for i in range(d):
            for region, j, k in self.iterate_regions(input[i]):
                output[i, j, k] = np.max(region)
        return output

    def backprop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        d, h, w = self.last_input.shape
        for i in range(d):
            for region, j, k in self.iterate_regions(self.last_input[i]):
                h_start, w_start = j*2, k*2
                h_end, w_end = h_start+2, w_start+2
                amax = np.max(region)
                for m in range(2):
                    for n in range(2):
                        if region[m, n] == amax:
                            d_L_d_input[i, h_start+m, w_start+n] = d_L_d_out[i, j, k]
        return d_L_d_input


class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals - np.max(totals))
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        for i, grad in enumerate(d_L_d_out):
            if grad == 0:
                continue

            t_exp = np.exp(self.last_totals - np.max(self.last_totals))
            S = np.sum(t_exp)
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            d_L_d_t = grad * d_out_d_t
            d_L_d_w = d_t_d_w[:, np.newaxis] @ d_L_d_t[np.newaxis, :]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)

def forward(image, label):
    out = conv.forward(image)
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    return out, loss, acc


def train(im, label, lr=0.005):
    out, loss, acc = forward(im, label)

    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    grad_back = softmax.backprop(gradient, lr)
    grad_back = pool.backprop(grad_back)
    conv.backprop(grad_back, lr)
    return loss, acc


conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(13 * 13 * 8, 10)

X_train, y_train, X_test, y_test = load_mnist()
X_train, y_train = X_train[:1000], y_train[:1000]

print("Training CNN...")
losses, accuracies = [], []
for i, (im, label) in enumerate(zip(X_train, y_train)):
    l, acc = train(im, label)
    losses.append(l)
    accuracies.append(acc)
    if (i+1) % 100 == 0:
        print(f"[Step {i+1}] Avg Loss={np.mean(losses[-100:]):.3f} | Accuracy={np.mean(accuracies[-100:])*100:.1f}%")

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()
