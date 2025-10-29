import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Load MNIST using TensorFlow
# ----------------------------
def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    print(f"Loaded MNIST: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ----------------------------
# 2️⃣ CNN Layers from Scratch
# ----------------------------
class Conv3x3:
    # 3x3 Convolution Layer
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                region = image[i:(i + 3), j:(j + 3)]
                yield region, i, j

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((self.num_filters, h - 2, w - 2))
        for f in range(self.num_filters):
            for region, i, j in self.iterate_regions(input):
                output[f, i, j] = np.sum(region * self.filters[f])
        return output

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for f in range(self.num_filters):
            for region, i, j in self.iterate_regions(self.last_input):
                d_L_d_filters[f] += d_L_d_out[f, i, j] * region
        # Update filters
        self.filters -= learn_rate * d_L_d_filters
        return None


class MaxPool2:
    # 2x2 Max Pooling
    def iterate_regions(self, image):
        h, w = image.shape
        new_h, new_w = h // 2, w // 2
        for i in range(new_h):
            for j in range(new_w):
                region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield region, i, j

    def forward(self, input):
        self.last_input = input
        d, h, w = input.shape
        output = np.zeros((d, h // 2, w // 2))
        for i in range(d):
            for region, j, k in self.iterate_regions(input[i]):
                output[i, j, k] = np.max(region)
        return output

    def backprop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        for i in range(self.last_input.shape[0]):
            for region, j, k in self.iterate_regions(self.last_input[i]):
                h, w = region.shape
                max_val = np.max(region)
                for m in range(h):
                    for n in range(w):
                        if region[m, n] == max_val:
                            d_L_d_input[i, j * 2 + m, k * 2 + n] = d_L_d_out[i, j, k]
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
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            d_L_d_t = grad * d_out_d_t
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights and biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
        return d_L_d_inputs.reshape(self.last_input_shape)


# ----------------------------
# 3️⃣ Training Utilities
# ----------------------------
def forward(image, label, conv, pool, softmax):
    out = conv.forward(image)
    out = np.maximum(out, 0)  # ReLU
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    return out, loss, acc


def train_cnn(X_train, y_train, conv, pool, softmax, epochs=1, learn_rate=0.005, limit=1000):
    print("Training CNN...")
    loss_history = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        total_loss = 0
        num_correct = 0

        for i, (img, label) in enumerate(zip(X_train[:limit], y_train[:limit])):
            out, loss, acc = forward(img, label, conv, pool, softmax)
            total_loss += loss
            num_correct += acc

            # Gradient for softmax layer
            gradient = np.zeros(10)
            gradient[label] = -1 / out[label]

            grad_back = softmax.backprop(gradient, learn_rate)
            grad_back = pool.backprop(grad_back)
            grad_back = grad_back * (grad_back > 0)  # ReLU derivative
            conv.backprop(grad_back, learn_rate)

            if (i + 1) % 100 == 0:
                print(f"[Step {i + 1}] Avg Loss={total_loss / 100:.3f} | Accuracy={num_correct}%")
                loss_history.append(total_loss / 100)
                total_loss = 0
                num_correct = 0

    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Steps (x100 images)")
    plt.ylabel("Loss")
    plt.show()


# ----------------------------
# 4️⃣ Run it
# ----------------------------
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()

    conv = Conv3x3(8)             # 8 filters
    pool = MaxPool2()             # 2x2 pooling
    softmax = Softmax(13 * 13 * 8, 10)  # fully connected output

    train_cnn(X_train, y_train, conv, pool, softmax, epochs=1, learn_rate=0.005, limit=1000)
