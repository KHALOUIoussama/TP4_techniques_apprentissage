import numpy as np
import sklearn.svm
import matplotlib.pyplot as plt

def test_sklearn_svm(X_train, y_train, X_test, y_test):
    model = sklearn.svm.LinearSVC()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accu = (pred == y_test).mean()
    return accu


def grad_check_loss(loss_func):
    """Do a numerical gradient check on 'loss_func'

    'loss_func' must have the following signature: func(W, X, y)
    and must return a tuple (output, gradient)"""

    h = 1e-5  # Step size

    # Init random W, X, y
    N, D, C = 3, 4, 3
    W = np.random.randn(D, C) * 0.01
    X = np.random.randn(N, D)
    y = np.random.randint(0, C, size=(N,))

    # Learn an epoch
    for xs, ys in zip(X, y):
        _, dW = loss_func(W, xs, ys)
        W -= 1e-3 * dW

    # Do grad check for each sample
    total_error = 0
    for xs, ys in zip(X, y):
        # Compute analytical gradient
        _, dW = loss_func(W, xs, ys)  # at center

        # Compute numerical gradient

        for i in range(W.size):  # For each parameter
            oldval = W.flat[i]

            W.flat[i] += h
            loss_right, _ = loss_func(W, xs, ys)

            W.flat[i] = oldval - h
            loss_left, _ = loss_func(W, xs, ys)

            W.flat[i] = oldval

            dw_num = (loss_right - loss_left) / (2.0 * h)
            dw_ana = dW.flat[i]

            err = abs(dw_num - dw_ana) / max(abs(dw_num), abs(dw_ana))
            total_error += err

    total_error /= len(X) * W.size

    return total_error


def grad_check_net(net, loss_func):
    h = 1e-5  # Step size
    lr = 1e-2

    # Init random W, X, y
    N = 6
    D, C = net.in_size, net.num_classes
    X = np.random.randn(N, D)
    y = np.random.randint(0, C, size=(N,))

    # Learn an epoch
    for _ in range(10):
        for xs, ys in zip(X, y):
            net.forward_backward(xs, ys)
            # Take gradient step
            for p, grad in zip(net.parameters, net.gradients):
                p -= lr * grad

    # Do grad check
    error_per_module = []
    for p, grad in zip(net.parameters, net.gradients):  # For each module
        total_err = 0
        for i in range(p.size):  # For each parameter
            for xs, ys in zip(X, y):  # For each sample
                oldval = p.flat[i]

                # Compute numerical gradient
                p.flat[i] += h
                loss_right = net.forward_backward(xs, ys)

                p.flat[i] = oldval - h
                loss_left = net.forward_backward(xs, ys)

                dw_num = (loss_right - loss_left) / (2.0 * h)
                p.flat[i] = oldval

                # Compute analytical gradient
                net.forward_backward(xs, ys)  # at center
                dw_ana = grad.flat[i]

                err = abs(dw_num - dw_ana) / (max(abs(dw_num), abs(dw_ana)) + 1e10)
                total_err += err / len(X)
        error_per_module.append(total_err / p.size)
        print(error_per_module[-1])

    return np.mean(error_per_module)


def plot_curves(loss_train_curve, loss_val_curve, accu_train_curve, accu_val_curve):
    xdata = np.arange(1, len(loss_train_curve) + 1)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.ylabel('Loss')
    plt.plot(xdata, loss_train_curve, label='training')
    plt.plot(xdata, loss_val_curve, label='validation')
    plt.xticks(xdata)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.ylabel('Accuracy')
    plt.plot(xdata, accu_train_curve, label='training')
    plt.plot(xdata, accu_val_curve, label='validation')
    plt.xticks(xdata)
    plt.legend()
    plt.show(block=False)


def bend_data(X, angle_delta, radius_min, radius_delta):
    assert X.shape[1] == 2

    margin = 0.0
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xy_min = np.array([x_min, y_min])
    xy_max = np.array([x_max, y_max])

    X_norm = (X - xy_min) / (xy_max - xy_min)
    theta = X_norm[:, 0] * angle_delta
    rho = radius_min + radius_delta * X_norm[:, 1]
    bent_x = np.cos(theta) * rho
    bent_y = np.sin(theta) * rho

    return np.stack([bent_x, bent_y], axis=1)


def make_grid():
    steps = 50
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    return np.c_[xx.ravel(), yy.ravel()]


def plot_kernels(model):
    vec_x, vec_y, pos_x = model.W[0], model.W[1], np.zeros(model.num_classes)
    if model.bias:
        pos_y = model.W[-1]
        kernels = model.W[:-1]
    else:
        pos_y = pos_x
        kernels = model.W
    plt.quiver(pos_x, pos_y, vec_x, vec_y)
    rot_mat = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = rot_mat.dot(kernels)  # (2, 3)
    pos_xy = np.stack([pos_x, pos_y], axis=0)
    start = pos_xy - rotated * 1.0
    end = pos_xy + rotated * 1.0  # (2, 3)
    x_start_end = np.stack([start[0], end[0]], axis=0)
    y_start_end = np.stack([start[1], end[1]], axis=0)
    for i, (x12, y12) in enumerate(zip(x_start_end.T, y_start_end.T)):
        plt.plot(x12, y12, label=str(i))