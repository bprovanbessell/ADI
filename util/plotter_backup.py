import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import manifold
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, roc_curve

def model_mse(model, X):
    return 1

def plot_latent_space(model, X, labels, index_x=0, index_y=1, name=""):
    # display a 2D plot of the digit classes in the latent space
    z = model.encode(X)
    plt.figure(figsize=(12, 10))
    if labels is None:
        plt.scatter(z[:, index_x], z[:, index_y])
    else:
        labels = ['r' if x else 'g' for x in labels]
        plt.scatter(z[:, index_x], z[:, index_y], c=labels)
    plt.colorbar()
    plt.xlabel("z[" + str(index_x) + "]")
    plt.ylabel("z[" + str(index_y) + "]")
    plt.title('Latent space plotting ' + name)

    plt.margins(0)
    plt.show()

def plot_latent_space_grid_4(model, X, labels, xs, ys, name):
    # display a 2D plot of the digit classes in the latent space
    plt.figure(figsize=(24, 20))
    labels = ['r' if x else 'g' for x in labels]

    z = model.encode(X)
    print(z.shape)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Latent space plotting ' + name)
    ax1.scatter(z[:, xs[0]], z[:, ys[0]], c=labels)
    ax2.scatter(z[:, xs[1]], z[:, ys[1]], c=labels)
    ax3.scatter(z[:, xs[2]], z[:, ys[2]], c=labels)
    ax4.scatter(z[:, xs[3]], z[:, ys[3]], c=labels)

    for ax in fig.get_axes():
        ax.label_outer()
    plt.show()

def plot_latent_space_all(model, X_train, X_test, X_anomaly, name):
    z_train = model.encode(X_train)
    z_test = model.encode(X_test)
    z_anomaly = model.encode(X_anomaly)
    latent_dim = z_train.shape[-1]

    fig, m_axs = plt.subplots(latent_dim, latent_dim, figsize=(latent_dim * 5, latent_dim * 5))
    if latent_dim == 1:
        m_axs = [[m_axs]]
    for i, n_axs in enumerate(m_axs, 0):
        for j, c_ax in enumerate(n_axs, 0):
            c_ax.scatter(np.concatenate([z_train[:, i], z_test[:, i], z_anomaly[:, i]], 0),
                         np.concatenate([z_train[:, j], z_test[:, j], z_anomaly[:, j]], 0),
                         c=(['b'] * z_train.shape[0]) + ['g'] * z_test.shape[0] + ['r'] * z_anomaly.shape[0], alpha=0.5)


    for ax in fig.get_axes():
        ax.label_outer()
    plt.show()

def plot_tsne_anomaly(model, X, X_anomaly):
    # display a 2D plot of the digit classes in the latent space
    print(X.shape)
    print(X_anomaly.shape)
    latent_space_tsne = manifold.TSNE(2, verbose = True, n_iter = 2000)
    z = model.encode(X)
    z_anomaly = model.encode(X_anomaly)
    xa_tsne = latent_space_tsne.fit_transform(np.concatenate([z[:, :], z_anomaly[:,:]],0))
    plt.figure(figsize=(6, 6))
    plt.scatter(xa_tsne[:, 0], xa_tsne[:, 1],
                c=(['g'] * z.shape[0]) + ['r'] * z_anomaly.shape[0], alpha=0.5)
    plt.show()

def plot_tsne(model, X_train, X_test, X_anomaly):
    # display a 2D plot of the digit classes in the latent space
    latent_space_tsne = manifold.TSNE(2, verbose = True, n_iter = 2000)
    z_train = model.encode(X_train)
    z_test = model.encode(X_test)
    z_anomaly = model.encode(X_anomaly)
    xa_tsne = latent_space_tsne.fit_transform(np.concatenate([z_train, z_test, z_anomaly ],0))
    plt.figure(figsize=(6, 6))
    plt.scatter(xa_tsne[:, 0], xa_tsne[:, 1],
                c=(['b'] * z_train.shape[0]) + ['g'] * z_test.shape[0] + ['r'] * z_anomaly.shape[0], alpha=0.5)
    plt.show()

def plot_reconstruction_error(model, X_train, X_test, X_anomaly):
    model_mse = lambda x: np.mean(tf.keras.losses.mean_squared_error(x, model.predict(x)), axis=1)
    model_mse = lambda x: tf.keras.losses.mean_squared_error(x, model.predict(x))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    ax1.hist(model_mse(X_train), bins=50, density=True, label='Training Sample', alpha=1.0)
    ax1.hist(model_mse(X_test), bins=50, density=True, label='Testing Sample', alpha=0.5)
    ax1.hist(model_mse(X_anomaly), bins=50, density=True, label='Anomaly Sample', alpha=0.5)
    ax1.legend()
    ax1.set_xlabel('Reconstruction Error');

    plt.show()

def plot_pdf(model, X_train, X_test, X_anomaly):
    kd = KernelDensity()
    kd.fit(model.encode(X_train))
    test_score = [kd.score(x.reshape(1, -1)) for x in model.encode(X_test)]
    anom_score = [kd.score(x.reshape(1, -1)) for x in model.encode(X_anomaly)]
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    ax1.hist(test_score, label='Test Digits', density=True, alpha=1.0)
    ax1.hist(anom_score, label='Anomaly Digits', density=True, alpha=0.5)
    ax1.legend()
    print('Test data score', np.mean(test_score))
    print('Anomaly score', np.mean(anom_score))
    plt.show()


def plot_roc(model, X_test, X_anomaly, name):
    model_mse = lambda x: tf.keras.losses.mean_absolute_error(x, model.predict(x))
    model_mse = lambda x: np.mean(tf.keras.losses.mean_squared_error(x, model.predict(x)), axis=1)

    mse_score = np.concatenate([model_mse(X_test), model_mse(X_anomaly)], 0)
    print(mse_score)
    print(mse_score.shape)
    true_label = [0] * X_test.shape[0] + [1] * X_anomaly.shape[0]
    print(len(true_label))
    if roc_auc_score(true_label, mse_score) < 0.5:
        mse_score *= -1
    fpr, tpr, thresholds = roc_curve(true_label, mse_score)
    auc_score = roc_auc_score(true_label, mse_score)
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    ax1.plot(fpr, tpr, 'b.-', label='ROC Curve (%2.2f)' % auc_score)
    ax1.plot(fpr, fpr, 'k-', label='Random Guessing')
    ax1.legend()
    plt.title("ROC Curve " + name)

    plt.show()

def plot_reconstruction_error_time(model, X_train, X_test, X_anomaly, meta_train, meta_test, meta_anomaly, name):
    # display a 2D plot of the digit classes in the latent space

    plt.scatter([datetime.fromtimestamp(x).time().strftime("%H:%M") for x in self.meta_test[:, 2]],
                self.model_mse(self.X_test), c='g')
    plt.scatter([datetime.fromtimestamp(x).time().strftime("%H:%M") for x in self.meta_anomaly[:, 2]],
                self.model_mse(self.X_anomaly), c='r')

    model_mse = lambda x: np.mean(tf.keras.losses.mean_squared_error(x, model.predict(x)), axis=1)
    plt.scatter(meta_train[:, 2], model_mse(X_train), c='b')
    plt.scatter(meta_test[:, 2], model_mse(X_test), c='g')
    plt.scatter(meta_anomaly[:, 2], model_mse(X_anomaly), c='r')
    # plt.ylim(0, 1.5)
    plt.title("Reconstruction error over time " + name)
    plt.xlabel("Sample timestamp")
    plt.ylabel("Reconstruction error")
    plt.figure(figsize=(6, 6))
    plt.show()



def reconstruction_comparison(solvers, X, limit, input_shape):
    err_solvers = np.zeros((len(solvers)))
    count = 0
    for x in X:
        print("\rClassify ", count + 1, " \tout of ", X.shape[0], end="")
        count += 1
        x = x.reshape(input_shape)
        for i, s in enumerate(solvers):
            err_solvers[i] += np.mean(tf.keras.losses.mean_absolute_error(x, s.predict(x)))
        if count > limit:
            break

    err_solvers /= count

    for i, s in enumerate(solvers):
        print("\nAverage compressed error of ", s.name, ": \t", np.mean(err_solvers[i]))

    return err_solvers


def reconstruction_comparison_anomaly(solvers, X, limit, input_shape, anomaly_vector):
    error = np.zeros((len(solvers)))
    error_anomaly = np.zeros((len(solvers)))
    count = 0
    anomaly_count = 0
    print(anomaly_vector)
    for j, x in enumerate(X):
        print("\rClassify ", j + 1, " \tout of ", X.shape[0], end="")
        x = x.reshape(input_shape)
        if anomaly_vector[j]:
            anomaly_count += 1
            for i, s in enumerate(solvers):
                error_anomaly[i] += np.mean(tf.keras.losses.mean_absolute_error(x, s.predict(x)))
        else:
            count += 1
            for i, s in enumerate(solvers):
                error[i] += np.mean(tf.keras.losses.mean_absolute_error(x, s.predict(x)))
        if j >= limit:
            break
    error_anomaly /= anomaly_count
    error /= count
    print(anomaly_count)
    print(count)

    for i, s in enumerate(solvers):
        print("\nAverage error of normal samples", s.name, ": \t", np.mean(error[i]))
        print("Average error of anomaly samples", s.name, ": \t", np.mean(error_anomaly[i]))

    return error



