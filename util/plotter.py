import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import manifold
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, roc_curve
from datetime import datetime
from influx import results_to_influx

class Plotter:
    def __init__(self, show_plot=False):
        self.name = "Experiment"
        self.model = None
        self.encoder = None
        self.decoder = None
        self.X_train = None
        self.X_test = None
        self.X_anomaly = None
        self.X_after = None
        self.meta_train = None
        self.meta_test = None
        self.meta_anomaly = None
        self.meta_after = None
        self.best_val_loss = np.Inf
        self.input_shape = None
        self.latent_dim = 2
        self.image_folder = "img/"
        self.show_plot = show_plot


    # def plot_latent_space(self, index_x=0, index_y=1):
    #     # display a 2D plot of the digit classes in the latent space
    #     z = self.model.encode(X)
    #     plt.figure(figsize=(12, 10))
    #     if labels is None:
    #         plt.scatter(z[:, index_x], z[:, index_y])
    #     else:
    #         labels = ['r' if x else 'g' for x in labels]
    #         plt.scatter(z[:, index_x], z[:, index_y], c=labels)
    #     plt.colorbar()
    #     plt.xlabel("z[" + str(index_x) + "]")
    #     plt.ylabel("z[" + str(index_y) + "]")
    #     plt.title('Latent space plotting ' + name)
    #
    #     plt.margins(0)
    #     plt.show()
    def model_mse(self, x):
        return np.mean(tf.keras.losses.mean_squared_error(x, self.model.predict(x)), axis=1)

    def latent_space_complete(self, anomaly=False):
        print(self.X_train.shape)
        z_train = self.model.encode(self.X_train)
        z_test = self.model.encode(self.X_test)
        if anomaly:
            z_anomaly = self.model.encode(self.X_anomaly)
        latent_dim = z_train.shape[-1]
        if latent_dim > 4:
            latent_dim = 4

        fig, m_axs = plt.subplots(latent_dim, latent_dim, figsize=(latent_dim * 5, latent_dim * 5))
        if latent_dim == 1:
            m_axs = [[m_axs]]
        for i, n_axs in enumerate(m_axs, 0):
            for j, c_ax in enumerate(n_axs, 0):
                if anomaly:
                    c_ax.scatter(np.concatenate([z_train[:, i], z_test[:, i], z_anomaly[:, i]], 0),
                             np.concatenate([z_train[:, j], z_test[:, j], z_anomaly[:, j]], 0),
                             c=(['b'] * z_train.shape[0]) + ['g'] * z_test.shape[0] + ['r'] * z_anomaly.shape[0],
                             alpha=0.5)
                else:
                    c_ax.scatter(np.concatenate([z_train[:, i], z_test[:, i]], 0),
                             np.concatenate([z_train[:, j], z_test[:, j]], 0),
                             c=(['b'] * z_train.shape[0]) + ['g'] * z_test.shape[0] ,
                             alpha=0.5)
        f = self.image_folder + self.name
        if anomaly:
            f += "_anomaly"

        for ax in fig.get_axes():
            ax.label_outer()
        fig.suptitle("Projection in the latent space " + self.name, fontsize=20)
        plt.savefig(f + "_latent_space.png", transparent=True, bbox_inches='tight', dpi=300)
        if self.show_plot:
            plt.show()

        plt.close(fig)

    def plot_tsne(self, anomaly=True, train=True, after_anomaly=True):
        # display a 2D plot of the digit classes in the latent space
        latent_space_tsne = manifold.TSNE(2, verbose=True, n_iter=2000)
        z = self.model.encode(self.X_test)
        color = ['g'] * z.shape[0]
        z_test = self.model.encode(self.X_test)
        plt.figure(figsize=(6, 6))
        if anomaly:
            z_anomaly = self.model.encode(self.X_anomaly)
            z = np.concatenate([z, z_anomaly], 0)
            color = np.concatenate([color, ['r'] * z_anomaly.shape[0]], 0)
        if train:
            z_train = self.model.encode(self.X_train)
            z = np.concatenate([z, z_train], 0)
            color = np.concatenate([color, ['b'] * z_train.shape[0]], 0)
        if after_anomaly:
            z_after_anom = self.model.encode(self.X_after)
            z = np.concatenate([z, z_after_anom], 0)
            color = np.concatenate([color, ['g'] * z_after_anom.shape[0]], 0)


        xa_tsne = latent_space_tsne.fit_transform(z)
        plt.scatter(xa_tsne[:, 0], xa_tsne[:, 1],
                    c=color, alpha=0.5)
        plt.title("t-SNE Representation of latent space " + self.name)
        f = self.image_folder + self.name
        if anomaly:
            f += "_anomaly"
        if train:
            f += "_train"
        if after_anomaly:
            f += "_after"
        plt.savefig(f + "_tsne.png", transparent=True, bbox_inches='tight', dpi=300)

        if self.show_plot:
            plt.show()

        plt.close()

    def reconstruction_error(self, bins = np.linspace(0, 2, 50), train = True, anomaly =False, after_anomaly=False):
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        f = self.image_folder + self.name
        if train:
            ax1.hist(self.model_mse(self.X_train), bins=bins, range=[0, 2.5],density=True, label='Training Sample', alpha=1.0, hatch='/', edgecolor='k', fill=True)
        ax1.hist(self.model_mse(self.X_test), bins=bins, density=True, label='Testing Sample', alpha=0.5, hatch='\\', edgecolour='k', fill=True)
        if anomaly:
            f += "_anomaly"
            ax1.hist(self.model_mse(self.X_anomaly), bins=bins, density=True, label='Anomaly Sample', alpha=0.5, hatch='+', edgecolor='k', fill=True)
        if after_anomaly:
            f += "_after_anomaly"
            ax1.hist(self.model_mse(self.X_after), bins=bins, density=True, label='After Anomaly Sample', alpha=0.5, hatch='/', edgecolor='k', fill=True)
        ax1.legend()
        ax1.set_xlabel('Reconstruction Error')
        plt.title("Reconstruction Error " + self.name)
        plt.savefig(f + "_reconstruction_error.png", transparent=True, dpi=300, bbox_inches='tight')
        if self.show_plot:
            plt.show()

        plt.close(fig)

    def pdf(self, bins = np.linspace(-16, -4, 50), train = True, anomaly=False):
        kd = KernelDensity()
        kd.fit(self.model.encode(self.X_train))
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        if train:
            train_score = [kd.score(x.reshape(1, -1)) for x in self.model.encode(self.X_train)]
            ax1.hist(train_score, bins=bins, label='Train Sample', density=True, alpha=1.0)

        test_score = [kd.score(x.reshape(1, -1)) for x in self.model.encode(self.X_test)]
        ax1.hist(test_score, bins=bins, label='Test Sample', density=True, alpha=1)
        if anomaly:
            anom_score = [kd.score(x.reshape(1, -1)) for x in self.model.encode(self.X_anomaly)]
            ax1.hist(anom_score, bins=bins, label='Anomaly Sample', density=True, alpha=0.5)
            print('Anomaly score', np.mean(anom_score))

        ax1.legend()
        plt.title("PDF " + self.name)

        print('Test data score', np.mean(test_score))
        plt.savefig(self.image_folder + self.name + "_pdf.png", transparent=True, bbox_inches='tight', dpi=300)

        if self.show_plot:
            plt.show()

        plt.close(fig)


    def roc(self):
        mse_score = np.concatenate([self.model_mse(self.X_test), self.model_mse(self.X_anomaly)], 0)
        print(mse_score)
        print(mse_score.shape)
        true_label = [0] * self.X_test.shape[0] + [1] * self.X_anomaly.shape[0]
        print(len(true_label))
        if roc_auc_score(true_label, mse_score) < 0.5:
            mse_score *= -1
        fpr, tpr, thresholds = roc_curve(true_label, mse_score)
        auc_score = roc_auc_score(true_label, mse_score)
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        ax1.plot(fpr, tpr, 'b.-', label='ROC Curve (%2.2f)' % auc_score)
        ax1.plot(fpr, fpr, 'k-', label='Random Guessing')
        ax1.legend()
        plt.title("ROC Curve " + self.name)
        plt.savefig(self.image_folder + self.name + "_roc.png", transparent=True, bbox_inches='tight', dpi=300)

        if self.show_plot:
            plt.show()

    def reconstruction_error_time(self, limit = 0, train = True, test = True, anomaly = False, after_anomaly=False):
        fig, ax = plt.subplots()
        f = self.image_folder + self.name
        if test:
            time_list = self.meta_test[:, 2]
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_test[:, 2]],
                        self.model_mse(self.X_test), c='g')
        if train:
            f += "_train"
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_train[:, 2]], self.model_mse(self.X_train), c='b')
            if test:
                time_list = np.concatenate([time_list, self.meta_train[:, 2]], 0)
            else:
                time_list = self.meta_train[:, 2]

        if anomaly:
            f += "_anomaly"
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_anomaly[:, 2]],
                        self.model_mse(self.X_anomaly), c='r')
            time_list = np.concatenate([time_list, self.meta_anomaly[:, 2]], 0)

        if after_anomaly:
            f += "_after"
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_after[:, 2]],
                        self.model_mse(self.X_after), c='g')
            time_list = np.concatenate([time_list, self.meta_after[:, 2]], 0)

        width = np.diff(time_list).min()
        ax.xaxis_date()

        # Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()
        if limit > 0:
            f += "_" + str(limit)
            plt.ylim(0, limit)
        plt.title("Reconstruction error over time " + self.name)
        plt.xlabel("Sample timestamp")
        plt.ylabel("Reconstruction error")
        # plt.figure(figsize=(6, 6))
        plt.savefig(f + "_reconstruction_error_time.png", transparent=True, dpi=300, bbox_inches='tight')

        if self.show_plot:
            plt.show()

    def reconstruction_error_time_moving_avg(self, limit=0, train=True, test=True, anomaly=False, window_size=6):
        fig, ax = plt.subplots()
        f = self.image_folder + self.name
        if test:
            time_list = self.meta_test[:, 2]
            moving_res = self.moving_average(self.model_mse(self.X_test), window_size)
            time_list = time_list[0:len(time_list) - window_size + 1]
            plt.scatter([datetime.fromtimestamp(x) for x in time_list], moving_res, c='g')
        if train:
            f += "_train"
            time_list = self.meta_train[:, 2]
            moving_res = self.moving_average(self.model_mse(self.X_train), window_size)
            time_list = time_list[0:len(time_list) - window_size + 1]
            plt.scatter([datetime.fromtimestamp(x) for x in time_list], moving_res, c='b')
            if test:
                time_list = np.concatenate([time_list, self.meta_train[:, 2]], 0)


        if anomaly:
            f += "_anomaly"
            time_list = self.meta_anomaly[:, 2]
            moving_res = self.moving_average(self.model_mse(self.X_anomaly), window_size)
            time_list = time_list[0:len(time_list) - window_size + 1]
            plt.scatter([datetime.fromtimestamp(x) for x in time_list], moving_res, c='r')
            time_list = np.concatenate([time_list, self.meta_anomaly[:, 2]], 0)

        # width = np.diff(time_list).min()
        ax.xaxis_date()

        # Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()
        if limit > 0:
            f += "_" + str(limit)
            plt.ylim(0, limit)
        plt.title("Reconstruction error over time moving average" + self.name)
        plt.xlabel("Sample timestamp")
        plt.ylabel("Reconstruction error")
        # plt.figure(figsize=(6, 6))
        plt.savefig(f + "_reconstruction_error_time_moving_avg.png", transparent=True)

        if self.show_plot:
            plt.show()

        plt.close(fig)

    def moving_average(self, X, window_size):
        res = []
        for i in range(0, len(X) - window_size + 1):
            w_avg = sum(X[i:i + window_size]) / window_size
            res.append(w_avg)

        return res

    def mean_absolute_vibration(self, limit=0, train=True, test=True, anomaly=False, after_anomaly=False):

        fig, ax = plt.subplots()
        f = self.image_folder + self.name
        if test:
            abs = np.absolute(self.X_test)
            mean_vib = np.transpose(np.mean(abs, axis=1))
            time_list = self.meta_test[:, 2]
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_test[:, 2]],
                        mean_vib[0], c='limegreen', label="test vibx")
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_test[:, 2]],
                        mean_vib[1], c='forestgreen', label="test vibz")
        if train:
            f += "_train"
            abs = np.absolute(self.X_train)
            mean_vib = np.transpose(np.mean(abs, axis=1))
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_train[:, 2]],
                        mean_vib[0], c='cornflowerblue', label="train vibx")
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_train[:, 2]],
                        mean_vib[1], c='royalblue', label="train vibz")
            if test:
                time_list = np.concatenate([time_list, self.meta_train[:, 2]], 0)
            else:
                time_list = self.meta_train[:, 2]

        if anomaly:
            f += "_anomaly"
            abs = np.absolute(self.X_anomaly)
            mean_vib = np.transpose(np.mean(abs, axis=1))
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_anomaly[:, 2]],
                        mean_vib[0], c='lightcoral', label="anomaly vibx")
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_anomaly[:, 2]],
                        mean_vib[1], c='firebrick', label="anomaly vibz")
            time_list = np.concatenate([time_list, self.meta_anomaly[:, 2]], 0)

        if after_anomaly:
            abs = np.absolute(self.X_after)
            mean_vib = np.transpose(np.mean(abs, axis=1))
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_after[:, 2]],
                        mean_vib[0], c='limegreen', label="test vibx")
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_after[:, 2]],
                        mean_vib[1], c='forestgreen', label="test vibz")
            time_list = np.concatenate([time_list, self.meta_anomaly[:, 2]], 0)


        width = np.diff(time_list).min()
        ax.xaxis_date()

        # Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()
        if limit > 0:
            f += "_" + str(limit)
            plt.ylim(0, limit)
        plt.title("Mean absolute vibration over time")
        plt.xlabel("Sample timestamp")
        plt.ylabel("Vibration")
        # plt.figure(figsize=(6, 6))
        plt.savefig(f + "_avg_vibration_time.png", transparent=True, dpi=300, bbox_inches='tight')

        if self.show_plot:
            plt.show()

        plt.close(fig)

    def reconstruction_error_time_influx(self, train, test, model_name, write_dict):

        if test:
            time_list = self.meta_test[:, 2]
            results = self.model_mse(self.X_test)

        if train:
            if test:
                time_list = np.concatenate([time_list, self.meta_train[:, 2]], 0)
                results = np.concatenate([results, self.model_mse(self.X_train)], 0)
            else:
                time_list = self.meta_train[:, 2]
                results = self.model_mse(self.X_train)

        points = results_to_influx.make_results_points(time_list, results, self.name, model_name)
        results_to_influx.write_results_to_influx(write_dict, points)


    def rpm_time(self, limit = 0, train = True, anomaly=False):
        fig, ax = plt.subplots()
        f = self.image_folder + self.name

        if train:
            f += "_train"
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_train[:, 2]], self.meta_train[:, 1], c='b')
        plt.scatter([datetime.fromtimestamp(x) for x in self.meta_test[:, 2]],
                    self.meta_test[:, 1], c='g')
        if anomaly:
            f += "_anomaly"
            plt.scatter([datetime.fromtimestamp(x) for x in self.meta_anomaly[:, 2]],
                    self.meta_anomaly[:, 1], c='r')
        ax.xaxis_date()
        # Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()
        if limit > 0:
            f += "_" + str(limit)
            plt.ylim(0, limit)
        plt.title("RPM over time " + self.name)
        plt.xlabel("Sample timestamp")
        plt.ylabel("RPM")
        plt.figure(figsize=(6, 6))
        plt.savefig(f + "_rpm_time.png", transparent=True)

        if self.show_plot:
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
