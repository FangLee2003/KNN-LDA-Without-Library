import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Visualization:
    def scatter(X, y, model):
        X_set, y_set = X, y
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                            np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

        fig = plt.figure(figsize=(6, 6))

        plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                    alpha=0.75, cmap=ListedColormap(('red', 'blue', 'yellow')))

        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())

        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('green', 'orange', 'purple'))(i), label=j, marker='.')

        # plt.title('K-NN (training set) using our implementation')
        plt.xlabel('Wine')
        plt.ylabel('Customer Segment')
        plt.legend()

        return fig