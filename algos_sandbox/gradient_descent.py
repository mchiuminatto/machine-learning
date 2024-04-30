import matplotlib.pyplot as plt
import numpy as np
import abc


class GradientDescent(abc.ABC):

    def __init__(self, eta, epsilon):

        self.epsilon = epsilon
        self.eta = eta
        self.best_fit = None
        self.models: np.array = np.array([])
        self.norms: np.array = np.array([])

    @abc.abstractmethod
    def train(self, X, y):
        pass

    def predict(self, X) -> np.array:
        if self.best_fit is None:
            raise ValueError("No trained model")

        y = X.dot(self.best_fit.T)
        return y

    def plot_norm(self, ax=plt):

        ax.plot(self.norms)
        ax.set_title("Gradient norms")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Gradient Norm")

    def plot_fit(self, X, y, ax=plt):
        """
        WRNING: For now Only supports one feature (and bias term)
        """
        plt_obj = ax.scatter(X, y, s=1)

        for model in self.models:
            y_pred = model[1] * X + model[0]
            ax.plot(X, y_pred, label=f"{model[1]}x + {model[0]}", alpha=0.2)
        ax.set_title("Fit evolution")


class StochasticGradientDescent(GradientDescent):

    def __init__(self, eta: float,
                 epsilon: float,
                 epochs: int = 100,
                 tolerance: float = 1e-5,
                 mult_method: str = "dot-product",
                 ):

        super().__init__(eta, epsilon)

        self.epochs = epochs
        self.tolerance = tolerance

        self.epochs: int = epochs
        self.mult_method: str = mult_method

        self._t0 = 5
        self._t1 = 50

    def train(self, X, y):

        m = X.shape[0]  # instances
        n = X.shape[1]  # features
        X_bias = np.c_[np.ones(m), X]  # add bias term as column
        theta = np.random.randn(n + 1, 1)  # column vector

        self.models = np.array([theta])

        for epoch in range(self.epochs):
            for instance in range(m):

                # select random instance
                instance_index = np.random.randint(m)
                X_i = X_bias[instance_index: instance_index + 1]
                y_i = y[instance_index].reshape(1)

                if self.mult_method == "dot-product":
                    # vectorized version of gradient calculation  dot product
                    gradient = 2*(X_i.T).dot(X_i.dot(theta) - y_i)
                elif self.mult_method == "numpy-mult":
                    # matrix multiplication vector (numpy)
                    gradient = 2*np.matmul(X_i.T, (np.matmul(X_i, theta) - y))
                elif self.mult_method == "python-mult":
                    # matrix multiplication vector (numpy)
                    gradient = 2*(X_i.T @ (X_i @ theta - y))
                else:
                    raise ValueError("Matrix multiplication method not recognized")

                # compute gradient's norm
                norm = np.linalg.norm(gradient)

                # update parameters
                eta = self.learning_schedule(epoch * m + instance)
                theta = theta - eta * gradient

            self.models = np.append(self.models, [theta], axis=0)
            self.norms = np.append(self.norms, [norm], axis=0)

        self.best_fit = theta

    def learning_schedule(self, t):
        return self._t0/(t + self._t1)


class BatchGradientDescent(GradientDescent):
    def __init__(self, eta: float, epsilon: float, mult_method: str = "dot-product"):

        super().__init__(eta, epsilon)

        self.epochs: int = 0
        self.mult_method: str = mult_method

    def train(self, X, y):

        m = X.shape[0]  # instances
        n = X.shape[1]  # features
        X_bias = np.c_[np.ones(m), X]  # add bias term as column
        theta = np.random.randn(n + 1,1)  # column vector

        self.models = np.array([theta])
        
        print("Checking shape ...")
        print("X_bias shape ", X_bias.shape)
        print("y shape ", y.shape)
        print("theta shape ", theta.shape)

        norm = np.inf
        norms: np.array = np.array([])
        
        while norm > self.eta:

            if self.mult_method == "dot-product":
                # vectorized version of gradient calculation  dot product
                gradient = (2/m)*(X_bias.T).dot(X_bias.dot(theta) - y)
            elif self.mult_method == "numpy-mult":
                # matrix multiplication vector (numpy)
                gradient = (2/m)*np.matmul(X_bias.T, (np.matmul(X_bias, theta) - y))

            elif self.mult_method == "python-mult":
                # matrix multiplication vector (numpy)
                gradient = (2/m)*(X_bias.T @ (X_bias @ theta - y))
            else:
                raise ValueError("Matrix multiplication method not recognized")

            # calculate gradient's norm
            norm = np.linalg.norm(gradient)
            self.norms = np.append(self.norms, [norm], axis=0)

            #  update parameters
            theta = theta - self.eta*gradient
            self.models = np.append(self.models, [theta], axis=0)

            self.epochs += 1

        self.best_fit = theta
