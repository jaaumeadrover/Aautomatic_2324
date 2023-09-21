import numpy as np


class Perceptron:

    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of miss classifications in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None  # defined in method fit

    def fit(self, X, y):

        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])  # First position corresponds to threshold
        # TODO: Put your code (fit algorithm)
        for _ in range(self.n_iter):            #for each iter
            for item_row,y_row in zip(X,y):     #for each row
                y_hat=self.predict(item_row)    #predict
                #Weight update
                update=self.eta*(y_row-y_hat)
                update_vector=update*item_row
                self.w_[1:]+=update_vector


    def predict(self, X):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """
        # TODO: Put your code
        pred=np.dot(X,self.w_[1:])+self.w_[0]
        return np.where(pred>= 1,1,-1)
