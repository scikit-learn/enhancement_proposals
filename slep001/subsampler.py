from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


class SubSampler(BaseEstimator):

    def __init__(self, ratio=.3, random_state=None):
        self.ratio = ratio
        self.random_state = random_state
        self.random_state_ = None

    def transform_pipe(self, X, y=None):
        # Awkward situation: random_state_ is set at transform time :)
        if self.random_state_ is None:
            self.random_state_ = check_random_state(self.random_state)
        n_samples, _ = X.shape
        random_choice = self.random_state_.random_sample(n_samples)
        random_choice = random_choice < self.ratio
        X_out = X[random_choice]
        y_out = None
        if y is not None:
            y_out = y[random_choice]
        return X_out, y_out

