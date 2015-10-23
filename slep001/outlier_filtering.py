from sklearn.base import BaseEstimator
from sklearn.covariance import EllipticEnvelope


class EllipticEnvelopeFilter(BaseEstimator):

    def __init__(self, assume_centered=False,
                 support_fraction=None, contamination=0.1,
                 random_state=None):
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.random_state = random_state

    def fit_pipe(self, X, y=None):
        self.elliptic_envelope_ = EllipticEnvelope(**self.get_params())
        self.elliptic_envelope_.fit(X)
        return self.transform_pipe(X, y)

    def transform_pipe(self, X, y):
        # XXX: sample_props not taken care off
        is_inlier = self.elliptic_envelope_.predict(X) == 1
        X_out = X[is_inlier]
        if y is None:
            y_out = None
        else:
            y_out = y[is_inlier]
        return X_out, y_out

    def transform(self, X, y=None):
        return X

