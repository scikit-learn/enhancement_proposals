import numpy as np

from defs import (accuracy, group_cv, get_scorer, SelectKBest,
                  LogisticRegressionCV, cross_validate,
                  make_pipeline, X, y, my_groups, my_weights,
                  my_other_weights)

# %%
# Case A: weighted scoring and fitting


GROUPS_IDX = -1
WEIGHT_IDX = -2


def unwrap_X(X):
    return X[:, -2:]


class WrappedGroupCV:
    def __init__(self, base_cv, groups_idx=GROUPS_IDX):
        self.base_cv = base_cv
        self.groups_idx = groups_idx

    def split(self, X, y, groups=None):
        groups = X[:, self.groups_idx]
        return self.base_cv.split(unwrap_X(X), y, groups=groups)

    def get_n_splits(self, X, y, groups=None):
        groups = X[:, self.groups_idx]
        return self.base_cv.get_n_splits(unwrap_X(X), y, groups=groups)


wrapped_group_cv = WrappedGroupCV(group_cv)


class WrappedLogisticRegressionCV(LogisticRegressionCV):
    def fit(self, X, y):
        return super().fit(unwrap_X(X), y, sample_weight=X[:, WEIGHT_IDX])


acc_scorer = get_scorer('accuracy')


def wrapped_weighted_acc(est, X, y, sample_weight=None):
    return acc_scorer(est, unwrap_X(X), y, sample_weight=X[:, WEIGHT_IDX])


lr = WrappedLogisticRegressionCV(
    cv=wrapped_group_cv,
    scoring=wrapped_weighted_acc,
).set_props_request(['sample_weight'])
cross_validate(lr, np.hstack([X, my_weights, my_groups]), y,
               cv=wrapped_group_cv,
               scoring=wrapped_weighted_acc)

# %%
# Case B: weighted scoring and unweighted fitting

class UnweightedWrappedLogisticRegressionCV(LogisticRegressionCV):
    def fit(self, X, y):
        return super().fit(unwrap_X(X), y)


lr = UnweightedWrappedLogisticRegressionCV(
    cv=wrapped_group_cv,
    scoring=wrapped_weighted_acc,
).set_props_request(['sample_weight'])
cross_validate(lr, np.hstack([X, my_weights, my_groups]), y,
               cv=wrapped_group_cv,
               scoring=wrapped_weighted_acc)


# %%
# Case C: unweighted feature selection

class UnweightedWrappedSelectKBest(SelectKBest):
    def fit(self, X, y):
        return super().fit(unwrap_X(X), y)


lr = WrappedLogisticRegressionCV(
    cv=wrapped_group_cv,
    scoring=wrapped_weighted_acc,
).set_props_request(['sample_weight'])
sel = UnweightedWrappedSelectKBest()
pipe = make_pipeline(sel, lr)
cross_validate(pipe, np.hstack([X, my_weights, my_groups]), y,
               cv=wrapped_group_cv,
               scoring=wrapped_weighted_acc)

# %%
# Case D: different scoring and fitting weights

SCORING_WEIGHT_IDX = -3

# TODO: proceed from here. Note that this change implies the need to add
# a parameter to unwrap_X, since we will now append an additional column to X.
