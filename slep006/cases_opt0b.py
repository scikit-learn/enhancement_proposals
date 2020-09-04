import pandas as pd
from defs import (accuracy, group_cv, get_scorer, SelectKBest,
                  LogisticRegressionCV, cross_validate,
                  make_pipeline, X, y, my_groups, my_weights,
                  my_other_weights)

X = pd.DataFrame(X)
MY_GROUPS = pd.Series(my_groups)
MY_WEIGHTS = pd.Series(my_weights)
MY_OTHER_WEIGHTS = pd.Series(my_other_weights)

# %%
# Case A: weighted scoring and fitting


class WrappedGroupCV:
    def __init__(self, base_cv):
        self.base_cv = base_cv

    def split(self, X, y, groups=None):
        return self.base_cv.split(X, y, groups=MY_GROUPS.loc[X.index])

    def get_n_splits(self, X, y, groups=None):
        return self.base_cv.get_n_splits(X, y, groups=MY_GROUPS.loc[X.index])


wrapped_group_cv = WrappedGroupCV(group_cv)


class WeightedLogisticRegressionCV(LogisticRegressionCV):
    def fit(self, X, y):
        return super().fit(X, y, sample_weight=MY_WEIGHTS.loc[X.index])


acc_scorer = get_scorer('accuracy')


def wrapped_weighted_acc(est, X, y, sample_weight=None):
    return acc_scorer(est, X, y, sample_weight=MY_WEIGHTS.loc[X.index])


lr = WeightedLogisticRegressionCV(
    cv=wrapped_group_cv,
    scoring=wrapped_weighted_acc,
).set_props_request(['sample_weight'])
cross_validate(lr, X, y,
               cv=wrapped_group_cv,
               scoring=wrapped_weighted_acc)

# %%
# Case B: weighted scoring and unweighted fitting

lr = LogisticRegressionCV(
    cv=wrapped_group_cv,
    scoring=wrapped_weighted_acc,
).set_props_request(['sample_weight'])
cross_validate(lr, X, y,
               cv=wrapped_group_cv,
               scoring=wrapped_weighted_acc)


# %%
# Case C: unweighted feature selection

lr = WeightedLogisticRegressionCV(
    cv=wrapped_group_cv,
    scoring=wrapped_weighted_acc,
).set_props_request(['sample_weight'])
sel = SelectKBest()
pipe = make_pipeline(sel, lr)
cross_validate(pipe, X, y,
               cv=wrapped_group_cv,
               scoring=wrapped_weighted_acc)

# %%
# Case D: different scoring and fitting weights


def other_weighted_acc(est, X, y, sample_weight=None):
    return acc_scorer(est, X, y, sample_weight=MY_OTHER_WEIGHTS.loc[X.index])


lr = WeightedLogisticRegressionCV(
    cv=wrapped_group_cv,
    scoring=other_weighted_acc,
).set_props_request(['sample_weight'])
sel = SelectKBest()
pipe = make_pipeline(sel, lr)
cross_validate(pipe, X, y,
               cv=wrapped_group_cv,
               scoring=other_weighted_acc)
