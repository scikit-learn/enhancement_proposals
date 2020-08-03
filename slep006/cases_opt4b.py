from defs import (accuracy, group_cv, make_scorer, SelectKBest,
                  LogisticRegressionCV, cross_validate,
                  make_pipeline, X, y, my_groups, my_weights,
                  my_other_weights)

# %%
# Case A: weighted scoring and fitting

# Here we presume that GroupKFold requests `groups` by default.
# We need to explicitly request weights in make_scorer and for
# LogisticRegressionCV. Both of these consumers understand the meaning
# of the key "sample_weight".

weighted_acc = make_scorer(accuracy, request_props=['sample_weight'])
lr = LogisticRegressionCV(
    cv=group_cv,
    scoring=weighted_acc,
).request_sample_weight(fit=['sample_weight'])
cross_validate(lr, X, y, cv=group_cv,
               props={'sample_weight': my_weights, 'groups': my_groups},
               scoring=weighted_acc)

# Error handling: if props={'sample_eight': my_weights, ...} was passed,
# cross_validate would raise an error, since 'sample_eight' was not requested
# by any of its children.

# %%
# Case B: weighted scoring and unweighted fitting

# Since LogisticRegressionCV requires that weights explicitly be requested,
# removing that request means the fitting is unweighted.

weighted_acc = make_scorer(accuracy, request_props=['sample_weight'])
lr = LogisticRegressionCV(
    cv=group_cv,
    scoring=weighted_acc,
)
cross_validate(lr, X, y, cv=group_cv,
               props={'sample_weight': my_weights, 'groups': my_groups},
               scoring=weighted_acc)

# %%
# Case C: unweighted feature selection

# Like LogisticRegressionCV, SelectKBest needs to request weights explicitly.
# Here it does not request them.

weighted_acc = make_scorer(accuracy, request_props=['sample_weight'])
lr = LogisticRegressionCV(
    cv=group_cv,
    scoring=weighted_acc,
).request_sample_weight(fit=['sample_weight'])
sel = SelectKBest()
pipe = make_pipeline(sel, lr)
cross_validate(pipe, X, y, cv=group_cv,
               props={'sample_weight': my_weights, 'groups': my_groups},
               scoring=weighted_acc)

# %%
# Case D: different scoring and fitting weights

# Despite make_scorer and LogisticRegressionCV both expecting a key
# sample_weight, we can use aliases to pass different weights to different
# consumers.

weighted_acc = make_scorer(accuracy,
                           request_props={'scoring_weight': 'sample_weight'})
lr = LogisticRegressionCV(
    cv=group_cv,
    scoring=weighted_acc,
).request_sample_weight(fit='fitting_weight')
cross_validate(lr, X, y, cv=group_cv,
               props={
                    'scoring_weight': my_weights,
                    'fitting_weight': my_other_weights,
                    'groups': my_groups,
               },
               scoring=weighted_acc)
