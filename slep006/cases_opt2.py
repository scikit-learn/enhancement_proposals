from defs import (group_cv, SelectKBest, LogisticRegressionCV,
                  cross_validate, make_pipeline, X, y, my_groups,
                  my_weights, my_other_weights)

# %%
# Case A: weighted scoring and fitting

lr = LogisticRegressionCV(
    cv=group_cv,
    scoring='accuracy',
)
props = {'cv__groups': my_groups,
         'estimator__cv__groups': my_groups,
         'estimator__sample_weight': my_weights,
         'scoring__sample_weight': my_weights,
         'estimator__scoring__sample_weight': my_weights}
cross_validate(lr, X, y, cv=group_cv,
               props=props,
               scoring='accuracy')

# error handling: if props={'estimator__sample_eight': my_weights, ...} was
# passed instead, the estimator would raise an error.

# %%
# Case B: weighted scoring and unweighted fitting

lr = LogisticRegressionCV(
    cv=group_cv,
    scoring='accuracy',
)
props = {'cv__groups': my_groups,
         'estimator__cv__groups': my_groups,
         'scoring__sample_weight': my_weights,
         'estimator__scoring__sample_weight': my_weights}
cross_validate(lr, X, y, cv=group_cv,
               props=props,
               scoring='accuracy')

# %%
# Case C: unweighted feature selection

lr = LogisticRegressionCV(
    cv=group_cv,
    scoring='accuracy',
)
pipe = make_pipeline(SelectKBest(), lr)
props = {'cv__groups': my_groups,
         'estimator__logisticregressioncv__cv__groups': my_groups,
         'estimator__logisticregressioncv__sample_weight': my_weights,
         'scoring__sample_weight': my_weights,
         'estimator__scoring__sample_weight': my_weights}
cross_validate(pipe, X, y, cv=group_cv,
               props=props,
               scoring='accuracy')

# %%
# Case D: different scoring and fitting weights

lr = LogisticRegressionCV(
    cv=group_cv,
    scoring='accuracy',
)
props = {'cv__groups': my_groups,
         'estimator__cv__groups': my_groups,
         'estimator__sample_weight': my_other_weights,
         'scoring__sample_weight': my_weights,
         'estimator__scoring__sample_weight': my_weights}
cross_validate(lr, X, y, cv=group_cv,
               props=props,
               scoring='accuracy')
