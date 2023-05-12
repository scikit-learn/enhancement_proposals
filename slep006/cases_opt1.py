from defs import (accuracy_score, GroupKFold, make_scorer, SelectKBest,
                  LogisticRegressionCV, cross_validate, make_pipeline, X, y,
                  my_groups, my_weights, my_other_weights)

# %%
# Case A: weighted scoring and fitting

lr = LogisticRegressionCV(
    cv=GroupKFold(),
    scoring='accuracy',
)
cross_validate(lr, X, y, cv=GroupKFold(),
               props={'sample_weight': my_weights, 'groups': my_groups},
               scoring='accuracy')

# Error handling: if props={'sample_eight': my_weights, ...} was passed
# instead, the estimator would fit and score without weight, silently failing.

# %%
# Case B: weighted scoring and unweighted fitting


class MyLogisticRegressionCV(LogisticRegressionCV):
    def fit(self, X, y, props=None):
        props = props.copy()
        props.pop('sample_weight', None)
        super().fit(X, y, props=props)


# %%
# Case C: unweighted feature selection

# Currently feature selection does not handle sample_weight, and as long as
# that remains the case, it will simply ignore the prop passed to it. Hence:

lr = LogisticRegressionCV(
    cv=GroupKFold(),
    scoring='accuracy',
)
sel = SelectKBest()
pipe = make_pipeline(sel, lr)
cross_validate(pipe, X, y, cv=GroupKFold(),
               props={'sample_weight': my_weights, 'groups': my_groups},
               scoring='accuracy')

# %%
# Case D: different scoring and fitting weights

weighted_acc = make_scorer(accuracy_score)


def specially_weighted_acc(est, X, y, props):
    props = props.copy()
    props['sample_weight'] = 'scoring_weight'
    return weighted_acc(est, X, y, props)


lr = LogisticRegressionCV(
    cv=GroupKFold(),
    scoring=specially_weighted_acc,
)
cross_validate(lr, X, y, cv=GroupKFold(),
               props={
                    'scoring_weight': my_weights,
                    'sample_weight': my_other_weights,
                    'groups': my_groups,
               },
               scoring=specially_weighted_acc)
