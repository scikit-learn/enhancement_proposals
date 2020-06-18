from defs import (accuracy, make_scorer, SelectKBest, LogisticRegressionCV,
                  group_cv, cross_validate, make_pipeline, X, y, my_groups,
                  my_weights, my_other_weights)

# %%
# Case A: weighted scoring and fitting

lr = LogisticRegressionCV(
    cv=group_cv,
    scoring='accuracy',
    prop_routing={'cv': ['groups'],
                  'scoring': ['sample_weight'],
                  }
    # one question here is whether we need to explicitly route sample_weight
    # to LogisticRegressionCV's fitting...
)

# Alternative syntax, which assumes cv receives 'groups' by default, and that a
# method-based API is provided on meta-estimators:
#   lr = LogisticRegressionCV(
#       cv=group_cv,
#       scoring='accuracy',
#   ).add_prop_route(scoring='sample_weight')

cross_validate(lr, X, y, cv=group_cv,
               props={'sample_weight': my_weights, 'groups': my_groups},
               scoring='accuracy',
               prop_routing={'estimator': '*',  # pass all props
                             'cv': ['groups'],
                             'scoring': ['sample_weight'],
                             })

# Error handling: if props={'sample_eight': my_weights, ...} was passed
# instead, LogisticRegressionCV would have to identify that a key was passed
# that could not be routed nor used, in order to raise an error.

# %%
# Case B: weighted scoring and unweighted fitting

# Here we rename the sample_weight prop so that we can specify that it only
# applies to scoring.
lr = LogisticRegressionCV(
    cv=group_cv,
    scoring='accuracy',
    prop_routing={'cv': ['groups'],
                  # read the following as "scoring should consume
                  # 'scoring_weight' as if it were 'sample_weight'."
                  'scoring': {'sample_weight': 'scoring_weight'},
                  },
)
cross_validate(lr, X, y, cv=group_cv,
               props={'scoring_weight': my_weights, 'groups': my_groups},
               scoring='accuracy',
               prop_routing={'estimator': '*',
                             'cv': ['groups'],
                             'scoring': {'sample_weight': 'scoring_weight'},
                             })

# %%
# Case C: unweighted feature selection

lr = LogisticRegressionCV(
    cv=group_cv,
    scoring='accuracy',
    prop_routing={'cv': ['groups'],
                  'scoring': ['sample_weight'],
                  })
pipe = make_pipeline(SelectKBest(), lr,
                     prop_routing={'logisticregressioncv': ['sample_weight',
                                                            'groups']})
cross_validate(lr, X, y, cv=group_cv,
               props={'sample_weight': my_weights, 'groups': my_groups},
               scoring='accuracy',
               prop_routing={'estimator': '*',
                             'cv': ['groups'],
                             'scoring': ['sample_weight'],
                             })

# %%
# Case D: different scoring and fitting weights
lr = LogisticRegressionCV(
    cv=group_cv,
    scoring='accuracy',
    prop_routing={'cv': ['groups'],
                  # read the following as "scoring should consume
                  # 'scoring_weight' as if it were 'sample_weight'."
                  'scoring': {'sample_weight': 'scoring_weight'},
                  },
)
cross_validate(lr, X, y, cv=group_cv,
               props={'scoring_weight': my_weights, 'groups': my_groups,
                      'fitting_weight': my_other_weights},
               scoring='accuracy',
               prop_routing={'estimator': {'sample_weight': 'fitting_weight',
                                           'scoring_weight': 'scoring_weight',
                                           'groups': 'groups'},
                             'cv': ['groups'],
                             'scoring': {'sample_weight': 'scoring_weight'},
                             })
