==================================
SLEP006: Propagating feature names
==================================

:Author: Andreas Mueller, Joris Van den Bossche
:Status: Draft
:Type: Standards Track
:Created: 2019-03-01

Abstract
--------

This SLEP proposes to add a transformative ``get_feature_names()`` method that
gets recursively called on each step of a ``Pipeline`` so that the feature
names get propagated throughout the full ``Pipeline``. This will allow to
inspect the input and output feature names in each step of a ``Pipeline``.


Detailed description
--------------------

Motivating example
^^^^^^^^^^^^^^^^^^

We've been making it easier to build complex workflows with the
ColumnTransformer and we expect it will find wide adoption. However, using it
results in very opaque models, even more so than before.

We have a great usage example in the gallery that applies a classifier to the
titanic data set. This is a very simple standard usecase, but it still close to
impossible to inspect the names of the features that went into the final
estimator, for example to match this with the coefficients or feature
importances.

The full pipeline construction can be seen seen at
https://scikit-learn.org/dev/auto_examples/compose/plot_column_transformer_mixed_types.html,
but it consists of a final classifier and a preprocessor that scales the
numerical features and one-hot encodes the categorical features. To obtain the
feature names that correspond to what the final classifier receives, we can do
now::


    numeric_features = ['age', 'fare']
    categorical_features = ['embarked', 'sex', 'pclass']
    # extract OneHotEncoder from pipeline
    onehotencoder = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    categorical_features2 = onehotencoder.get_feature_names(
        input_features=categorical_features)
    feature_names = numeric_features + list(categorical_features2)
    
Note: this only works if the Imputer didn't drop any all-NaN columns.

This SLEP proposes that the input features derived on fit time are progragated through and stored in all the steps of the pipeline, so then we can acces them like this:

    clf.named_steps['classifier'].input_feature_names_
    
For this specific example, this would give::

    >>> clf.named_steps['classifier'].input_feature_names_
    ['num__age', 'num__fare', 'cat__embarked_C', 'cat__embarked_Q', 'cat__embarked_S', 'cat__embarked_missing', 'cat__sex_female', 'cat__sex_male', 'cat__pclass_1', 'cat__pclass_2', 'cat__pclass_3']

and which then could be easily matched with eg::

    >>> clf.named_steps['classifier'].coef_


Propagating feature names through Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The core idea of this proposal is that all transformers get a transformative *"update feature names"* method that can determine the output feature names, optionally given input feature names (specified with the `input_features` parameter of this method).

The `Pipeline` and all meta-estimators implement this method by calling it recursively on all its child steps / sub-estimators, and in this way the input features names get propagated through the full pipeline.
In addition, it sets the `input_feature_names_` attribute on each step of the pipeline.


A Pipeline calls the method at the end of `fit()` (using the DataFrame column names or generated 'x0, 'x1', .. names for numpy arrays or sparse matrices as initial input feature names), which ensures that a fitted pipeline has the input features set, without the need for the user to first call the "update feature names" method before those attributes are available.

The "update feature names" method that is added to all transformers does the following things:

- ability to specify custom input feature names (the `input_features` parameter)
- set the `input_feature_names_` attribute (only done by `Pipeline` and `MetaEstimator`)
- transform the input feature names into output feature names
- return the output feature names




onehotencoder.set_feature_names(categorical_features)
onehotencoder.output_features_

Implementation
--------------

There is an implementation of this proposal in PR #13307 (https://github.com/scikit-learn/scikit-learn/pull/13307).


Open design questions
^^^^^^^^^^^^^^^^^^^^^


#### Name of the "update feature names" method?

Currently, a few transformers already have a `get_feature_names` method (more specifically the vectorizers, PolynomialFeatures and OneHotEncoder). Moreover, in some cases this method already does exactly what we need (accepting `input_features` and returning the transformed ones).

However, there are some downsides about this name: for `Pipeline` and meta-estimators (and potentially also other transformers, see question below), this method also *sets* the `input_feature_names_` attribute on each of the estimators. Further, it may not directly be clear from this name that it returns the *output* feature names and not the input feature names. 

In addition to re-using `get_feature_names`, some ideas for other names: `set_feature_names`, `get_output_names`, `transform_feature_names`, `propagate_feature_names`, `update_feature_names`, ...


#### Name of the attribute: `input_feature_names_` vs `input_features_` vs `input_names_`?

The most explicit is `input_feature_names_`, but this is long. `input_features_` can probably to easily be confused with the actual input *features* (the X data). `input_names_` can provide a shorter alternative.

`input_features_` is already used as parameter name in `get_feature_names`, which is a plus. Probably, whatever name we choose, it would be good to eventually make this consistent with the keyword used in the "update feature names" method.

An example code snippet::

    >>> clf.named_steps['classifier'].input_feature_names_

With pipeline slicing, this could become::

    >>> clf['classifier'].input_feature_names_
    >>> clf[-1].input_feature_names_


Other mentioned alternative: `feature_names_in_` and `feature_names_out_`.


#### Do we also want a `output_feature_names_` attribute?

In addition to setting the `input_feature_names_` attribute on each estimator, we could also have a `output_feature_names_`.

This would return the same as the current `get_feature_names()` method, potentially removing the need to have an explicit output feature names *getter method*. 
The "update feature names" method would then mainly be used for setting the input features and making sure they get propagated. 


#### What should the "update feature names" method do in the less obvious cases?

The clear cases on how to transform input to output features are:

- Transformers that pass through (One-to-one), e.g. StandardScaler
- Transformers that generate new features, e.g. OneHotEncoder, Vectorizers
- Transformers that output a subset of the original features (SelectKBest?)

But, what to do with:

- Transformers that create linear combinations, eg PCA 
- Transformers based on arbitrary functions




#### Should all estimators (so including regressors, classifiers, ...) have a "update feature names" method?

In the current implementation, only transformers (and Pipeline and meta-estimators, which could act as transformer) have a "update feature names" method.

For consistency, we could also add them to *all* estimators.

For a regressor or classifier, the method could set the `input_feature_names_` attribute and return `None`.



#### Should all estimators call the "update feature names" method inside `fit()` ?

In the current implementation, the `Pipeline` is responsible to catch the input feature names and calling the "update feature names" method with those names at the end of `fit()`.

However, that means that if you use a transformer/estimator in itself and not in a Pipeline, we won't have the feature names automatically set (assuming `X_df` is a DataFrame with columns A and B)::

    >>> ohe = OneHotEncoder()
    >>> ohe.fit(X_df)
    >>> ohe.input_feature_names_
    AttributeError: ...
    >>> ohe.get_feature_names()
    ['x0_cat1', 'x0_cat2', 'x1_cat1', 'x2_cat2']

vs

    >>> ohe_pipe = Pipeline([('ohe', OneHotEncoder())])
    >>> ohe_pipe.fit(X_df)
    >>> ohe.input_feature_names_
    ['A', 'B']
    >>> ohe.get_feature_names()
    ['A_cat1', 'A_cat2', 'B_cat1', 'B_cat2']


Currently, the `input_feature_names_` attribute of an estimator is set by the "update feature names" method of the parent estimator. But, this logic could also be moved into the "update feature names" method of the estimator itself. 

In that case, the `fit` method of the estimator could also call the "update feature names" method at the end, ensuring consistency between on-itself standing estimators and Pipelines.  However, the clear downside of this consistency is that this would add one line to each `fit` method throughout scikit-learn.



update_feature_names(['a', 'b'])
-> set .input_features_



--------



-----

Vectorizers

countvectorizer.update_feature_names(None)


ohe.fit(df)
ohe.get_feature_names(df.columns)


"update feature names":
- 1) input features keyword
- 2) self.input_features_
- 3) generate x0, x1, ...


pipeline and MetaEstimator:
inside fit: call recursively "update feature names" on all steps after fitting



#### What happens if one part of the pipeline does not implement "update feature names"?

Instead of raising an error, the current PR sets the output feature names to None, which if passed to the next step of the pipeline, allows it to still generate feature names.

#### Interaction with column validation

Another, potentially related, change that has been discussed is to do input validation on transform/predict time: ensuring that the column names and order is identical when transforming/predicting compared to fit (currently, scikit-learn silently returns "incorrect" results as long as the number of columns matches).

To do proper validation, the idea would be to store the column names at fit time, so they can be compared at transform/predict time. 
Those stored column names could be very similar to the `input_feature_names_` described in this SLEP. 

However, if a user calls `Pipeline.get_feature_names(input_features=[...])` with a set of custom input feature names that are not identical to the original DataFrame column names, the stored column names to do validation and the stored column names to propagate the feature names would get out of sync. Or should calling `get_feature_names` also affect future validation in a `predict()` call?

One solution is to disallow setting feature names if the original input are pandas DataFrames (so `pipe.get_feature_names(['other', 'names'])` would raise an error if `pipe` was fitted with a DataFrame). This would prevent ending up in potentially confusing or ambiguous situations. 
Calling `get_feature_names` with custom input names is of course still possible when the input was not a pandas DataFrame.


Backward compatibility
----------------------

This SLEP does not affect backward compatibility, as all described attributes and methods would be new ones, not affecting existing ones.

The only possible compatibility question is, if we decide to use another name than `get_feature_names()`, what to do with those existing methods? Those could in principle be deprecated.


Alternatives
------------

The alternatives described here are alternatives to the combination of the transformative "update feature names" method and calling it recursively in the Pipeline setting the `input_feature_names_` attribute on each step.


1. Only implement the "update feature names" method and require the user to slice the pipeline to call this method manually on the appropriate subset.
  For example::

        clf[:-1].get_feature_names(input_feature=[....])

  would then propagate the original provided names up to the output names of the final step of this sliced pipeline.
  
  This is what was implemented in https://github.com/scikit-learn/scikit-learn/pull/12627. 
  
  The main drawback of this more limited proposal is the user interface: the user needs to manually slice the pipeline and call `get_feature_names()` to get the output feature names of this subset, in order to get the input feature names of the final classifier/regressor.
  
  only difference is not setting the `input_features_` attributes


2. Use "pandas in - pandas out" everywhere (also fitted attributes): user does not need an explicit way to get or set the feature names as they are included in the output of estimators

 only output:
    
    clf[:-1].transform(X).columns

  all fitted attributes: `coef_` is a Series with the feature names
  
All of those need th

2. Implement a more comprehensive feature description language (as done in ELI-5?)
User interface:

- only get_feature_names and slicing of the pipeline

clf[:-1].get_feature_names()



    

If there were any alternative solutions to solving the same problem, they
should be discussed here, along with a justification for the chosen
approach.



To me there are four main options for interfaces to enable this:

    Implement transformative get_feature_names as in this PR
    Implement a more comprehensive feature description language (as done in ELI-5, I think)
    Tie in more strongly with pandas and use dataframes / column names
    a) to output feature semantics.
    b) to determine feature semantics
    Leave it to the user.

While I think 2) and 3) a) is are valid option for the future, I think trying to implement this now will probably result in a gridlock and/or take too much time. I think we should iterate and provide something that solves the 80% use-case quickly. We can create a more elaborate solution later, in particular since this proposal/PR doesn't introduce any concepts that are not in sklearn already.
3 b) is discussed below.

I don't think 4) is a realistic option. I assume we can agree that the titanic example above is a valid use-case, and that getting the semantics of features is important. Below is the code that the user would have to write to do this themselves. This will become even harder in the future if the pipeline will do cloning.



Discussion
----------



This section may just be a bullet list including links to any discussions
regarding the SLEP:

- This includes links to mailing list threads or relevant GitHub issues.


References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
