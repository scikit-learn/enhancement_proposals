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
titanic data set. This is a very simple standard use case, but it still close to
impossible to inspect the names of the features that went into the final
estimator, for example to match this with the coefficients or feature
importances.

The full pipeline construction can be seen seen at
https://scikit-learn.org/dev/auto_examples/compose/plot_column_transformer_mixed_types.html,
but it consists of a final classifier and a preprocessor that scales the
numerical features and one-hot encodes the categorical features. To obtain the
feature names that correspond to what the final classifier receives, we can do
now:

.. code-block:: python

    numeric_features = ['age', 'fare']
    categorical_features = ['embarked', 'sex', 'pclass']
    # extract OneHotEncoder from pipeline
    onehotencoder = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    categorical_features2 = onehotencoder.get_feature_names(
        input_features=categorical_features)
    feature_names = numeric_features + list(categorical_features2)
    
Note: this only works if the Imputer didn't drop any all-NaN columns.

This SLEP proposes that the input features derived on fit time are propagated
through and stored in all the steps of the pipeline, so then we can access them
like this:

.. code-block:: python

    clf.named_steps['classifier'].input_feature_names_
    
For this specific example, this would give::

    >>> clf.named_steps['classifier'].input_feature_names_
    ['num__age', 'num__fare', 'cat__embarked_C', 'cat__embarked_Q', 'cat__embarked_S', 'cat__embarked_missing', 'cat__sex_female', 'cat__sex_male', 'cat__pclass_1', 'cat__pclass_2', 'cat__pclass_3']

and which then could be easily matched with eg::

    >>> clf.named_steps['classifier'].coef_


Propagating feature names through Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The core idea of this proposal is that all transformers get a transformative
*"update feature names"* method that can determine the output feature names,
optionally given input feature names (specified with the ``input_features``
parameter of this method).

The ``Pipeline`` and all meta-estimators implement this method by calling it
recursively on all its child steps / sub-estimators, and in this way the input
features names get propagated through the full pipeline. In addition, it sets
the ``input_feature_names_`` attribute on each step of the pipeline.

A Pipeline calls the method at the end of ``fit()`` (using the DataFrame column
names or generated 'x0, 'x1', .. names for numpy arrays or sparse matrices as
initial input feature names), which ensures that a fitted pipeline has the
input features set, without the need for the user to first call the "update
feature names" method before those attributes are available.

The "update feature names" method that is added to all transformers does the
following things:

- ability to specify custom input feature names (the ``input_features`` parameter)
- otherwise, if not specified, generate default keyword names (eg "x0", "x1", ...)
- set the ``input_feature_names_`` attribute (only done by ``Pipeline``
  and ``MetaEstimator``)
- transform the input feature names into output feature names
- return the output feature names


Implementation
--------------

There is an implementation of this proposal in PR #13307
(https://github.com/scikit-learn/scikit-learn/pull/13307).

Open design questions
^^^^^^^^^^^^^^^^^^^^^

Name of the "update feature names" method?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, a few transformers already have a ``get_feature_names`` method (more
specifically the vectorizers, PolynomialFeatures and OneHotEncoder). Moreover,
in some cases this method already does exactly what we need (accepting
``input_features`` and returning the transformed ones).

However, there are some downsides about this name: for ``Pipeline`` and
meta-estimators (and potentially also other transformers, see question below),
this method also *sets* the ``input_feature_names_`` attribute on each of the
estimators. Further, it may not directly be clear from this name that it
returns the *output* feature names and not the input feature names. 

In addition to re-using ``get_feature_names``, some ideas for other names:
``set_feature_names``, ``get_output_feature_names``, ``transform_feature_names``,
``propagate_feature_names``, ``update_feature_names``, ...


Name of the attribute: ``input_feature_names_`` vs ``input_features_`` vs ``input_names_``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There was initially some discussion about the attribute to store the names.

The most explicit is ``input_feature_names_``, but this is long.
``input_features_`` can probably to easily be confused with the actual input
*features* (the X data). ``input_names_`` can provide a shorter alternative.

``input_features_`` is already used as parameter name in `get_feature_names`,
which is a plus. Probably, whatever name we choose, it would be good to
eventually make this consistent with the keyword used in the "update feature
names" method.

An example code snippet::

    >>> clf.named_steps['classifier'].input_feature_names_

With pipeline slicing, this could become::

    >>> clf['classifier'].input_feature_names_
    >>> clf[-1].input_feature_names_

Other mentioned alternative: `feature_names_in_` and `feature_names_out_`.


Do we also want a ``output_feature_names_`` attribute?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to setting the ``input_feature_names_`` attribute on each estimator,
we could also have an ``output_feature_names_``.

This would return the same as the current ``get_feature_names()`` method,
potentially removing the need to have an explicit output feature names *getter
method*. The "update feature names" method would then mainly be used for
setting the input features and making sure they get propagated. 


Should all estimators call the "update feature names" method inside ``fit()`` ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the current implementation, the ``Pipeline`` is responsible to catch the input
feature names of the data and calling the "update feature names" method with
those names at the end of ``fit()``.

However, that means that if you use a transformer/estimator in itself and not
in a Pipeline, we won't have the feature names automatically set. For example
(assuming ``X_df`` is a DataFrame with columns A and B)::

    >>> ohe = OneHotEncoder()
    >>> ohe.fit(X_df)
    >>> ohe.input_feature_names_
    AttributeError: ...
    >>> ohe.get_feature_names()
    ['x0_cat1', 'x0_cat2', 'x1_cat1', 'x2_cat2']

vs

::

    >>> ohe_pipe = Pipeline([('ohe', OneHotEncoder())])
    >>> ohe_pipe.fit(X_df)
    >>> ohe.input_feature_names_
    ['A', 'B']
    >>> ohe.get_feature_names()
    ['A_cat1', 'A_cat2', 'B_cat1', 'B_cat2']


Currently, the ``input_feature_names_`` attribute of an estimator is set by the
"update feature names" method of the parent estimator (e.g. the Pipeline from
which the estimator is called). But, this logic could also be moved into the
"update feature names" method of the estimator itself. 

In that case, the ``fit`` method of the estimator could also call the "update
feature names" method at the end, ensuring consistency between on-itself
standing estimators and Pipelines.  However, the clear downside of this
consistency is that this would add one line to each ``fit`` method throughout
scikit-learn.


What happens if one part of the pipeline does not implement "update feature names"?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of raising an error, the current PR sets the output feature names to
``None``, which if passed to the next step of the pipeline, allows it to still
generate feature names.

What should the "update feature names" method do in the less obvious cases?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The clear cases on how to transform input to output features are:

- Transformers that pass through (One-to-one), e.g. StandardScaler
- Transformers that generate new features, e.g. OneHotEncoder, Vectorizers
- Transformers that output a subset of the original features (SelectKBest?)

But, what to do with:

- Transformers that create linear combinations, eg PCA 
- Transformers based on arbitrary functions


Should all estimators (so including regressors, classifiers, ...) have a "update feature names" method?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the current implementation, only transformers (and Pipeline and
meta-estimators, which could act as transformer) have a "update feature names"
method.

For consistency, we could also add them to *all* estimators.

For a regressor or classifier, the method could set the ``input_feature_names_``
attribute and return ``None``.


How should feature names be transformed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this question, there is a separate SLEP:
https://github.com/scikit-learn/enhancement_proposals/pull/17


Interaction with column validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another, potentially related, change that has been discussed is to do input
validation on transform/predict time: ensuring that the column names and order
is identical when transforming/predicting compared to fit (currently,
scikit-learn silently returns "incorrect" results as long as the number of
columns matches).

To do proper validation, the idea would be to store the column names at fit
time, so they can be compared at transform/predict time. Those stored column
names could be very similar to the ``input_feature_names_`` described in this
SLEP. 

However, if a user calls ``Pipeline.get_feature_names(input_features=[...])``
with a set of custom input feature names that are not identical to the original
DataFrame column names, the stored column names to do validation and the stored
column names to propagate the feature names would get out of sync. Or should
calling ``get_feature_names`` also affect future validation in a ``predict()``
call?

One solution is to disallow setting feature names if the original input are
pandas DataFrames (so ``pipe.get_feature_names(['other', 'names'])`` would
raise an error if ``pipe`` was fitted with a DataFrame). This would prevent
ending up in potentially confusing or ambiguous situations. Calling
``get_feature_names`` with custom input names is of course still possible when
the input was not a pandas DataFrame.


Backward compatibility
----------------------

This SLEP does not affect backward compatibility, as all described attributes
and methods would be new ones, not affecting existing ones.

The only possible compatibility question is, if we decide to use another name
than ``get_feature_names()``, what to do with those existing methods? Those
could in principle be deprecated.


Alternatives
------------

The alternatives described here are alternatives to the combination of the
transformative "update feature names" method and calling it recursively in the
Pipeline setting the ``input_feature_names_`` attribute on each step.

1. Only implement the "update feature names" method and require the user to
   slice the pipeline to call this method manually on the appropriate subset.
   For example::

       >>> clf[:-1].get_feature_names(input_features=[....])

   would then propagate the original provided names up to the output names of
   the final step of this sliced pipeline (which will be the input feature
   names for the last step of the pipeline, in this example).
  
   This is what was implemented in https://github.com/scikit-learn/scikit-learn/pull/12627. 
  
   The main drawback of this more limited proposal is the user interface: the
   user needs to manually slice the pipeline and call ``get_feature_names()``
   to get the output feature names of this subset, in order to get the input
   feature names of the final classifier/regressor.
  
   The main difference is not automatically calling this method in the Pipeline
   ``fit()`` method and storing the `input_feature_names_` attributes.

2. Use "pandas in - pandas out" everywhere (also fitted attributes): user does
   not need an explicit way to get or set the feature names as they are
   included in the output of estimators (e.g. ``coef_`` would be a Series
   with the input feature names to the final estimator as the index).

   However, this would tie this feature much more to pandas (and eg would not
   be available when working with numpy arrays) and would be much more evasive
   for the codebase (and raise a lot more general issues about tying to pandas).

3. Implement a more comprehensive feature description language.

4. Leave it to the user.


While we think that alternatives 2) and 3) are valid option for the future,
trying to implement this now will probably result in a gridlock and/or take too
much time. The solution proposed in this SLEP can provide something that solves
the majority of the use cases relatively easy. We can create a more elaborate
solution later, in particular since this SLEP doesn't introduce any concepts
that are not in scikit-learn already. 

We don't think that doing nothing (4) is a good option. The titanic example
shown in the introduction is valid use case, and currently, getting the
feature names is very hard (the example was even simplified, as not taking into
account features being dropped  if all NaN).


Discussion
----------

Discussions have been held at several places:

- https://github.com/scikit-learn/scikit-learn/issues/6424
- https://github.com/scikit-learn/scikit-learn/issues/6425
- https://github.com/scikit-learn/scikit-learn/pull/12627
- https://github.com/scikit-learn/scikit-learn/pull/13307


References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
