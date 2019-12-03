 .. _slep_007:

=============
Feature Names
=============

This proposal talks about how feature names are generated and not how they are
propagated.

Scope
-----

All estimators implement a ``feature_names_in_`` API, and any estimator with
a ``transform`` method implements a ``feature_names_out_`` API.

Input Feature Names
-------------------

The input feature names are stored in a fitted estimator in a
``feature_names_in_`` attribute. The feature names are taken from the given
data during fit. It is taken either from a ``pandas`` data frame or our own
data structure (something on top of a ``numpy`` array, ``NamedArray`` for
instance). The attribute will be ``None`` if the given data does not include
any feature names.

Output Feature Names
--------------------

A fitted estimator exposes the output feature names through the
``feature_names_out_`` attribute. At the same time, the data structure returned
by a ``transform`` method also includes the generated feature names.

Same Features Transformers
**************************

From the perspective of feature names, the simplest transformers are the ones
which have the same feature names in the output as in the input.

- Input provides feature names: ``feature_names_in_`` and
  ``feature_names_out_`` are the same.
- Input provides no feature names: ``feature_names_out_`` will be ``x0`` to
  ``xn``, where ``n`` is the number of features.

Feature Selector Transformers
*****************************

The output feature names are the ones selected from the input, and if no
feature names are provided, ``x0`` to ``xn`` are assumed to be their names. For
example, if the transformer selects the first and the third features, and no
names are provided, the ``feature_names_out_`` will be ``[x0, x2]``.

Feature Generating Transformers
*******************************

The transformers which generate features from the given set of features,
generate their own feature names, which has the form of ``name0`` to ``namen``,
where ``name`` represents the transformer. For instance, a ``PCA`` transformer
will output ``[pca0, ..., pcan]``, ``n`` being the number of PCA components.

Meta-Estimators
***************

Meta estimators can choose to prefix the output feature names or not.

By default, ``Pipeline`` adds no prefix, *i.e* its ``feature_names_out_`` is
the same as the ``feature_names_out_`` of the last step, and ``None`` if the
last step is not a transformer.

``ColumnTransformer`` by default adds a prefix to the output feature names,
indicating the name of the step applied on them.


Backward Compatibility
----------------------

All estimators should implement the ``feature_names_in_`` and
``feature_names_out_`` API. This is checked in ``check_estimator``, and the
transition is done with a ``FutureWarning`` to give time to third party
developers to implement the API.

Notes
-----

This SLEP also applies to resamplers the same way as transformers.
