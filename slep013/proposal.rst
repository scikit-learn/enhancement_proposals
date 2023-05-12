.. _slep_013:

======================================
SLEP013: ``n_features_out_`` attribute
======================================

:Author: Adrin Jalali
:Status: Rejected
:Type: Standards Track
:Created: 2020-02-12

Abstract
########

This SLEP proposes the introduction of a public ``n_features_out_`` attribute
for most transformers (where relevant).

Motivation
##########

Knowing the number of features that a transformer outputs is useful for
inspection purposes. This is in conjunction with `*SLEP010: ``n_features_in_``*
<https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html>`_.

Take the following piece as an example::

    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    # We will train our classifier with the following features:
    # Numeric Features:
    # - age: float.
    # - fare: float.
    # Categorical Features:
    # - embarked: categories encoded as strings {'C', 'S', 'Q'}.
    # - sex: categories encoded as strings {'female', 'male'}.
    # - pclass: ordinal integers {1, 2, 3}.

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_features = ['age', 'fare']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['embarked', 'sex', 'pclass']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression())])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf.fit(X_train, y_train)

The user could then inspect the number of features going out from each step::

    # Total number of output features from the `ColumnTransformer`
    clf[0].n_features_out_

    # Number of features as a result of the numerical pipeline:
    clf[0].named_transformers_['num'].n_features_out_

    # Number of features as a result of the categorical pipeline:
    clf[0].named_transformers_['cat'].n_features_out_

Solution
########

The proposed solution is for the ``n_features_out_`` attribute to be set once a
call to ``fit`` is done. In many cases the value of ``n_features_out_`` is the
same as some other attribute stored in the transformer, *e.g.*
``n_components_``, and in these cases a ``Mixin`` such as a ``ComponentsMixin``
can delegate ``n_features_out_`` to those attributes.

Testing
-------

A test to the common tests is added to ensure the presence of the attribute or
property after calling ``fit``.

Considerations
##############

The main consideration is that the addition of the common test means that
existing estimators in downstream libraries will not pass our test suite,
unless the estimators also have the ``n_features_out_`` attribute.

The newly introduced checks will only raise a warning instead of an exception
for the next 2 releases, so this will give more time for downstream packages
to adjust.

There are other minor considerations:

- In some meta-estimators, this is delegated to the
  sub-estimator(s). The ``n_features_out_`` attribute of the meta-estimator is
  thus explicitly set to that of the sub-estimator, either via a ``@property``,
  or directly in ``fit()``.
- Some transformers such as ``FunctionTransformer`` may not know the number
  of output features since arbitrary arrays can be passed to `transform`. In
  such cases ``n_features_out_`` is set to ``None``.

Copyright
---------

This document has been placed in the public domain. [1]_

References and Footnotes
------------------------

.. [1] _Open Publication License: https://www.opencontent.org/openpub/


