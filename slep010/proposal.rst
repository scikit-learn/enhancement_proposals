.. _slep_010:

=================================
SLEP010: n_features_in_ attribute
=================================

:Author: Nicolas Hug
:Status: Under review
:Type: Standards Track
:Created: 2019-11-23

Abstract
########

This SLEP proposes the introduction of a public ``n_features_in_`` attribute
for most estimators (where relevant). This attribute is automatically set
when calling ``_validate_X()`` or ``_validate_X_y`` which are meant to replace
``check_array`` and ``check_X_y`` (which are still called under the hood).

Motivation
##########

Knowing the number of features that an estimator expects is useful for
inspection purposes, as well as for input validation.

Solution
########

The proposed solution is to replace most calls to ``check_array()`` or
``check_X_y()`` by calls to two newly created private methods::

    def _validate_X(self, X, check_n_features=False, **check_array_params)
        ...

    def _validate_X_y(self, X, check_n_features=False, **check_X_y_params)
        ...

The ``_validate_XXX()`` methods will call the corresponding ``check_XXX()``
functions.

The ``check_n_features`` parameter is False by default and can be set to True
to raise an error when ``self.n_features_in_ != X.shape[1]``. The idea is to
leave it to False in ``fit()`` and set it to True in ``predict()`` or
``transform()``.

In most cases, the attribute exists only once ``fit`` has been called, but
there are exceptions (see below).

A new common check is added: it makes sure that for most esitmators, the
``n_features_in_`` attribute does not exist until ``fit`` is called, and
that its value is correct.

The logic that is proposed here (calling a stateful method instead of a
stateless function) is a pre-requisit to fixing the dataframe column
ordering issue: at the moment, there is no way to raise an error if the
column ordering of a dataframe was changed between ``fit`` and ``predict``.

Considerations
##############

The main consideration is that the addition of the common test means that
existing estimators in downstream libraries will not pass our test suite,
unless the estimators also have the `n_features_in_` attribute (which can be
done by updating calls to ``check_XXX`` into calls to ``_validate_XXX``).

There are other minor considerations:

- In most meta-estimators, the input validation is handled by the
  sub-estimator(s). The ``n_features_in_`` attribute of the meta-estimator
  is thus explicitly set to that of the sub-estimator, either via a
  ``@property``, or directly in ``fit()``.
- Some estimators like the dummy estimators do not validate the input
  (the 'no_validation' tag should be True). The ``n_features_in_`` attribute
  should be set to None, though this is not enforced in the common tests.
- Some estimators expect a non-rectangular input: the vectorizers. These
  estimators never have a ``n_features_in_`` attribute (they never call
  ``check_array`` anyway).
- Some estimators may know the number of input features before ``fit`` is
  called: typically the ``SparseCoder``, where ``n_feature_in_`` is known at
  ``__init__`` from the ``dictionary`` parameter. In this case the attribute is
  set in ``__init__``.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
