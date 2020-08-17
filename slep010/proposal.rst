.. _slep_010:

=====================================
SLEP010: ``n_features_in_`` attribute
=====================================

:Author: Nicolas Hug
:Status: Accepted
:Type: Standards Track
:Created: 2019-11-23

Abstract
########

This SLEP proposes the introduction of a public ``n_features_in_`` attribute
for most estimators (where relevant).

Motivation
##########

Knowing the number of features that an estimator expects is useful for
inspection purposes. This is also useful for implementing the feature names
propagation (`SLEP 8
<https://github.com/scikit-learn/enhancement_proposals/pull/18>`_) . For
example any of the scaler can easily create feature names if they know
``n_features_in_``.

Solution
########

The proposed solution is to replace most calls to ``check_array()`` or
``check_X_y()`` by calls to a newly created private method::

    def _validate_data(self, X, y=None, reset=True, **check_array_params)
        ...

The ``_validate_data()`` method will call ``check_array()`` or
``check_X_y()`` function depending on the ``y`` parameter.

If the ``reset`` parameter is True (default), the method will set the
``n_feature_in_`` attribute of the estimator, regardless of its potential
previous value. This should typically be used in ``fit()``, or in the first
``partial_fit()`` call. Passing ``reset=False`` will not set the attribute but
instead check against it, and potentially raise an error. This should typically
be used in ``predict()`` or ``transform()``, or on subsequent calls to
``partial_fit``.

In most cases, the ``n_features_in_`` attribute exists only once ``fit`` has
been called, but there are exceptions (see below).

A new common check is added: it makes sure that for most estimators, the
``n_features_in_`` attribute does not exist until ``fit`` is called, and
that its value is correct. Instead of raising an exception, this check will
raise a warning for the next two releases. This will give downstream
packages some time to adjust (see considerations below).

Since the introduced method is private, third party libraries are
recommended not to rely on it.

The logic that is proposed here (calling a stateful method instead of a
stateless function) is a pre-requisite to fixing the dataframe column
ordering issue: with a stateless ``check_array``, there is no way to raise
an error if the column ordering of a dataframe was changed between ``fit``
and ``predict``. This is however out os scope for this SLEP, which only focuses
on the introduction of the ``n_features_in_`` attribute.

Considerations
##############

The main consideration is that the addition of the common test means that
existing estimators in downstream libraries will not pass our test suite,
unless the estimators also have the ``n_features_in_`` attribute.

The newly introduced checks will only raise a warning instead of an exception
for the next 2 releases, so this will give more time for downstream packages
to adjust.

There are other minor considerations:

- In most meta-estimators, the input validation is handled by the
  sub-estimator(s). The ``n_features_in_`` attribute of the meta-estimator
  is thus explicitly set to that of the sub-estimator, either via a
  ``@property``, or directly in ``fit()``.
- Some estimators like the dummy estimators do not validate the input
  (the 'no_validation' tag should be True). The ``n_features_in_`` attribute
  should be set to None, though this is not enforced in the common check.
- Some estimators expect a non-rectangular input: the vectorizers. These
  estimators expect dicts or lists, not a ``n_samples * n_features`` matrix.
  ``n_features_in_`` makes no sense here and these estimators just don't have
  the attribute.
- Some estimators may know the number of input features before ``fit`` is
  called: typically the ``SparseCoder``, where ``n_feature_in_`` is known at
  ``__init__`` from the ``dictionary`` parameter. In this case the attribute
  is a property and is available right after object instantiation.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
