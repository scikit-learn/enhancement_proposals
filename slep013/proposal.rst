.. _slep_013:

======================================
SLEP013: ``n_features_out_`` attribute
======================================

:Author: Adrin Jalali
:Status: Under Review
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


