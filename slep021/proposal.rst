.. _slep_021:

==================================================
SLEP021: Unified API to compute feature importance
==================================================

:Author: Thomas J Fan, Guillaume Lemaitre
:Status: Draft
:Type: Standards Track
:Created: 2023-03-09

Abstract
--------

This SLEP proposes a common API for computing feature importance.

Detailed description
--------------------

Motivation
~~~~~~~~~~

Data scientists rely on feature importance when inspecting a trained model.
Feature importance is a measure of how much a feature contributes to the
prediction and thus gives insights on the model the predictions provided by
the model.

However, there is currently not a single method to compute feature importance.
All available methods are designed upon axioms or hypotheses that are not
necessarly respected in practice.

Some work in scikit-learn has been done to provide documentation to highlight
the limitations of some implemented methods. However, there is currently not
a common way to expose feature importance in scikit-learn. In addition, for
some historical reasons, some estimators (e.g. decision tree) provide a single
feature importance that could be used as the "method-to-use" to analyse the
model. It is problematic since there is not defacto standard to analyse the
feature importance of a model.

Therefore, this SLEP proposes an API for providing feature importance allowing
to be flexible to switch between methods and extensible to add new methods. It
is a follow-up of initial discussions from [2]_.

Current state
~~~~~~~~~~~~~

Current pitfalls
~~~~~~~~~~~~~~~~

Solution
~~~~~~~~

Discussion
----------

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open Publication
   License`_.
.. [2] https://github.com/scikit-learn/scikit-learn/issues/20059#issuecomment-869811256

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain [1]_.
