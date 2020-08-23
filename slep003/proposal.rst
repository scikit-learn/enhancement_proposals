.. _slep_003:

===============================================
SLEP003: Consistent inspection for transformers
===============================================

. topic:: **Summary**

    Inspect transformers' output shape and dependence on input features
    consistently with
    ``get_feature_dependence() -> boolean (n_outputs, n_inputs)``

.. contents:: Table of contents
   :depth: 2

Goals
=====

* Make it easy to describe features output from transformation pipelines in
  terms of input features.
* Enable describing (e.g. with feature names) only a subset of output features.
* Enable inspecting which input features are used or important in the model,
  such that models can be compressed.

Design
======

Transformers will provide a method of the following description.

::

    def get_feature_dependence(self):
        """Identify dependencies between input and output features

        Returns
        =======
        D : array or CSR matrix, shape [n_output_features, n_input_features]
            If D[i, j] == 0 then input feature j does not contribute to output
            feature i. Otherwise, feature i may depend on feature j.

            This should not be mutated by the user.
        """

Thus:

* it is easy to inspect a transformer to see how many output features it
  produces.
* by calling this method on the consitutent transformers of a ``FeatureUnion``
  can determine from the shapes which features derive from which transformer.
* it is easy to inspect a transformer to see how many input features it
  requires, making it easy to invent default feature names for transformation.
* models can be compressed by first identifying features unused among the most
  informative features for a downstream classifier
* determining and storing all feature names/descriptions within a pipeline may
  be expensive. By tracing dependencies, feature descriptions can be
  calculated for sparsely many features, perhaps via a method
  ``describe_features(sparse_input_feature_names, indices)``

Examples
========

The following are rough implementations for some existing transformers::

    class StandardScaler:
        def get_feature_dependence(self):
            # A diagonal matrix
            n_features = len(self.scale_)
            # csr_matrix constructed with (data, indices, indptr)
            return csr_matrix((np.ones(n_features), np.arange(n_features),
                               np.arange(n_features + 1)))

    class PCA:
        def get_feature_dependence(self):
            return self.components_

    class SelectorMixin:
        def get_feature_dependence(self):
            # One nonzero element per row
            mask = self._get_support_mask()
            idx = np.flatnonzero(mask)
            return csr_matrix((np.ones(idx.shape), idx, np.arange(len(idx) + 1)),
                              shape=(len(mask), len(idx)))

    class Pipeline:
        def get_feature_dependence(self):
            # Dot product of constituent dependency matrices
            D = None
            for name, trans in self.steps:
                if trans is not None:
                    if D is None:
                        D = trans.get_feature_dependence()
                    else:
                        D = safe_sparse_dot(trans.get_feature_dependence(), D)
            return D

    class FeatureUnion:
        def get_feature_dependence(self):
            # Concatenation of constituent dependency matrices
            Ds = [trans.get_feature_dependence()
                  for _, trans in self.transformer_list
                  if trans is not None]
            if any(issparse(D) for D in Ds):
                Ds = [csr_matrix(D) for D in Ds]
                return sp.hstack(Ds)
            return np.hstack(Ds)

Problem cases
=============

1d input / output
-----------------

The input to a transformer may be a 1-dimensional array-like. This is often the
case for feature extractors, which may take a list of dicts, a list of strings
or a list of files, for instance. In this case, ``get_feature_dependence``
should spoof the existence of a single input feature, returning a matrix of
shape ``(1, n_output_features)``.

While not included in scikit-learn repository, transformers may translate one
1-d array (or Series) into another 1-d array.  It would be appropriate in this
context for ``get_feature_dependence`` to return ``array([[1]])``.

Pandas DataFrame input
----------------------

The input features should correspond to columns in the case that a
transformer is designed to take a Pandas DataFrame as input.

Constituent transformers lack this feature
------------------------------------------

Where ``Pipeline`` or ``FeatureUnion`` has a constituent transformer that
lacks this method, calling ``hasattr(pipeline, 'get_feature_dependence')``
should similarly return False.  This can be implemented using a ``property``.

Alternatives
============

An attribute
------------

An attribute ``feature_dependence_`` could be used instead of a method, but
for the following issues:

1. ``feature_dependence_`` cannot be constructed statically in ``Pipeline`` and
   ``FeatureUnion`` in case some constituent transformers. These could be
   implemented dynamically with a ``property`` and raise an error when .
2. Often one would want to calculate the feature dependence matrix for all
   steps of a ``Pipeline`` excluding the last.  This entails a dynamic approach
   to calculating the dependencies.
3. An attribute will in some cases be redundant relative to existing attributes,
   such as ``RFE.support_``

The main advantage of an attribute is that it may encourage the information to
be stored at fit time avoiding recalculation. However this can be done when
necessary with a method. An attribute may or may not have greater visibility to
users.


Allow other sparse formats
--------------------------

I have suggested consistently using CSR so that it is efficient to perform
matrix products as well as to look up active input features given selected
output feature (a standard model inspection task).

DIA format may be more efficient in some cases, taking half the memory and
allowing for more efficient matrix products and lookup relative to CSR.
However the API assurances of a single format seem to outweigh DIA's benefits.

Require binary values
---------------------

Not binarising the output has the benefit of not copying in some cases.
It does, howeve, risk numerical over/underflow in matrix multiplication.

Transposition
-------------

Return shape could be ``(n_input, n_output)``, which some users may find more
intuitive.  The current proposal has the following advantages:

* consistency with notion of dependence: matrix maps first axis to
  dependencies in second.
* consistency with ``PCA.components_``
* main purpose is model inspection, hence lookup by row is common in the
  current proposal.
* transposing the shape would imply using CSC for the same efficiencies, which
  is less commonly used than CSR throghout scikit-learn.
