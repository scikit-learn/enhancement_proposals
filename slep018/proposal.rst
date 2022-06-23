.. _slep_018:

=======================================================
SLEP018: Pandas Output for Transformers with set_output
=======================================================

:Author: Thomas J. Fan
:Status: Draft
:Type: Standards Track
:Created: 2022-06-22

Abstract
--------

This SLEP proposes a ``set_output`` method to configure the output container of
scikit-learn transformers.

Detailed description
--------------------

Currently, scikit-learn transformers return NumPy ndarrays or SciPy sparse
matrices. This SLEP proposes adding a ``set_output`` method to configure a
transformer to output pandas DataFrames::

   scalar = StandardScalar().set_output(transform="pandas")
   scalar.fit(X_df)

   # X_trans_df is a pandas DataFrame
   X_trans_df = scalar.transform(X_df)

The index of the output DataFrame must match the index of the input. If the
transformer does not support ``transform="pandas"``, then it must raise a
``ValueError`` stating that it does not support the feature.

For this SLEP, ``set_output`` will only configure the output for dense data.  If
the transformer returns sparse data, then ``transform`` will raise a
``ValueError`` if ``set_output(transform="pandas")``.

For a pipeline, calling ``set_output`` on the pipeline will configure all steps
in the pipeline::

   num_prep = make_pipeline(SimpleImputer(), StandardScalar(), PCA())
   num_preprocessor.set_output(transform="pandas")

   # X_trans_df is a pandas DataFrame
   X_trans_df = num_preprocessor.fit_transform(X_df)

Meta-estimators that support ``set_output`` are required to configure all inner
transformer by calling ``set_output``. If an inner transformer does not define
``set_output``, then an error is raised.

Global Configuration
....................

This SLEP proposes a global configuration flag that sets the output for all
transformers::

   import sklearn
   sklearn.set_config(transform_output="pandas")

The global default configuration is ``"default"`` where the transformer
determines the output container.

Implementation
--------------

The implementation of this SLEP is in :pr:`23734`.

Backward compatibility
----------------------

There are no backward compatibility concerns, because the ``set_output`` method
is a new API. Third party transformers can opt-in to the API by defining
``set_output``.

Alternatives
------------

Alternatives to this SLEP includes:

1. `SLEP014 <https://github.com/scikit-learn/enhancement_proposals/pull/37>`__
   proposes that if the input is a DataFrame than the output is a DataFrame.
2. :ref:`SLEP012 <slep_012>` proposes a custom scikit-learn container for dense
   and sparse data that contains feature names. This SLEP also proposes a custom
   container for sparse data, but pandas for dense data.
3. Prototype `#20100
   <https://github.com/scikit-learn/scikit-learn/pull/20100>`__ showcases
   ``array_out="pandas"`` in `transform`. This API is limited because does not
   directly support fitting on a pipeline where the steps requires data frames
   input.

Discussion
----------

A list of issues discussing Pandas output are: `#14315
<https://github.com/scikit-learn/scikit-learn/pull/14315>`__, `#20100
<https://github.com/scikit-learn/scikit-learn/pull/20100>`__, and `#23001
<https://github.com/scikit-learn/scikit-learn/issueas/23001>`__.

Future Extensions
-----------------

Sparse Data
...........

The Pandas DataFrame is not suitable to provide column names because it has
performance issues as shown in `#16772
<https://github.com/scikit-learn/scikit-learn/pull/16772#issuecomment-615423097>`__.
A future extension to this SLEP is to have a ``"pandas_or_namedsparse"`` option.
This option will use a scikit-learn specific sparse container that subclasses
SciPy's sparse matrices. This sparse container includes the sparse data, feature
names and index. This enables pipelines with Vectorizers without performance
issues::

   pipe = make_pipeline(
      CountVectorizer(),
      TfidfTransformer(),
      LogisticRegression(solver="liblinear")
   )
   pipe.set_output(transform="pandas_or_namedsparse")

   # feature names for logistic regression
   pipe[-1].feature_names_in_

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open Publication
   License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
