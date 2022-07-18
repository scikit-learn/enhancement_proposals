.. _slep_018:

=======================================================
SLEP018: Pandas Output for Transformers with set_output
=======================================================

:Author: Thomas J. Fan
:Status: Accepted
:Type: Standards Track
:Created: 2022-06-22

Abstract
--------

This SLEP proposes a ``set_output`` method to configure the output data container of
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

This SLEP's only focus is dense data for ``set_output``. If a transformer returns
sparse data, e.g. `OneHotEncoder(sparse=True), then ``transform`` will raise a
``ValueError`` if ``set_output(transform="pandas")``. Dealing with sparse output
might be the scope of another future SLEP.

For a pipeline, calling ``set_output`` will configure all inner transformers::

   num_preprocessor = make_pipeline(SimpleImputer(), StandardScalar(), PCA())
   num_preprocessor.set_output(transform="pandas")

   # X_trans_df is a pandas DataFrame
   X_trans_df = num_preprocessor.fit_transform(X_df)

   # X_trans_df is again a pandas DataFrame
   X_trans_df = num_preprocessor[0].transform(X_df)

``set_output`` on a pipeline only configures transformers and does not configure
non-transformers. This enables the following workflow::

   log_reg = make_pipeline(SimpleImputer(), StandardScalar(), LogisticRegression())
   log_reg.set_output(transform="pandas")

   # All transformers return DataFrames during fit
   log_reg.fit(X_f, y)

   # The final step contains the feature names in
   log_reg[-1].feature_names_in_

Meta-estimators that support ``set_output`` are required to configure all inner
transformer by calling ``set_output``. Specifically all fitted and non-fitted
inner transformers must be configured with ``set_output``. This enables
``transform``'s output to be a DataFrame before and after the meta-estimator is
fitted. If an inner transformer does not define ``set_output``, then an error is
raised.


Global Configuration
....................

For ease of use, this SLEP proposes a global configuration flag that sets the output for all
transformers::

   import sklearn
   sklearn.set_config(transform_output="pandas")

The global default configuration is ``"default"`` where the transformer
determines the output container.

The configuration can also be set locally using the ``config_context`` context
manager::

   from sklearn import config_context
   with config_context(transform_output="pandas"):
      num_prep = make_pipeline(SimpleImputer(), StandardScalar(), PCA())
      num_preprocessor.fit_transform(X_df)

The following specifies the precedence levels for the three ways to configure
the output container:

1. Locally configure a transformer: ``transformer.set_output``
2. Context manager: ``config_context``
3. Global configuration: ``set_config``

Implementation
--------------

A possible implementation of this SLEP is worked out in :pr:`23734`.

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
2. Prototype `#20100
   <https://github.com/scikit-learn/scikit-learn/pull/20100>`__ showcases
   ``array_out="pandas"`` in `transform`. This API is limited because does not
   directly support fitting on a pipeline where the steps requires data frames
   input.

Discussion
----------

A list of issues discussing Pandas output are: `#14315
<https://github.com/scikit-learn/scikit-learn/pull/14315>`__, `#20100
<https://github.com/scikit-learn/scikit-learn/pull/20100>`__, and `#23001
<https://github.com/scikit-learn/scikit-learn/issueas/23001>`__. This SLEP
proposes configuring the output to be pandas because it is the DataFrame library
that is most widely used and requested by users. The ``set_output`` can be
extended to support support additional DataFrame libraries in the future.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open Publication
   License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
