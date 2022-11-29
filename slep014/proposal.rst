.. _slep_014:

==============================
SLEP014: Pandas In, Pandas Out
==============================

:Author: Thomas J Fan
:Status: Rejected
:Type: Standards Track
:Created: 2020-02-18

Abstract
########

This SLEP proposes using pandas DataFrames for propagating feature names
through ``scikit-learn`` transformers.

Motivation
##########

``scikit-learn`` is commonly used as a part of a larger data processing
pipeline. When this pipeline is used to transform data, the result is a
NumPy array, discarding column names. The current workflow for
extracting the feature names requires calling ``get_feature_names`` on the
transformer that created the feature. This interface can be cumbersome when used
together with a pipeline with multiple column names::

    import pandas as pd
    import numpy as np
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame({'letter': ['a', 'b', 'c'],
                      'pet': ['dog', 'snake', 'dog'],
                      'num': [1, 2, 3]})
    y = [0, 0, 1]
    orig_cat_cols, orig_num_cols = ['letter', 'pet'], ['num']

    ct = make_column_transformer(
        (OneHotEncoder(), orig_cat_cols), (StandardScaler(), orig_num_cols))
    pipe = make_pipeline(ct, LogisticRegression()).fit(X, y)

    cat_names = (pipe['columntransformer']
                 .named_transformers_['onehotencoder']
                 .get_feature_names(orig_cat_cols))

    feature_names = np.r_[cat_names, orig_num_cols]

The ``feature_names`` extracted above corresponds to the features directly
passed into ``LogisticRegression``. As demonstrated above, the process of
extracting ``feature_names`` requires knowing the order of the selected
categories in the ``ColumnTransformer``. Furthemore, if there is feature
selection in the pipeline, such as ``SelectKBest``, the ``get_support`` method
would need to be used to determine column names that were selected.

Solution
########

The pandas DataFrame has been widely adopted by the Python Data ecosystem to
store data with feature names. This SLEP proposes using a DataFrame to
track the feature names as the data is transformed. With this feature, the
API for extracting feature names would be::

    from sklearn import set_config
    set_config(pandas_in_out=True)

    pipe.fit(X, y)
    X_trans = pipe[:-1].transform(X)

    X_trans.columns.tolist()
    ['letter_a', 'letter_b', 'letter_c', 'pet_dog', 'pet_snake', 'num']

This SLEP proposes attaching feature names to the output of ``transform``. In
the above example, ``pipe[:-1].transform(X)`` propagates the feature names
through the multiple transformers.

This feature is only available through a soft dependency on pandas. Furthermore,
it will be opt-in with the the configuration flag: ``pandas_in_out``. By
default, ``pandas_in_out`` is set to ``False``, resulting in the output of all
estimators to be a ndarray.

Enabling Functionality
######################

The following enhancements are **not** a part of this SLEP. These features are
made possible if this SLEP gets accepted.

1. Allows estimators to treat columns differently based on name or dtype. For
   example, the categorical dtype is useful for tree building algorithms.

2. Storing feature names inside estimators for model inspection::

    from sklearn import set_config
    set_config(store_feature_names_in=True)

    pipe.fit(X, y)

    pipe['logisticregression'].feature_names_in_

3. Allow for extracting the feature names of estimators in meta-estimators::

    from sklearn import set_config
    set_config(store_feature_names_in=True)

    est = BaggingClassifier(LogisticRegression())
    est.fit(X, y)

    # Gets the feature names used by an estimator in the ensemble
    est.estimators_[0].feature_names_in_

For options 2 and 3 the default value of configuration flag:
``store_feature_names_in`` is False.

Considerations
##############

Memory copies
-------------

As noted in `pandas #27211 <https://github.com/pandas-dev/pandas/issues/27211>`_,
there is not a guarantee that there is a zero-copy round-trip going from numpy
to a DataFrame. In other words, the following may lead to a memory copy in
a future version of ``pandas``::

    X = np.array(...)
    X_df = pd.DataFrame(X)
    X_again = np.asarray(X_df)

This is an issue for ``scikit-learn`` when estimators are placed into a
pipeline. For example, consider the following pipeline::

    set_config(pandas_in_out=True)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X, y)

Interally, ``StandardScaler.fit_transform`` will operate on a ndarray and
wrap the ndarray into a DataFrame as a return value. This is will be
piped into ``LogisticRegression.fit`` which calls ``check_array`` on the
DataFrame, which may lead to a memory copy in a future version of
``pandas``. This leads to unnecessary overhead from piping the data from one
estimator to another.

Sparse matrices
---------------

Traditionally, ``scikit-learn`` prefers to process sparse matrices in
the compressed sparse row (CSR) matrix format. The `sparse data structure <https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html>`_ in pandas 1.0 only supports converting directly to
the coordinate format (COO). Although this format was designed to quickly
convert to CSR or CSC formats, the conversion process still needs to allocate
more memory to store. This can be an issue with transformers such as the
``OneHotEncoder.transform`` which has been optimized to construct a CSR matrix.

Backward compatibility
######################

The ``set_config(pandas_in_out=True)`` global configuration flag will be set to
``False`` by default to ensure backward compatibility. When this flag is False,
the output of all estimators will be a ndarray.

Community Adoption
##################

With the new ``pandas_in_out`` configuration flag, third party libraries may
need to query the configuration flag to be fully compliant with this SLEP.
Specifically, "to be fully compliant" entails the following policy:

1. If ``pandas_in_out=False``, then ``transform`` always returns numpy array.
2. If ``pandas_in_out=True``, then ``transform`` returns a DataFrame if the
   input is a Dataframe.

This policy can either be enforced with ``check_estimator`` or not:

- **Enforce**: This increases the maintaince burden of third party libraries.
  This burden includes: checking for the configuration flag, generating feature names and including pandas as a dependency to their library.

- **Not Enforce**: Currently, third party transformers can return a DataFrame
  or a numpy and this is mostly compatible with ``scikit-learn``. Users with
  third party transformers would not be able to access the features enabled
  by this SLEP.


Alternatives
############

This section lists alternative data structures that can be used with their
advantages and disadvantages when compared to a pandas DataFrame.

InputArray
----------

The proposed ``InputArray`` described
:ref:`SLEP012 Custom InputArray Data Structure <slep_012>` introduces a new
data structure for homogenous data.

Pros
~~~~

- A thin wrapper around a numpy array or a sparse matrix with a minimial feature
  set that ``scikit-learn`` can evolve independently.

Cons
~~~~

- Introduces another data structure for data storage in the PyData ecosystem.
- Currently, the design only allows for homogenous data.
- Increases maintenance responsibilities for ``scikit-learn``.

XArray Dataset
--------------

`xarray's Dataset <http://xarray.pydata.org/en/stable/data-structures.html#dataset>`_
is a multi-dimenstional version of panda's DataFrame.

Pros
~~~~

- Can be used for heterogeneous data.

Cons
~~~~

- ``scikit-learn`` does not require many of the features Dataset provides.
- Needs to be converted to a DataArray before it can be converted to a numpy array.
- The `conversion from a pandas DataFrame to a Dataset <http://xarray.pydata.org/en/stable/pandas.html>`_
  is not lossless. For example, categorical dtypes in a pandas dataframe will
  lose their categorical information when converted to a Dataset.
- xarray does not have as much adoption as pandas, which increases the learning
  curve for using Dataset with ``scikit-learn``.

XArray DataArray
----------------

`xarray's DataArray <http://xarray.pydata.org/en/stable/data-structures.html#dataarray>`_
is a data structure that store homogenous data.

Pros
~~~~

- xarray guarantees that there will be no copies during round-trips from
  numpy. (`xarray #3077 <https://github.com/pydata/xarray/issues/3077>`_)

Cons
~~~~

- Can only be used for homogenous data.
- As with XArray's Dataset, DataArray does not as much adoption as pandas,
  which increases the learning curve for using DataArray with ``scikit-learn``.

References and Footnotes
########################

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
#########

This document has been placed in the public domain. [1]_
