.. _slep_012:

=======================
SLEP012: ``InputArray``
=======================

:Author: Adrin jalali
:Status: Draft
:Type: Standards Track
:Created: 2019-12-20

Motivation
##########

This proposal results in a solution to propagating feature names through
transformers, pipelines, and the column transformer. Ideally, we would have::

    df = pd.readcsv('tabular.csv')
    # transforming the data in an arbitrary way
    transformer0 = ColumnTransformer(...)
    # a pipeline preprocessing the data and then a classifier (or a regressor)
    clf = make_pipeline(transformer0, ..., SVC())

    # now we can investigate features at each stage of the pipeline
    clf[-1].input_feature_names_

The feature names are propagated throughout the pipeline and the user can
investigate them at each step of the pipeline.

This proposal suggests adding a new data structure, called ``InputArray``,
which augments the data array ``X`` with additional meta-data. In this proposal
we assume the feature names (and other potential meta-data) are attached to the
data when passed to an estimator. Alternative solutions are discussed later in
this document.

A main constraint of this data structure is that is should be backward
compatible, *i.e.* code which expects a ``numpy.ndarray`` as the output of a
transformer, would not break. This SLEP focuses on *feature names* as the only
meta-data attached to the data. Support for other meta-data can be added later.

Backward/NumPy/Pandas Compatibility
###################################

Since currently transformers return a ``numpy`` or a ``scipy`` array, backward
compatibility in this context means the operations which are valid on those
arrays should also be valid on the new data structure.

All operations are delegated to the *data* part of the container, and the
meta-data is lost immediately after each operation and operations result in a
``numpy.ndarray``. This includes indexing and slicing, *i.e.* to avoid
performance degradation, ``__getitem__`` is not overloaded and if the user
wishes to preserve the meta-data, they shall do so via explicitly calling a
method such as ``select()``. Operations between two ``InpuArray``s will not
try to align rows and/or columns of the two given objects.

``pandas`` compatibility comes ideally as a ``pd.DataFrame(inputarray)``, for
which ``pandas`` does not provide a clean API at the moment. Alternatively,
``inputarray.todataframe()`` would return a ``pandas.DataFrame`` with the
relevant meta-data attached.

Feature Names
#############

Feature names are an object ``ndarray`` of strings aligned with the columns.
They can be ``None``.

Operations
##########

Estimators understand the ``InputArray`` and extract the feature names from the
given data before applying the operations and transformations on the data.

All transformers return an ``InputArray`` with feature names attached to it.
The way feature names are generated is discussed in *SLEP007 - The Style of The
Feature Names*.

Sparse Arrays
#############

Ideally sparse arrays follow the same pattern, but since ``scipy.sparse`` does
not provide the kinda of API provided by ``numpy``, we may need to find
compromises.

Factory Methods
###############

There will be factory methods creating an ``InputArray`` given a
``pandas.DataFrame`` or an ``xarray.DataArray`` or simply an ``np.ndarray`` or
an ``sp.SparseMatrix`` and a given set of feature names.

An ``InputArray`` can also be converted to a ``pandas.DataFrame`` using a
``todataframe()`` method.

``X`` being an ``InputArray``::

    >>> np.array(X)
    >>> X.todataframe()
    >>> pd.DataFrame(X) # only if pandas implements the API

And given ``X`` a ``np.ndarray`` or an ``sp.sparse`` matrix and a set of
feature names, one can make the right ``InputArray`` using::

    >>> make_inputarray(X, feature_names)

Alternative Solutions
#####################

Since we expect the feature names to be attached to the data given to an
estimator, there are a few potential approaches we can take:

- ``pandas`` in, ``pandas`` out: this means we expect the user to give the data
  as a ``pandas.DataFrame``, and if so, the transformer would output a
  ``pandas.DataFrame`` which also includes the [generated] feature names. This
  is not a feasible solution since ``pandas`` plans to move to a per column
  representation, which means ``pd.DataFrame(np.asarray(df))`` has two
  guaranteed memory copies.
- ``XArray``: we could accept a ``pandas.DataFrame``, and use
  ``xarray.DataArray`` as the output of transformers, including feature names.
  However, ``xarray`` has a hard dependency on ``pandas``, and uses
  ``pandas.Index`` to handle row labels and aligns rows when an operation
  between two ``xarray.DataArray`` is done, which can be time consuming, and is
  not the semantic expected in ``scikit-learn``; we only expect the number of
  rows to be equal, and that the rows always correspond to one another in the
  same order.

As a result, we need to have another data structure which we'll use to transfer
data related information (such as feature names), which is lightweight and
doesn't interfere with existing user code.

Another alternative to the problem of passing meta-data around is to pass that
as a parameter to ``fit``. This would heavily involve modifying meta-estimators
since they'd need to pass that information, and extract the relevant
information from the estimators to pass that along to the next estimator. Our
prototype implementations showed significant challenges compared to when the
meta-data is attached to the data.
