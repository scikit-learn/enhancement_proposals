.. _slep_012:

==========
InputArray
==========

This proposal suggests adding a new data structure, called ``InputArray``,
which wraps a data matrix with some added information about the data. This was
motivated when working on input and output feature names. Since we expect the
feature names to be attached to the data given to an estimator, there are a few
approaches we can take:

- ``pandas`` in, ``pandas`` out: this means we expect the user to give the data
  as a ``pandas.DataFrame``, and if so, the transformer would output a
  ``pandas.DataFrame`` which also includes the [generated] feature names. This
  is not a feasible solution since ``pandas`` plans to move to a per column
  representation, which means ``pd.DataFrame(np.asarray(df))`` has two
  guaranteed memory copies.
- ``XArray``: we could accept a `pandas.DataFrame``, and use
  ``xarray.DataArray`` as the output of transformers, including feature names.
  However, ``xarray`` depends on ``pandas``, and uses ``pandas.Series`` to
  handle row labels and aligns rows when an operation between two
  ``xarray.DataArray`` is done. None of these are favorable for our use-case.

As a result, we need to have another data structure which we'll use to transfer
data related information (such as feature names), which is lightweight and
doesn't interfere with existing user code.

A main constraint of this data structure is that is should be backward
compatible, *i.e.* code which expects a ``numpy.ndarray`` as the output of a
transformer, would not break. This SLEP focuses on *feature names* as the only
meta-data attached to the data. Support for other meta-data can be added later.


Feature Names
*************

Feature names are an array of strings aligned with the columns. They can be
``None``.

Operations
**********

All usual operations (including slicing through ``__getitem__``) return an
``np.ndarray``. The ``__array__`` method also returns the underlying data, w/o
any modifications. This prevents any unwanted computational overhead as a
result of migrating to this data structure.

The ``select()`` method will act like a ``__getitem__``, except that it
understands feature names and it also returns an ``InputArray``, with the
corresponding meta-data.

Sparse Arrays
*************

All of the above applies to sparse arrays.

Factory Methods
***************

There will be factory methods creating an ``InputArray`` given a
``pandas.DataFrame`` or an ``xarray.DataArray`` or simply an ``np.ndarray`` or
an ``sp.SparseMatrix`` and a given set of feature names.

An ``InputArray`` can also be converted to a `pandas.DataFrame`` using a
``toDataFrame()`` method.
