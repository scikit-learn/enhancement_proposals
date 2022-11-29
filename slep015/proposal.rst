.. _slep_015:

==================================
SLEP015: Feature Names Propagation
==================================

:Author: Thomas J Fan
:Status: Rejected
:Type: Standards Track
:Created: 2020-10-03

Abstract
########

This SLEP proposes adding the ``get_feature_names_out`` method to all
transformers and the ``feature_names_in_`` attribute for all estimators.
The ``feature_names_in_`` attribute is set during ``fit`` if the input, ``X``,
contains the feature names.

Motivation
##########

``scikit-learn`` is commonly used as a part of a larger data processing
pipeline. When this pipeline is used to transform data, the result is a
NumPy array, discarding column names. The current workflow for
extracting the feature names requires calling ``get_feature_names`` on the
transformer that created the feature. This interface can be cumbersome when used
together with a pipeline with multiple column names::

    X = pd.DataFrame({'letter': ['a', 'b', 'c'],
                      'pet': ['dog', 'snake', 'dog'],
                      'distance': [1, 2, 3]})
    y = [0, 0, 1]
    orig_cat_cols, orig_num_cols = ['letter', 'pet'], ['num']

    ct = ColumnTransformer(
        [('cat', OneHotEncoder(), orig_cat_cols),
         ('num', StandardScaler(), orig_num_cols)])
    pipe = make_pipeline(ct, LogisticRegression()).fit(X, y)

    cat_names = (pipe['columntransformer']
                 .named_transformers_['onehotencoder']
                 .get_feature_names(orig_cat_cols))

    feature_names = np.r_[cat_names, orig_num_cols]

The ``feature_names`` extracted above corresponds to the features directly
passed into ``LogisticRegression``. As demonstrated above, the process of
extracting ``feature_names`` requires knowing the order of the selected
categories in the ``ColumnTransformer``. Furthermore, if there is feature
selection in the pipeline, such as ``SelectKBest``, the ``get_support`` method
would need to be used to infer the column names that were selected.

Solution
########

This SLEP proposes adding the ``feature_names_in_`` attribute to all estimators
that will extract the feature names of ``X`` during ``fit``. This will also
be used for validation during non-``fit`` methods such as ``transform`` or
``predict``. If the ``X`` is not a recognized container with columns, then
``feature_names_in_`` can be undefined. If ``feature_names_in_`` is undefined,
then it will not be validated.

Secondly, this SLEP proposes adding ``get_feature_names_out(input_names=None)``
to all transformers. By default, the input features will be determined by the
``feature_names_in_`` attribute. The feature names of a pipeline can then be
easily extracted as follows::

    pipe[:-1].get_feature_names_out()
    # ['cat__letter_a', 'cat__letter_b', 'cat__letter_c',
       'cat__pet_dog', 'cat__pet_snake', 'num__distance']

Note that ``get_feature_names_out`` does not require ``input_names``
because the feature names was stored in the pipeline itself. These
features will be passed to each step's ``get_feature_names_out`` method to
obtain the output feature names of the ``Pipeline`` itself.

Enabling Functionality
######################

The following enhancements are **not** a part of this SLEP. These features are
made possible if this SLEP gets accepted.

1. This SLEP enables us to implement an ``array_out`` keyword argument to
   all ``transform`` methods to specify the array container outputted by
   ``transform``. An implementation of ``array_out`` requires
   ``feature_names_in_`` to validate that the names in ``fit`` and
   ``transform`` are consistent. An implementation of ``array_out`` needs
   a way to map from the input feature names to output feature names, which is
   provided by ``get_feature_names_out``.

2. An alternative to ``array_out``: Transformers in a pipeline may wish to have
   feature names passed in as ``X``. This can be enabled by adding a
   ``array_input`` parameter to ``Pipeline``::

        pipe = make_pipeline(ct, MyTransformer(), LogisticRegression(),
                             array_input='pandas')

   In this case, the pipeline will construct a pandas DataFrame to be inputted
   into ``MyTransformer`` and ``LogisticRegression``. The feature names
   will be constructed by calling ``get_feature_names_out`` as data is passed
   through the ``Pipeline``. This feature implies that ``Pipeline`` is
   doing the construction of the DataFrame.

Considerations and Limitations
##############################

1. The ``get_feature_names_out`` will be constructed using the name generation
   specification from :ref:`slep_007`.

2. For a ``Pipeline`` with only one estimator, slicing will not work and one
   would need to access the feature names directly::

      pipe1 = make_pipeline(StandardScaler(), LogisticRegression())
      pipe[:-1].feature_names_in_  # Works

      pipe2 = make_pipeline(LogisticRegression())
      pipe[:-1].feature_names_in_  # Does not work

   This is because `pipe2[:-1]` raises an error because it will result in
   a pipeline with no steps. We can work around this by allowing pipelines
   with no steps.

3. ``feature_names_in_`` can be any 1-D ``Sequence``, such as an list or
   an ndarray.

4. Meta-estimators will delegate the setting and validation of
   ``feature_names_in_`` to its inner estimators. The meta-estimator will
   define ``feature_names_in_`` by referencing its inner estimators. For
   example, the ``Pipeline`` can use ``steps[0].feature_names_in_`` as
   the input feature names. If the inner estimators do not define
   ``feature_names_in_`` then the meta-estimator will not defined
   ``feature_names_in_`` as well.

Backward compatibility
######################

1. This SLEP is fully backward compatible with previous versions. With the
   introduction of ``get_feature_names_out``, ``get_feature_names`` will
   be deprecated. Note that ``get_feature_names_out``'s signature will
   always contain ``input_features`` which can be used or ignored. This
   helps standardize the interface for the get feature names method.

2. The inclusion of a ``get_feature_names_out`` method will not introduce any
   overhead to estimators.

3. The inclusion of a ``feature_names_in_`` attribute will increase the size of
   estimators because they would store the feature names. Users can remove
   the attribute by calling ``del est.feature_names_in_`` if they want to
   remove the feature and disable validation.

Alternatives
############

There have been many attempts to address this issue:

1. ``array_out`` in keyword parameter in ``transform`` : This approach requires
   third party estimators to unwrap and wrap array containers in transform,
   which introduces more burden for third party estimator maintainers.
   Furthermore, ``array_out`` with sparse data will introduce an overhead when
   being passed along in a ``Pipeline``. This overhead comes from the
   construction of the sparse data container that has the feature names.

2. :ref:`slep_007` : ``SLEP007`` introduces a ``feature_names_out_`` attribute
   while this SLEP proposes a ``get_feature_names_out`` method to accomplish
   the same task. The benefit of the ``get_feature_names_out`` method is that
   it can be used even if the feature names were not passed in ``fit`` with a
   dataframe. For example, in a ``Pipeline`` the feature names are not passed
   through to each step and a ``get_feature_names_out`` method can be used to
   get the names of each step with slicing.

3. :ref:`slep_012` : The ``InputArray`` was developed to work around the
   overhead of using a pandas ``DataFrame`` or an xarray ``DataArray``. The
   introduction of another data structure into the Python Data Ecosystem, would
   lead to more burden for third party estimator maintainers.


References and Footnotes
########################

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
#########

This document has been placed in the public domain. [1]_
