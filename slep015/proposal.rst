.. _slep_015:

==================================
SLEP015: Feature Names Propagation
==================================

:Author: Thomas J Fan
:Status: Draft
:Type: Standards Track
:Created: 2020-10-03

Abstract
########

This SLEP proposes adding the `feature_names_in_` attribute for all estimators
and the ``get_feature_names_out`` method too all transformers.

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
would need to be used to select column names that were selected.

Solution
########

This SLEP proposes adding the ``feature_names_in_`` attribute to all estimators
that will extract the feature names of ``X`` during ``fit``. This will also
be used for validation during non-``fit`` methods such as ``transform`` or
``predict``. If the ``X`` is not a recognized container, then
``feature_names_in_`` would be set to ``None``.

Secondly, this SLEP proposes adding
``get_feature_names_out(feature_names_in=None)`` to all transformers. By
default, the input features will be determined by the ``feature_names_in_``
attribute. The feature names of a pipeline can then be easily extracted as
follows::

    pipe[:-1].get_feature_names_out()
    # ['cat__letter_a', 'cat__letter_b', 'cat__letter_c',
       'cat__pet_dog', 'cat__pet_snake', 'num__distance']

Note that ``get_feature_names_out`` does not require ``feature_names_in``
because the feature names was stored in the pipeline itself. These
features will be passed to each step's `get_feature_names_out` method to
obtain the output feature names of the `Pipeline` itself.

Enabling Functionality
######################

The following enhancements are **not** a part of this SLEP. These features are
made possible if this SLEP gets accepted.

1. Transformers in a pipeline may wish to have feature names passed in as
``X``. This can be enabled by adding a ``array_input`` parameter to
``Pipeline``::

    pipe = make_pipeline(ct, MyTransformer(), LogisticRegression(),
                         array_input='pandas')

In this case, the pipeline will construct a pandas DataFrame to be inputted
into ``MyTransformer`` and ``LogisticRegression``. The feature names
will be constructed by calling ``get_feature_names_out`` as data is passed
through the ``Pipeline``.

Considerations
##############

The ``get_feature_names_out`` will be constructed using the name generation
specification from [slep_007]_.

For a ``Pipeline`` with only one estimator, slicing will not work and one
would need to access the feature names directly::

    pipe = make_pipeline(LogisticRegression())
    pipe[-1].feature_names_in_

Backward compatibility
######################

This SLEP is fully backward compatibility with previous versions. With the
introduction of ``get_feature_names_out``, ``get_feature_names`` will
be deprecated.

The inclusion of ``get_feature_names_out`` and ``feature_names_in_`` will
not introduce any overhead to ``Pipeline``.

Community Adoption
##################

We can enforce the ``feature_names_in_`` attribute and
``get_feature_names_out`` method with additional tests to
``check_estimator``.

Alternatives
############

There have been many attempts to address this issue:

1. ``array_out`` in keyword parameter in ``transform`` : This approach requires
   third party estimators to unwrap and wrap array containers to in transform,
   which introduces more burden for third party estimator maintainers. This
   SLEP is easier to implement because it requires less changes. Furthermore,
   ``array_out`` with sparse data will introduce an overhead when being passed
   along in a ``Pipeline``.

2. [slep_007]_ : For ``SLEP007`` to function it still requires a mechanism to
   pass feature names through a pipeline such as ``array_out``.
   Furthermore, ``feature_names_out_`` would not be needed because it can be
   computed with ``get_feature_names_out``.

3. [slep_012] : The ``InputArray`` was developed to work around the overhead
    of using a pandas ``DataFrame`` or an xarray ``DataArray``. The
    introduction of another data structure into the Python Data Ecosystem,
    would be lead to more burden for third party estimator maintainers.


References and Footnotes
########################

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
#########

This document has been placed in the public domain. [1]_
