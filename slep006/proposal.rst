.. _slep_006:

=========================
SLEP006: Metadata Routing
=========================

:Author: Joel Nothman, Adrin Jalali, Alex Gramfort, Thomas J. Fan
:Status: Under Review
:Type: Standards Track
:Created: 2019-03-07

Abstract
--------

This SLEP proposes an API to configure estimators, scorers, and CV splitters to
request metadata in method calls. Meta-estimators or functions that consume
estimators, scorers, or CV splitters will use this API to inspect and pass in
the requested metadata.

Motivation and Scope
--------------------

Scikit-learn has limited support for passing around information that is not
`(X, y)`. For example, to pass `sample_weight` to a step of a `Pipeline`, one
needs to specify the step using dunder (`__`)  prefixing::

    >>> pipe = Pipeline([..., ('clf', LogisticRegression())])
    >>> pipe.fit(X, y, clf__sample_weight=sample_weight)

Several other meta-estimators, such as `GridSearchCV`, support forwarding these
fit parameters to their base estimator when fitting. Yet a number of important
use cases are currently not supported.

* Passing metadata (e.g. `sample_weight`) to a scorer used in cross-validation
* Passing metadata (e.g. `groups`) to a CV splitter in nested cross-validation
* Passing metadata (e.g. `sample_weight`) to some scorers and not others in a
  multi-metric cross-validation setup.
* Passing metadata to non-fit methods. For example, passing group indices
  for samples that are to be treated as a single sequence in prediction.

We define the following terms in this proposal:

* **consumer**: An object that receives and consumes metadata, such as
  estimators, scorers, or CV splitters.

* **producer**: An object that passes metadata to a **consumer**, such as
  be a meta-estimator or a function. (e.g. as `GridSearchCV` or
  `cross_validate`)

* **key**: A label passed along with the metadata (e.g. `sample_weight`).

This SLEP proposes to add `get_metadata_request` to all **consumers**,
`*_request` to estimators and CV splitters, and a `request_metadata`
keyword parameter to `make_scorer`. `score_request` configures an
estimator to request metadata::

    >>> log_reg = LogisticRegression()
    >>> log_reg.fit_request(sample_weight='sample_weight')

`get_metadata_request` are used by **producers** to inspect
the metadata needed by  **consumers**::

    >>> log_reg.get_metadata_request()
    {
        'fit': {'sample_weight': 'sample_weight'},
        'partial_fit': {},
        'predict': {},
        'transform': {},
        'score': {},
        'split': {},
        'inverse_transform': {}
     }

The `score_request` are provided to scorers through `make_scorer` and
`get_metadata_request` provides the metadata::

    >>> weighted_acc = make_scorer(accuracy_score,
    ...                            score_request=['sample_weight'])
    >>> weighted_acc.get_metadata_request()['score']
    {'score': {'sample_weight': 'sample_weight'}}

Usage and Impact
----------------

This SLEP unlocks many machine learning use cases that were not possible
before. The following examples demonstrates nested grouped cross validation
where a scorer and an estimator requests `sample_weight` and `groups` is
automatically passed to `GroupKFold`. When the user passes in `metadata` to
`cross_validate`, the function will use `get_metadata_request` to correctly
pass metadata to the **consumers**::

    >>> weighted_acc = make_scorer(accuracy_score,
    ...                            score_request=['sample_weight'])
    >>> group_cv = GroupKFold()
    >>> lr_weight = (LogisticRegressionCV(cv=group_cv, scoring=weighted_acc)
    ...              .fit_request(sample_weight=True]))
    >>> cross_validate(lr_weight, X, y, cv=group_cv,
    ...                metadata={'sample_weight': weights, 'groups': groups},
    ...                scoring=weighted_acc)

To support unweighted fitting and weighted scoring::

    >>> lr_unweight = (LogisticRegressionCV(cv=group_cv, scoring=weighted_acc)
    ...                .fit_request(sample_weight=False))
    >>> cross_validate(lr, X, y, cv=group_cv,
    ...                metadata={'sample_weight': weights, 'groups': groups},
    ...                scoring=weighted_acc)

Finally, the API can support key alias, which is useful when **consumers** need
different metadata. In the following example, the scorer needs a
`'scoring_weight'` while the estimator needs a different `'fitting_weight'`::

    >>> weighted_acc = make_scorer(
    ...     accuracy_score,
    ...     request_metadata={'sample_weight': 'scoring_weight'})
    >>> lr_weight = (LogisticRegressionCV(cv=group_cv, scoring=weighted_acc)
    ...              .fit_request(sample_weight='fitting_weight'))
    >>> lr_weight.get_metadata_request()
    {
        'fit': {'sample_weight': 'fitting_weight',
                'scoring_weight': 'scoring_weight'}
        'score': {'sample_weight': 'scoring_weight'}
        'split': {'groups': 'groups'}
        ...
    }
    >>> cross_validate(lr_weight, X, y, cv=group_cv,
    ...                metadata={'scoring_weight': scoring_weights,
    ...                          'fitting_weight': fitting_weight,
    ...                          'groups': groups},
    ...                scoring=weighted_acc)

Detailed description
--------------------

This SLEP proposes to add `get_metadata_request` to all **consumers**,
`*_request` to estimators and CV splitters, and a `request_metadata`
keyword parameter to `make_scorer`.

`get_metadata_request` returns a dictionary that specifies what metadata is
required by a **consumer**'s methods. For estimators, the relevant keys are:
`fit`, `transform`, `predict`, `transform`, `score`, and `inverse_transform`.
The only relevant key for CV splitters is `split` and the for scorers is
`score`. The values of the metadata dictionary is another dictionary. This
inner dictionary maps from a **key** to a **key** alias. For example, the
following asks the **producer** to pass in the metadata associated with
`'fitting_sample_weight'` as the `sample_weight` for `estimator.fit`::

    >>> estimator.get_metadata_request()['fit']
    {'sample_weight': 'fitting_sample_weight'}
    >>> estimator.fit(X, y, sample_weight=metadata['fitting_sample_weight'])

Note that it is optional for **producers** to pass in the metadata to the
**consumer**. For scorers, the `'score'` **key** provides metadata for
calling scorer itself and not a `score` method.

`*_request` configures the metadata requested by a **consumer**'s method. For
example, `fit_request` configures the metadata to be routed to `fit`.
`fit_request's` signature maps **keys** to a `bool`, `str`, or `None` value.
When the value is `None`, the **consumer** does not output any metadata
request.

    >>> est.fit_request(sample_weight=None)
    >>> est.get_metadata_request()['fit']
    {}

 When the value is a `True`, it configures the **consumer** to expect the `key`
 as metadata. For example, the following configures `est` to expect
 `sample_weight` and `groups` to be passed into `fit`::

    >>> est.fit_request(sample_weight=True, groups=True)
    >>> est.get_metadata_request()['fit']
    { 'sample_weight': 'sample_weight', 'groups': 'groups'}

If the value is `False`, it configures the **consumer** to *not* expect the
`key` as metadata::

    >>> est.fit_request(sample_weight=False)
    >>> est.get_metadata_request()['fit']
    {'sample_weight': False}

If the value is a `str`, that string is used as the **key** alias for that
metadata. For example, the following configures `log_reg` to
expect a **key** alias `'my_sample_weight'` that should be passed to
`sample_weight`::

    >>> log_reg = (LogisticRegression()
    ...            .fit_request(sample_weight='my_sample_weight')
    >>> log_reg.get_metadata_request()["fit"]
    {'sample_weight': 'my_sample_weight'}
    >>> # Note that `sample_weight` is the key
    >>> log_reg.fit(X, y, sample_weight=metadata['my_sample_weight'])

For scorers, `make_scorer` accepts `score_request` to configure the
metadata it accepts::

    >>> acc = make_scorer(accuracy_score, score_request='sample_weight')
    >>> acc.get_metadata_request()['score']
    {'sample_weight': 'sample_weight'}

For CV splitters that split on groups, their default metadata request
is `groups`::

    >>> group_fold = GroupKFold()
    >>> group_fold.get_metadata_request()['split']
    {'groups': 'groups'}

With the exception of `Group*CV`, the default values in `*_request` is set to
`None`. By default, `Group*CV` will require `groups` in it's `split` method .
Setting metadata request does not alter the behavior of the **consumer**. The
**producer** is responsible for validating the passed in metadata.

Backward compatibility
----------------------

Scikit-learn's meta-estimators will deprecate the dunder (`__`) syntax for
routing and enforce explicit request method calls. During the deprecation
period, using dunder syntax routing and explicit request calls together will
raise an error.

During the deprecation period, meta-estimators such as `GridSearchCV` will
route `fit_params` to the inner estimators' `fit` by default, but
a deprecation warning is raised::

    >>> # Deprecation warning, stating that the provided metadata is not
    >>> # requested
    >>> GridSearchCV(LogisticRegression()).fit(X, y, sample_weight=sw)

Meta-estimators such as `GridSearchCV` will check that the metadata requested
and will error when metadata is passed in, but the inner estimator is
not configured to request it::

    >>> grid = GridSearchCV(
    ...     LogisticRegression(),
    ...     scoring=make_scorer(accuracy_score,
    ...                         request_metadata=['sample_weight'])
    ... )
    >>> # Raise that LR could accept `sample_weight`, but has
    >>> # not been specified by the user
    >> grid.fit(X, y, sample_weight=sw)

Third-party estimators will need to adopt this SLEP in order to support
metadata routing, while the dunder syntax is deprecated. Third-party
estimators that contain **consumers** will need to define
**get_metadata_request** that exposes the metadata of its **consumers**.
Their methods will need to be updated to correctly route data to the
**consumers**. Our implementation will provide utilities to help developers
adopt this SLEP.

Implementation
--------------

This SLEP has a draft implementation at :pr:`16079` by user:`adrinjalali`. The
implementation provides utilities that is used by scikit-learn and available to
third-party estimators for adopting this SLEP.

Alternatives
------------

Over the years, there has been many proposed alternatives before we landed
on this SLEP:

* :pr:`4696` A first implementation by :user:`amueller`
* `Discussion towards SLEP004
  <https://github.com/scikit-learn/enhancement_proposals/pull/6>`__ initiated
  by :user:`tguillemot`.
* :pr:`9566` Another implementation (solution 3 from this SLEP)
  by :user:`jnothman`
* This SLEP has emerged from many alternatives that is seen at
  :ref:`slep_006_other`.

Discussion & Related work
-------------------------

This SLEP was drafted based on the discussions of potential solutions
at the February 2019 development sprint in Paris. The overarching issue is
fond at "Consistent API for attaching properties to samples" at :issue:`4497`.

Related issues and discussions include: :issue:`1574`, :issue:`2630`,
:issue:`3524`, :issue:`4632`, :issue:`4652`, :issue:`4660`, :issue:`4696`,
:issue:`6322`, :issue:`7112`, :issue:`7646`, :issue:`7723`, :issue:`8127`,
:issue:`8158`, :issue:`8710`, :issue:`8950`, :issue:`11429`, :issue:`12052`,
:issue:`15282`, :issue:`15370`, :issue:`15425`, :issue:`18028`.

One benefit of the explicitness in this proposal is that even if it makes use
of `**kwarg` arguments, it does not preclude keywords arguments serving other
purposes.  In addition to requesting sample metadata, a future proposal could
allow estimators to request feature metadata or other keys.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.
.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
