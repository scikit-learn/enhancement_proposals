.. _slep_006:

=========================
SLEP006: Metadata Routing
=========================

:Author: Joel Nothman, Adrin Jalali, Alex Gramfort, Thomas J. Fan
:Status: Accepted
:Type: Standards Track
:Created: 2019-03-07

Abstract
--------

This SLEP proposes an API to configure estimators, scorers, and CV splitters to
request metadata when calling methods such as `fit`, `predict`, etc.
Meta-estimators or functions that wrap estimators, scorers, or CV splitters will
use this API to pass in the requested metadata.

Motivation and Scope
--------------------

Scikit-learn has limited support for passing around information that is not
`(X, y)`. For example, to pass `sample_weight` to a step of a `Pipeline`, one
needs to specify the step using dunder (`__`)  prefixing::

    >>> pipe = Pipeline([..., ('clf', LogisticRegression())])
    >>> pipe.fit(X, y, clf__sample_weight=sample_weight)

Several other meta-estimators, such as `GridSearchCV`, support forwarding these
fit parameters to their base estimator when fitting. Yet a number of important
use cases are currently not supported:

* Passing metadata (e.g. `sample_weight`) to a scorer used in cross-validation
* Passing metadata (e.g. `groups`) to a CV splitter in nested cross-validation
* Passing metadata (e.g. `sample_weight`) to some scorers and not others in
  multi-metric cross-validation. This is also required to handle fairness
  related metrics which usually expect one or more sensitive attributes to be
  passed to them along with the data.
* Passing metadata to non-`fit` methods. For example, passing group indices for
  samples that are to be treated as a single sequence in prediction, or passing
  sensitive attributes to `predict` or `transform` of a fairness related
  estimator.

We define the following terms in this proposal:

* **consumer**: An object that receives and consumes metadata, such as
  estimators, scorers, or CV splitters.

* **router**: An object that passes metadata to a **consumer** or
  another **router**. Examples of **routers** include meta-estimators or
  functions. (For example `GridSearchCV` or `cross_validate` route sample
  weights, cross validation groups, etc. to **consumers**)

This SLEP proposes to add

* `get_metadata_routing` to all **consumers** and **routers**
  (i.e. all estimators, scorers, and splitters supporting this API)
* `set_*_request` to consumers (including estimators, scorers, and CV
  splitters), where `*` is a method that requires metadata. (e.g.
  `set_fit_request`, `set_score_request`, `set_transform_request`, etc.)

For example, `set_fit_request` configures an estimator to request metadata::

    >>> log_reg = LogisticRegression().set_fit_request(sample_weight=True)

`get_metadata_routing` are used by **routers** to inspect the metadata needed
by **consumers**. `get_metadata_routing` returns a `MetadataRouter` or a
`MetadataRequest` object that stores and handles metadata routing.
`get_metadata_routing` returns enough information for a router to know what
metadata is requested, and whether the metadata is sample aligned or not. See
the draft implementation for more implementation details.

Note that in the core library nothing is requested by default, except
``groups`` in ``Group*CV`` objects which request the ``groups`` metadata. At
the time of writing this proposal, all metadata requested in the core library
are sample aligned.

Also note that ``X``, ``y``, and ``Y`` input arguments are never automatically
added to the routing mechanism and are always passed into their respective
methods.

Detailed description
--------------------

This SLEP unlocks many machine learning use cases that were not possible
before. In this section, we will focus on some workflows that are made possible
by this SLEP.

Nested Grouped Cross Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following examples demonstrates nested grouped cross validation
where a scorer and an estimator requests `sample_weight` and `GroupKFold`
requests `groups` by default::

    >>> weighted_acc = make_scorer(accuracy_score).score_request(sample_weight=True)
    >>> log_reg = (LogisticRegressionCV(cv=GroupKFold(), scoring=weighted_acc)
    ...           .set_fit_request(sample_weight=True))
    >>> cv_results = cross_validate(
    ...     log_reg, X, y,
    ...     cv=GroupKFold(),
    ...     metadata={"sample_weight": my_weights, "groups": my_groups},
    ...     scoring=weighted_acc)

To support unweighted fitting and weighted scoring, metadata is set to `False`
in `fit_request`::

    >>> log_reg = (LogisticRegressionCV(cv=group_cv, scoring=weighted_acc)
    ...           .fit_request(sample_weight=False))
    >>> cross_validate(
    ...     log_reg, X, y,
    ...     cv=GroupKFold(),
    ...     metadata={'sample_weight': weights, 'groups': groups},
    ...     scoring=weighted_acc)

Unweighted Feature selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Consumers** that do not accept weights during fitting such as `SelectKBest`
will _not_ be routed weights::

    >>> log_reg = (LogisticRegressionCV(cv=GroupKFold(), scoring=weighted_acc)
    ...            .set_fit_request(sample_weight=True))
    >>> sel = SelectKBest(k=2)
    >>> pipe = make_pipeline(sel, log_reg)
    >>> pipe.fit(X, y, sample_weight=weights, groups=groups)

Note that if a **consumer** or a **router** starts accepting and consuming a
certain metadata, the developer API enables developers to raise a warning
and avoid silent behavior changes in users' code. See the draft implementation
for more details.

Different Scoring and Fitting Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can pass different weights for scoring and fitting by using an aliases. In
this example, `scoring_weight` is passed to the scoring and `fitting_weight`
is passed to `LogisticRegressionCV`::

    >>> weighted_acc = (make_scorer(accuracy_score)
    ...                 .set_score_request(sample_weight="scoring_weight"))
    >>> log_reg = (LogisticRegressionCV(cv=GroupKFold(), scoring=weighted_acc)
    ...            .set_fit_request(sample_weight="fitting_weight"))
    >>> cv_results = cross_validate(
    ...     log_reg, X, y,
    ...     cv=GroupKFold(),
    ...     metadata={"scoring_weight": my_weights,
    ...            "fitting_weight": my_other_weights,
    ...            "groups": my_groups},
    ...     scoring=weighted_acc)

Nested Grouped Cross Validation with SearchCV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since `GroupKFold` requests group metadata by default, `GroupKFold` instances can
be passed to multiple **routers** to enable nested grouped cross validation. In
this example, both `RandomizedSearchCV` and `cross_validate` set
`cv=GroupKFold()` which enables grouped CV in the outer loop (`cross_validate`)
and the inner random search::

    >>> log_reg = LogisticRegression()
    >>> distributions = {"C": uniform(loc=0, scale=4),
    ...                  "penalty": ['l2', 'l1']}
    >>> random_search = RandomizedSearchCV(log_reg, distributions, cv=GroupKFold())
    >>> cv_results = cross_validate(
    ...     log_reg, X, y,
    ...     cv=GroupKFold(),
    ...     metadata={"groups": my_groups})

Implementation
--------------

This SLEP has a draft implementation at :pr:`22083` by :user:`adrinjalali`. The
implementation provides developer utilities that are used by scikit-learn and
available to third-party estimators for adopting this SLEP. Specifically, the
draft implementation makes it easier to define `get_metadata_routing` and
`set_*_request` for **consumers** and **routers**.

Backward compatibility
----------------------

Scikit-learn's meta-estimators will deprecate the dunder (`__`) syntax for
routing and enforce explicit request method calls. During the deprecation
period, using dunder syntax routing and explicit request calls together will
raise an error.

During the deprecation period, meta-estimators such as `GridSearchCV` will
route `fit_params` to the inner estimators' `fit` by default, but
a deprecation warning is raised::

    >>> # Deprecation warning, stating that the provided metadata is not requested
    >>> GridSearchCV(LogisticRegression(), ...).fit(X, y, sample_weight=sw)

To avoid the warning, one would need to specify the request in
`LogisticRegression`::

    >>> grid = GridSearchCV(
    ...     LogisticRegression().set_fit_request(sample_weight=True), ...
    ... )
    >>> grid.fit(X, y, sample_weight=sw)

Meta-estimators such as `GridSearchCV` will check which metadata is requested,
and will error when metadata is passed in and the inner estimator is
not configured to request it::

    >>> weighted_acc = make_scorer(accuracy_score).score_request(sample_weight=True)
    >>> log_reg = LogisticRegression()
    >>> grid = GridSearchCV(log_reg, ..., scoring=weighted_scorer)
    >>>
    >>> # Raise a TypeError that log_reg is not specified with any routing
    >>> # metadata for `sample_weight`, but sample_weight has been passed in to
    >>> # `grid.fit`.
    >>> grid.fit(X, y, sample_weight=sw)

To avoid the error, `LogisticRegression` must specify its metadata request by
calling `set_fit_request`::

    >>> # Request sample weights
    >>> log_reg_weights = LogisticRegression().set_fit_request(sample_weight=True)
    >>> grid = GridSearchCV(log_reg_with_weights, ...)
    >>> grid.fit(X, y, sample_weight=sw)
    >>>
    >>> # Do not request sample_weights
    >>> log_reg_no_weights = LogisticRegression().set_fit_request(sample_weight=False)
    >>> grid = GridSearchCV(log_reg_no_weights, ...)
    >>> grid.fit(X, y, sample_weight=sw)

Note that a meta-estimator will raise an error if the user passes a metadata
which is not requested by any of the child objects of the meta-estimator.

Third-party estimators will need to adopt this SLEP in order to support
metadata routing, while the dunder syntax is deprecated. Our implementation
will provide developer APIs to trigger warnings and errors as described above
to help with adopting this SLEP.

Alternatives
------------

Over the years, there have been many proposed alternatives before we landed
on this SLEP:

* :pr:`4696` A first implementation by :user:`amueller`
* `Discussion towards SLEP004
  <https://github.com/scikit-learn/enhancement_proposals/pull/6>`__ initiated
  by :user:`tguillemot`.
* :pr:`9566` Another implementation (solution 3 from this SLEP)
  by :user:`jnothman`
* This SLEP has emerged from many alternatives detailed at
  :ref:`slep_006_other`.

Discussion & Related work
-------------------------

This SLEP was drafted based on the discussions of potential solutions
at the February 2019 development sprint in Paris. The overarching issue is
found at "Consistent API for attaching properties to samples" at :issue:`4497`.

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
