.. _slep_006:

==========================================
SLEP006: Routing sample-aligned meta-data 
==========================================

:Author: Joel Nothman, Adrin Jalali, Alex Gramfort
:Status: Draft
:Type: Standards Track
:Created: 2019-03-07

Scikit-learn has limited support for information pertaining to each sample
(henceforth "sample properties") to be passed through an estimation pipeline.
The user can, for instance, pass fit parameters to all members of a
FeatureUnion, or to a specified member of a Pipeline using dunder (``__``)
prefixing::

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> pipe = Pipeline([('clf', LogisticRegression())])
    >>> pipe.fit([[1, 2], [3, 4]], [5, 6],
    ...          clf__sample_weight=[.5, .7])  # doctest: +SKIP

Several other meta-estimators, such as GridSearchCV, support forwarding these
fit parameters to their base estimator when fitting.

Features we currently do not support and which to include:

* passing sample properties (e.g. `sample_weight`) to a scorer used in
  cross-validation
* passing sample properties (e.g. `groups`) to a CV splitter in nested cross
  validation
* passing sample properties (e.g. `sample_weight`) to some
  scorers and not others in a multi-metric cross-validation setup
* passing sample properties to non-fit methods, for
  instance to index grouped samples that are to be treated as a single sequence
  in prediction.

The last two items are considered in the API design, yet will not be considered
in the first version of the implementation.

Naming
------

"Sample props" has become a name understood internally to the Scikit-learn
development team. For ongoing usage we propose to use as naming ``metadata``.

Definitions
-----------

consumer
    An estimator, scorer, splitter, etc., that receives and can make use of
    one or more passed metadata.
key
    A label passed along with sample metadata to indicate how it should be
    interpreted (e.g. "weight").
.. XXX : remove router?
router
    An estimator or function that passes metadata on to some other router or
    consumer, potentially selecting which metadata to pass to which destination,
    and by what key.

History
-------

This version was drafted after a discussion of the issue and potential
solutions at the February 2019 development sprint in Paris.

Supersedes `SLEP004
<https://github.com/scikit-learn/enhancement_proposals/tree/master/slep004>`_
with greater depth of desiderata and options.

Primary related issues and pull requests include:

- :issue:`4497`: Overarching issue,
  "Consistent API for attaching properties to samples"
  by :user:`GaelVaroquaux`
- :pr:`4696` A first implementation by :user:`amueller`
- `Discussion towards SLEP004
  <https://github.com/scikit-learn/enhancement_proposals/pull/6>`__ initiated
  by :user:`tguillemot`
- :pr:`9566` Another implementation (solution 3 from this SLEP)
  by :user:`jnothman`
- :pr:`16079` Another implementation (solution 4 from this SLEP)
  by :user:`adrinjalali`

Other related issues include: :issue:`1574`, :issue:`2630`, :issue:`3524`,
:issue:`4632`, :issue:`4652`, :issue:`4660`, :issue:`4696`, :issue:`6322`,
:issue:`7112`, :issue:`7646`, :issue:`7723`, :issue:`8127`, :issue:`8158`,
:issue:`8710`, :issue:`8950`, :issue:`11429`, :issue:`12052`, :issue:`15282`,
:issues:`15370`, :issue:`15425`, :issue:`18028`.

Desiderata
----------

We will consider the following aspects to develop and compare solutions:

.. XXX : maybe phrase this in an affirmative way
Usability
   Can the use cases be achieved in succinct, readable code? Can common use
   cases be achieved with a simple recipe copy-pasted from a QA forum?
Brittleness
   If a metadata is being routed through a Pipeline, does changing the
   structure of the pipeline (e.g. adding a layer of nesting) require rewriting
   other code?
Error handling
   If the user mistypes the name of a sample metadata, or misspecifies how it
   should be routed to a consumer, will an appropriate exception be raised?
Impact on meta-estimator design
   How much meta-estimator code needs to change? How hard will it be to
   maintain?
Impact on estimator design
   How much will the proposal affect estimator developers?
Backwards compatibility
   Can existing behavior be maintained?
Forwards compatibility
   Is the solution going to make users' code more
   brittle with future changes? (For example, will a user's pipeline change
   behaviour radically when `sample_weight` is implemented on some estimator)
Introspection
   If sensible to do so (e.g. for improved efficiency), can a
   meta-estimator identify whether its base estimator (recursively) would
   handle some particular sample metadata (e.g. so a meta-estimator can choose
   between weighting and resampling, or for automated invariance testing)?

Keyword arguments vs. a single argument
---------------------------------------

Currently, sample metadata are provided as keyword arguments to a `fit`
method. In redeveloping sample metadata, we can instead accept a single
parameter (named `metadata` or `sample_metadata`, for example) which maps
string keys to arrays of the same length (a "DataFrame-like").

Using single argument::

    >>> gs.fit(X, y, metadata={'groups': groups, 'weight': weight})

vs. using keyword arguments::

    >>> gs.fit(X, y, groups=groups, sample_weight=sample_weight)

Advantages of a single argument:

* we could consider kwargs to `fit` that are not sample-aligned, so that
  we could add further functionality (some that have been proposed:
  `with_warm_start`, `feature_names_in`, `feature_meta`).
* we would be able to redefine the default routing of weights etc. without being
  concerned by backwards compatibility.
* we could consider the use of keys that are not limited to strings or valid
  identifiers (and hence are not limited to using ``_`` as a delimiter).

Advantages of multiple keyword arguments:

* succinct
* explicit function signatures relying on interpreter checks on calls
* possible to maintain backwards compatible support for `sample_weight`, etc.
* we do not need to handle cases for whether or not some estimator expects a
  `metadata` argument.

In this SLEP, we will propose the solution based on keyword arguments.

Test case setup
---------------

Case A
~~~~~~

Cross-validate a ``LogisticRegressionCV(cv=GroupKFold(), scoring='accuracy')``
with weighted scoring and weighted fitting, while using groups in splitter.

Error handling: what would guarantee that if the user misspelled `sample_weight`
as `sample_eight` a meaningful error is raised.

Case B
~~~~~~

Cross-validate a ``LogisticRegressionCV(cv=GroupKFold(), scoring='accuracy')``
with weighted scoring and unweighted fitting.

Error handling: if `sample_weight` is required only in scoring and not in fit
of the sub-estimator the user should make explicit that it's not required
by the sub-estimator.

Case C
~~~~~~

Extend Case A to apply an unweighted univariate feature selector in a
``Pipeline``. This allows to check pipelines where only some steps
require a metadata.

Case D
~~~~~~

Different weights for scoring and for fitting in Case A.

Motivation: You can have groups used in a CV, which contains batches of data as groups,
and then an estimator which takes groups as sensitive attributes to a
fairness related model. Also in a third party library a estimator may have
the same name for a parameter, but with completely different semantics.

.. TODO: case involving props passed at test time, e.g. to pipe.transform
  to be considered later

Case E
~~~~~~

``LogisticRegression()`` with a weighted ``.score()`` method.

Solution sketches will import these definitions:

.. literalinclude:: defs.py

The following solution has emerged as the way to move forward,
yet others where considered. See :ref:`slep_006_other`.

Solution: Each consumer requests
--------------------------------

.. note::

  This solution was known as solution 4 during the discussions.

A meta-estimator provides only what each of its children requests.
A meta-estimator would also need to request, on behalf of its children,
any metadata that descendant consumers request. 

Each object that could receive metadata should have a method called
`get_metadata_request()` which returns a dict that specifies which
metadata is consumed by each of its methods (keys of this dictionary
are therefore method names, e.g. `fit`, `transform` etc.). `Group*CV`
splitters default to returning `{'split': 'groups'}`, for example.  Estimators
supporting weighted fitting may return `{}` by default, but have a
method called `request_sample_weight` which allows the user to specify
the requested `sample_weight` in each of its methods.
`make_scorer` accepts `request_metadata` as keyword parameter through
which user can specify what metadata is requested.

Advantages:

* This solution does not affect legacy estimators, since no metadata will be
  passed when a metadata request is not available.
* The implementation changes in meta-estimators is easy to provide via two
  helpers ``build_method_metadata_params(children, routing, metadata)``
  and ``build_router_metadata_request(children, routing)``.
* Easy to reconfigure what metadata an estimator gets in a grid search.
* Could make use of existing `**fit_params` syntax rather than introducing new
  `metadata` argument to `fit`.

Disadvantages:

* This will require modifying every estimator that may want any metadata,
  as well as all meta-estimators. Yet, this can be achieved with a mixin class
  to add metadata-request support to a legacy estimator.
* Aliasing is a bit confusing in this design, in that the consumer still
  accepts the fit param by its original name (e.g. `sample_weight`) even if it
  has a request that specifies a different key given to the router (e.g.
  `my_sample_weight`). This design has the advantage that the handling of
  metadata within a consumer is simple and unchanged; the complexity is in
  how it is forwarded to the sub-estimator by the meta-estimators. While
  it may be conceptually difficult for users to understand, this may be
  acceptable, as an advanced feature.
* For estimators to be cloned, this request information needs to be cloned with
  it. This implies that `clone` needs to be extended to explicitly copy
  request information.

Proposed public syntax:

* `BaseEstimator` will have a method `get_metadata_request`
* Estimators that can consume `sample_weight` will have a `request_sample_weight`
  method available via a mixin.
* `make_scorer` will have a `request_metadata` parameter to specify the requested
  metadata by the scorer.
* `get_metadata_request` will return a dict, whose keys are names of estimator
  methods (`fit`, `predict`, `transform` or `inverse_transform`) and values are
  dictionaries. These dictionaries map the input param names to requested
  metadata name. Example:

  >>> estimator.get_metadata_request()
  {'fit': {'my_sample_weight': {'sample_weight'}}, 'predict': {}, 'transform': {},
  'score': {}, 'split': {}, 'inverse_transform': {}}

* Methods like `request_sample_weight` will have a signature such as:
  `request_sample_weight(self, *, fit=None, score=None)` where fit keyword
  parameter can be `None`, `True`, `False` or a `str`.
  .. XXX explain
  .. accept either such a dict or a sequence `s` to be
  .. interpreted as the identity mapping for all elements in `s`
  .. (`{x: x for x in s}`). It will return `self` to enable chaining.

* `Group*` CV splitters will by default request the 'groups' prop, but its
  mapping can be changed with their `set_props_request` method.

Test cases:

.. literalinclude:: cases_opt4.py

Extensions and alternatives to the syntax considered while working on
:pr:`16079`:

* `set_prop_request` and `get_props_request` have lists of props requested
  **for each method** i.e. fit, score, transform, predict and perhaps others.
* `set_props_request` could be replaced by a method (or parameter) representing
  the routing of each prop that it consumes. For example, an estimator that
  consumes `sample_weight` would have a `request_sample_weight` method. One of
  the difficulties of this approach is automatically introducing
  `request_sample_weight` into classes inheriting from BaseEstimator without
  too much magic (e.g. meta-classes, which might be the simplest solution).

These are demonstrated together in the following:

.. literalinclude:: cases_opt4b.py

Proposal
--------

Having considered the above solutions, we propose:

* Solution 4 per :pr:`16079` which will be used to resolve further, specific
  details of the solution.
* Props will be known simply as Metadata.
* `**kw` syntax will be used to pass props by key.
* Default requests: A group splitter should return `['groups']` by default
  because requiring `groups` is essential to their functionality. In other
  cases, as for `sample_weight` where an estimator or scorer can operate
  without it, no default request should be assumed, and the consumer should
  have an explicit request for the prop. In the absence of an explicit request,
  or explicit request to disregard a prop: if a consumer is capable of
  requesting some prop by key `p`, and a prop called `p` is passed by the user,
  an error should be raised by the meta-estimator. This forces the user to be
  explicit about whether or not a consumer should be passed `p`.

TODO:

* if an estimator requests a prop, must it be not-null? Must it be provided or
  explicitly passed as None?

Backward compatibility
----------------------

Under this proposal, consumer behaviour will be backwards compatible, but
meta-estimators will change their routing behaviour.

By default, `sample_weight` will not be requested by estimators that support
it. This ensures that addition of `sample_weight` support to an estimator will
not change its behaviour.

During a deprecation period, fit_params will be handled dually: Keys that are
requested will be passed through the new request mechanism, while keys that are
not known will be routed using legacy mechanisms. At completion of the
deprecation period, the legacy handling will cease.

Similarly, during a deprecation period, `fit_params` in GridSearchCV and
related utilities will be routed to the estimator's `fit` by default, per
incumbent behaviour. After the deprecation period, an error will be raised for
any params not explicitly requested.

Grouped cross validation splitters will request `groups` since they were
previously unusable in a nested cross validation context, so this should not
often create backwards incompatibilities, except perhaps where a fit param
named `groups` served another purpose.

Discussion
----------

One benefit of the explicitness in Solution 4 is that even if it makes use of
`**kw` arguments, it does not preclude keywords arguments serving other
purposes in addition.  That is, in addition to requesting sample props, a
future proposal could allow estimators to request feature metadata or other
keys.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.
.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
