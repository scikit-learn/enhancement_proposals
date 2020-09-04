.. _slep_006:

==========================================
SLEP006: Routing sample-aligned meta-data 
==========================================

:Author: Joel Nothman
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

Desirable features we do not currently support include:

* passing sample properties (e.g. `sample_weight`) to a scorer used in
  cross-validation
* passing sample properties (e.g. `groups`) to a CV splitter in nested cross
  validation
* (maybe in scope) passing sample properties (e.g. `sample_weight`) to some
  scorers and not others in a multi-metric cross-validation setup
* (likely out of scope) passing sample properties to non-fit methods, for
  instance to index grouped samples that are to be treated as a single sequence
  in prediction.

Definitions
-----------

consumer
    An estimator, scorer, splitter, etc., that receives and can make use of
    one or more passed props.
key
    A label passed along with sample prop data to indicate how it should be
    interpreted (e.g. "weight").
router
    An estimator or function that passes props on to some other router or
    consumer, potentially selecting which props to pass to which destination,
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

Usability
   Can the use cases be achieved in succinct, readable code? Can common use
   cases be achieved with a simple recipe copy-pasted from a QA forum?
Brittleness
   If a property is being routed through a Pipeline, does changing the
   structure of the pipeline (e.g. adding a layer of nesting) require rewriting
   other code?
Error handling
   If the user mistypes the name of a sample property, or misspecifies how it
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
   behaviour radically when sample_weight is implemented on some estimator)
Introspection
   If sensible to do so (e.g. for improved efficiency), can a
   meta-estimator identify whether its base estimator (recursively) would
   handle some particular sample property (e.g. so a meta-estimator can choose
   between weighting and resampling, or for automated invariance testing)?

Keyword arguments vs. a single argument
---------------------------------------

Currently, sample properties are provided as keyword arguments to a `fit`
method. In redeveloping sample properties, we can instead accept a single
parameter (named `props` or `sample_props`, for example) which maps
string keys to arrays of the same length (a "DataFrame-like").

Keyword arguments::

    >>> gs.fit(X, y, groups=groups, sample_weight=sample_weight)

Single argument::

    >>> gs.fit(X, y, props={'groups': groups, 'weight': weight})

While drafting this document, we will assume the latter notation for clarity.

Advantages of multiple keyword arguments:

* succinct
* possible to maintain backwards compatible support for sample_weight, etc.
* we do not need to handle cases for whether or not some estimator expects a
  `props` argument.

Advantages of a single argument:

* we are able to consider kwargs to `fit` that are not sample-aligned, so that
  we can add further functionality (some that have been proposed:
  `with_warm_start`, `feature_names_in`, `feature_meta`).
* we are able to redefine the default routing of weights etc. without being
  concerned by backwards compatibility.
* we can consider the use of keys that are not limited to strings or valid
  identifiers (and hence are not limited to using ``_`` as a delimiter).

Test case setup
---------------

Case A
~~~~~~

Cross-validate a ``LogisticRegressionCV(cv=GroupKFold(), scoring='accuracy')``
with weighted scoring and weighted fitting.

Error handling: what would happen if the user misspelled `sample_weight` as
`sample_eight`?

Case B
~~~~~~

Cross-validate a ``LogisticRegressionCV(cv=GroupKFold(), scoring='accuracy')``
with weighted scoring and unweighted fitting.

Case C
~~~~~~

Extend Case A to apply an unweighted univariate feature selector in a
``Pipeline``.

Case D
~~~~~~

Different weights for scoring and for fitting in Case A.

TODO: case involving props passed at test time, e.g. to pipe.transform (???).
TODO: case involving score() method, e.g. not specifying scoring in
cross_val_score when wrapping an estimator with weighted score func ...

Solution sketches will import these definitions:

.. literalinclude:: defs.py

Status quo solution 0a: additional feature
------------------------------------------

Without changing scikit-learn, the following hack can be used:

Additional numeric features representing sample props can be appended to the
data and passed around, being handled specially in each consumer of features
or sample props.

.. literalinclude:: cases_opt0a.py

Status quo solution 0b: Pandas Index and global resources
---------------------------------------------------------

Without changing scikit-learn, the following hack can be used:

If `y` is represented with a Pandas datatype, then its index can be used to
access required elements from props stored in a global namespace (or otherwise
made available to the estimator before fitting). This is possible everywhere
that a ground-truth `y` is passed, including fit, split, score, and metrics.
A similar solution with `X` is also possible (except for metrics), if all
Pipeline components retain the original Pandas Index.

Issues:

* use of global data source
* requires Pandas data types and indices to be maintained

.. literalinclude:: cases_opt0b.py

Solution 1: Pass everything
---------------------------

This proposal passes all props to all consumers (estimators, splitters,
scorers, etc). The consumer would optionally use props it is familiar with by
name and disregard other props.

We may consider providing syntax for the user to control the interpretation of
incoming props:

* to require that some prop is provided (for an estimator where that prop is
  otherwise optional)
* to disregard some provided prop
* to treat a particular prop key as having a certain meaning (e.g. locally
  interpreting 'scoring_sample_weight' as 'sample_weight').

These constraints would be checked by calling a helper at the consumer.

Issues:

* Error handling: if a key is optional in a consumer, no error will be
  raised for misspelling. An introspection API might change this, allowing a
  user or meta-estimator to check if all keys passed are to be used in at least
  one consumer.
* Forwards compatibility: newly supporting a prop key in a consumer will change
  behaviour. Other than a ChangedBehaviorWarning, I don't see any way around
  this.
* Introspection: not inherently supported. Would need an API like
  ``get_prop_support(names: List[str]) -> Dict[str, Literal["supported", "required", "ignored"]]``.

In short, this is a simple solution, but prone to risk.

.. literalinclude:: cases_opt1.py


Solution 2: Specify routes at call
----------------------------------

Similar to the legacy behavior of fit parameters in
:class:`sklearn.pipeline.Pipeline`, this requires the user to specify the
path for each "prop" to follow when calling `fit`.  For example, to pass
a prop named 'weights' to a step named 'spam' in a Pipeline, you might use
`my_pipe.fit(X, y, props={'spam__weights': my_weights})`.

SLEP004's syntax to override the common routing scheme falls under this
solution.

Advantages:

* Very explicit and robust to misspellings.

Issues:

* The user needs to know the nested internal structure, or it is easy to fail
  to pass a prop to a specific estimator.
* A corollary is that prop keys need changing when the developer modifies their
  estimator structure (see case C).
* This gets especially tricky or impossible where the available routes
  change mid-fit, such as where a grid search considers estimators with
  different   structures.
* We would need to find a different solution for :issue:`2630` where a Pipeline
  could not be the base estimator of AdaBoost because AdaBoost expects the base
  estimator to accept a fit param keyed 'sample_weight'.
* This may not work if a meta-estimator were to have the role of changing a
  prop, e.g. a meta-estimator that passes `sample_weight` corresponding to
  balanced classes onto its base estimator.  The meta-estimator would need a
  list of destinations to pass modified props to, or a list of keys to modify.
* We would need to develop naming conventions for different routes, which may
  be more complicated than the current conventions; while a GridSearchCV
  wrapping a Pipeline currently takes parameters with keys like
  `{step_name}__{prop_name}`, this explicit routing, and conflict with
  GridSearchCV routing destinations, implies keys like
  `estimator__{step_name}__{prop_name}`.

.. literalinclude:: cases_opt2.py


Solution 3: Specify routes on metaestimators
--------------------------------------------

Each meta-estimator is given a routing specification which it must follow in
passing only the required parameters to each of its children. In this context,
a GridSearchCV has children including `estimator`, `cv` and (each element of)
`scoring`.

Pull request :pr:`9566` and its extension in :pr:`15425` are partial
implementations of this approach.

A major benefit of this approach is that it may allow only prop routing
meta-estimators to be modified, not prop consumers.

All consumers would be required to check that 

Issues:

* Routing may be hard to get one's head around, especially since the prop
  support belongs to the child estimator but the parent is responsible for the
  routing.
* Need to design an API for specifying routings.
* As in Solution 2, each local destination for routing props needs to be given
  a name.
* Every router along the route will need consistent instructions to pass a
  specific prop to a consumer. If the prop is optional in the consumer, routing
  failures may be hard to identify and debug.
* For estimators to be cloned, this routing information needs to be cloned with
  it. This implies one of: the routing information be stored as a constructor
  parameter; or `clone` is extended to explicitly copy routing information.

Possible public syntax:

Each meta-estimator has a `prop_routing` parameter to encode local routing
rules, and a set of named children which it routes to. In :pr:`9566`, the
`prop_routing` entry for each child may be a white list or black list of
named keys passed to the meta-estimator.

.. literalinclude:: cases_opt3.py


Solution 4: Each child requests
-------------------------------

Here the meta-estimator provides only what each of its children requests.
The meta-estimator would also need to request, on behalf of its children,
any prop that descendant consumers require. 

Each object that could receive props would have a method like
`get_prop_request()` which would return a list of prop names (or perhaps a
mapping for more sophisticated use-cases). Group* CV splitters would default to
returning `['groups']`, for example.  Estimators supporting weighted fitting
may return `[]` by default, but may have a parameter `request_props` which
may be set to `['weight']` if weight is sought, or perhaps just boolean
parameter `request_weight`. `make_scorer` would have a similar mechanism for
enabling weighted scoring.

Advantages:

* This will not need to affect legacy estimators, since no props will be
  passed when a props request is not available.
* This does not require defining a new syntax for routing.
* The implementation changes in meta-estimators may be easy to provide via a
  helper or two (perhaps even `call_with_props(method, target, props)`).
* Easy to reconfigure what props an estimator gets in a grid search.
* Could make use of existing `**fit_params` syntax rather than introducing new
  `props` argument to `fit`.

Disadvantages:

* This will require modifying every estimator that may want props, as well as
  all meta-estimators. We could provide a mixin or similar to add prop-request
  support to a legacy estimator; or `BaseEstimator` could have a
  `set_props_request` method (instead of the `request_props` constructor
  parameter approach) such that all legacy base estimators are
  automatically equipped.
* Aliasing is a bit confusing in this design, in that the consumer still
  accepts the fit param by its original name (e.g. `sample_weight`) even if it
  has a request that specifies a different key given to the router (e.g.
  `fit_sample_weight`). This design has the advantage that the handling of
  props within a consumer is simple and unchanged; the complexity is in
  how it is forwarded the data by the router, but it may be conceptually
  difficult for users to understand. (This may be acceptable, as an advanced
  feature.)
* For estimators to be cloned, this request information needs to be cloned with
  it. This implies one of: the request information be stored as a constructor
  parameter; or `clone` is extended to explicitly copy request information.

Possible public syntax:

* `BaseEstimator` will have methods `set_props_request` and `get_props_request`
* `make_scorer` will have a `request_props` parameter to set props required by
  the scorer.
* `get_props_request` will return a dict. It maps the key that the user
  passes to the key that the estimator expects.
* `set_props_request` will accept either such a dict or a sequence `s` to be
  interpreted as the identity mapping for all elements in `s`
  (`{x: x for x in s}`). It will return `self` to enable chaining.
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

Naming
------

"Sample props" has become a name understood internally to the Scikit-learn
development team. For ongoing usage we have several choices for naming:

* Sample meta
* Sample properties
* Sample props
* Sample extra

Proposal
--------

Having considered the above solutions, we propose:

* Solution 4 per :pr:`16079` which will be used to resolve further, specific
  details of the solution.
* Props will be known simply as Metadata.
* `**kw` syntax will be used to pass props by key.

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
