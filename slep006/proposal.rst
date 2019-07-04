.. _slep_006:

================================
Routing sample-aligned meta-data 
================================

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
* (likley out of scope) passing sample properties to non-fit methods, for
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
    consumer, while either consuming props itself or delivering props to
    more than one immediate destination.

History
-------

This version was drafted after a discussion of the issue and potential
solutions at the February 2019 development sprint in Paris.

Supersedes `SLEP004
<https://github.com/scikit-learn/enhancement_proposals/tree/master/slep004>`_
with greater depth of desiderata and options.

Related issues and pull requests include:

-  `Consistent API for attaching properties to samples
   #4497 <https://github.com/scikit-learn/scikit-learn/issues/4497>`__
-  `Acceptance of sample\_weights in pipeline.score
   #7723 <https://github.com/scikit-learn/scikit-learn/pull/7723>`__
-  `Establish global error state like np.seterr
   #4660 <https://github.com/scikit-learn/scikit-learn/issues/4660>`__
-  `Should cross-validation scoring take sample-weights into account?
   #4632 <https://github.com/scikit-learn/scikit-learn/issues/4632>`__
-  `Sample properties
   #4696 <https://github.com/scikit-learn/scikit-learn/issues/4696>`__

TODO: more. ndawe's work.

Desiderata
----------

We will consider the following attributes to develop and compare solutions:

Usability
   Can the use cases be achieved in succinct, readable code? Can common use
   cases be achieved with a simple recipe copy-pasted from a QA forum?
Brittleness
   If a property is being routed through a Pipeline, does changing the
   structure of the pipeline (e.g. adding a layer of nesting) require rewriting
   other code?
Error handling
   If the user mistypes the name of a sample property, will an appropriate
   exception be raised?
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
   between weighting and resampling)?

Keyword arguments vs. a single argument
---------------------------------------

Currently, sample properties are provided as keyword arguments to a `fit`
method. In redeveloping sample properties, we can instead accept a single
parameter (named `props` or `sample_props` or `etc`, for example) which maps
string keys to arrays of the same length (a "DataFrame-like").

Keyword arguments::

    >>> gs.fit(X, y, groups=groups, sample_weight=sample_weight)

Single argument::

    >>> gs.fit(X, y, prop={'groups': groups, 'weight': weight})

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

Extend Case A to apply an arbitrary SLEP005 resampler in a pipeline, which
rewrites ``sample_weight`` and ``groups``.

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

Issues (TODO: clean)

* Error handling: if a prop is optional in a consumer, no error will be
  raised for misspelling. An introspection API might change this, allowing a
  meta-estimator to check if all props are to be used.
* Forwards compatibility: newly supporting a prop in a consumer will change
  behaviour. Other than a ChangedBehaviorWarning I don't see any way around
  this.
* Introspection: not inherently supported. Would need an API like
  ``get_prop_support(names: List[str]) -> Dict[str, Literal["supported", "required", "ignored"]]``.

Solution 2: specify routes at call
----------------------------------


Solution 3: specify routes on metaestimators
--------------------------------------------

The metaestimators


Backward compatibility
----------------------

Discussion
----------

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.
.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_

