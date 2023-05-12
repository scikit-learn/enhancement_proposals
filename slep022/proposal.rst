.. _slep_022:

======================================
SLEP022: Fixing randomness ambiguities
======================================

:Author: Needs a champion
:Status: Under review
:Type: Standards Track
:Created: 2020-09-01

.. warning::

    Update from 2023: This SLEP was written a long time ago, when the
    understanding was that `RandomState` instances are **shared** across
    clones. `It turns out it's not the case
    <https://github.com/scikit-learn/scikit-learn/issues/26148>`_ and the
    instances are instead copied. So a few of the claims made in this SLEP are
    wrong, and may be slightly confusing. What this SLEP claims is happening for
    `RandomState` instances is in fact happening only when passing `None`, and
    `RandomState` instances behave mostly like `int` in CV procedures.

    Nonetheless, a lot of its content is still relevant. At the very least, the
    :ref:`key_use_cases` section may help framing discussions.

    Note: this SLEP is a complete re-write of the abandonned `SLEP 11
    <https://github.com/scikit-learn/enhancement_proposals/pull/24>`_ and
    proposes a different solution.



Abstract
========

This SLEP aims at fixing various issues related to how scikit-learn handles
randomness in estimators and Cross-Validation (CV) splitters. The
`random_state` design makes estimators and splitters stateful across calls to
`fit` and `split`, which induces a complex mental model for developers and
users. Various issues are:

  - Implicit behavior of CV procedures and meta-estimators: the estimator's
    `random_state` parameter implicitly (but significantly) influences the
    behaviour of seemingly unrelated procedures like `cross_val_score`, or
    meta-estimators like `RFE` and `SequentialFeatureSelection`. To understand
    these subtleties, users need to know inner details of these CV procedures
    and meta-estimators: namely, they need to know *when* `fit` and `clone`
    are called.
  - Similarly, users are required to know how many times `split` is called in
    a CV procedure (e.g `GridSearchCV.fit`), or there is a risk that they may
    make unsound performance comparison.
  - hard-to-find bugs, due to the insiduous ways in which `random_state`
    affects estimators and splitters.

These issues are described in more details in a dedicated section below. This
SLEP is *not* about:

- what the default value of `random_state` should be
- whether we should allow RandomState instances and None to be passed.

**The proposed solution is to**:

- Make estimators and splitter stateless, by drawing a random seed in
  `__init__`. This seed is used as the sole source of RNG in `fit` and in
  `split`. Neither `fit` nor `split` need to be changed. This will
  allow users and developers to reason about `random_state` with a simpler
  mental model.
- Let users explicitly choose the strategy executed by CV procedures and
  meta-estimators, by introducing new parameters in those procedures, where
  relevant. A candidate parameter name is `use_exact_clones` (name open for
  discussion).
  
  This will **explicitly** expose the choice of the CV and meta-estimator
  strategy to users, instead of **implicitly** relying on the type of the
  estimator's `random_state`. Moreover, unlike in the legacy design, all
  strategies will be supported by any estimator, no matter the type of the
  `random_state` attribute.

.. note::
    The *legacy* design refers to the design as used in the current
    scikit-learn master branch, i.e. `0.24.dev`.

.. note::
    Since passing `random_state=None` is equivalent to passing the global
    `RandomState` instance from `numpy`
    (`random_state=np.random.mtrand._rand`), we will not explicitly mention
    None in most cases, but everything that applies to instances also applies
    to using None.

Issues with the legacy design
=============================

Introduction: a complex mental model
------------------------------------

The rules that dictate how `random_state` impacts the behaviour of estimators
and CV splitters are seemingly simple: use an int and get the same results
across calls, use an instance and get different results. Things get however
much more complicated when these calls to `fit` and `split` are embedded
within other procedures like CV procedures or meta-estimators.

Users and core devs alike still seem to have a hard time figuring it out.
Quoting Andreas Müller from `#14042
<https://github.com/scikit-learn/scikit-learn/issues/14042>`_:

    The current design of random_state is often hard to understand and
    confusing for users. I think we should rethink how random_state works.

As another empirical evidence, the fact the CV splitters may yield different
splits across calls recently came as a surprise to a long-time core dev.

At the very least, it should be noted that prior to `#9517
<https://github.com/scikit-learn/scikit-learn/pull/9517/>`_ (merged in 2018)
and `#18363 <https://github.com/scikit-learn/scikit-learn/pull/18363>`_ (not
yet merged as of September 2020), most subtleties related to `random_state`
were largely under-documented, and it is likely that a lot of users are
actually oblivious to most pitfalls.

The next sections describe:

- Issues that illustrate how `random_state` implictly affects CV procedures
  and meta-estimators in insiduous ways.
- Examples of bugs that `random_state` has caused, which are another sign
  that the legacy design might be too complex, even for core devs.

Before reading these sections, please make sure to read `#18363
<https://github.com/scikit-learn/scikit-learn/pull/18363>`_ which provides
more context, and illustrates some use-cases that will be used here.

.. _estimator_issues:

Estimators: CV procedures and meta-estimators behaviour is implicitly dictated by the base estimator's `random_state`
---------------------------------------------------------------------------------------------------------------------

As described in more details in `#18363
<https://github.com/scikit-learn/scikit-learn/pull/18363>`_, the following
lines run two very different cross-validation procedures::

    cross_val_score(RandomForestClassifier(random_state=0), X, y)
    cross_val_score(RandomForestClassifier(random_state=np.RandomState(0)), X, y)

In the first one, the estimator RNG is the same on each fold and the RF
selects the same subset of features across folds. In the second one, the
estimator RNG varies across folds and the random subset of selected features
will vary. In other words, the CV strategy that `cross_val_score` uses
implicitly depends on the type of the estimators' `random_state` parameter.
This is unexpected, since the behaviour of `cross_val_predict` should
preferably be determined by parameters passed to `cross_val_predict`.

The same is true for any procedure that performs cross-validation (manual CV,
`GridSearchCV`, etc.). Things are particularly ambiguous in `GridSearchCV`:
when a RandomState instance is used, a new RNG is used on each fold, **but
also for each candidate**::

    fold 0: use different RNGs across candidates
    fold 1: use different RNGs across candidates (different RNGs from fold 0)
    fold 2: use different RNGs across candidates (different RNGs from folds 0 and 1)
    etc...

Users might actually expect that the RNG would be different on each fold, but
still constant across candidates, i.e. something like::

    fold 0: use same RNG for all candidates
    fold 1: use same RNG for all candidates (different RNG from fold 0)
    fold 2: use same RNG for all candidates (different RNG from folds 0 and 1)
    etc...

.. note::
    This strategy is in fact not even supported right now: neither integers,
    RandomState instances or None can achieve this.

Unfortunately, there is no way for users to figure out what strategy is used
until they look at the code. It is not just a documentation problem. The core
problem here is that **the behaviour of the CV procedure is implicitly
dictated by the estimator's** `random_state`.

There are similar issues in meta-estimators, like `RFE` or
`SequentialFeatureSelection`: these are iterative feature selection
algorithms that will use either *exact* or *statistical* clones at each
iteration, depending on the estimator's `random_state`. Exact and statistical
clones lead to two significantly different strategies. Here again, the
behavior of these meta-estimators **is only implicitly dictated by the
estimator's** `random_state`.

In addition, since the sub-estimator's `random_state` type dictates the
strategy, users are bound to one single strategy once the estimator has been
created: it is for example impossible for an estimator to use a different RNG
across folds if that estimator was initialized with an integer.

It is unlikely that users have a perfect understanding of these subtleties.
For users to actually understand how `random_state` impacts the CV procedures
and meta-estimators, they actually need to know inner details of these: they
need to know where and when `fit` is called, and also when `clone` is called.

There is a very similar problem with CV splitters as described in the next
section.

.. _cv_splitters_issues:

CV Splitters: users need to know inner details of CV procedures to avoid erroneous performance comparison
---------------------------------------------------------------------------------------------------------

CV splitters yield different splits every time `split` is called if their
`random_state` is a RandomState instance. This means that the following code
doesn't allow fold-to-fold comparison of scores::

    rng = np.random.RandomState(0)
    cv = KFold(shuffle=True, random_state=rng)
    estimators = [...]  # the estimators you want to compare
    scores = {
                est: cross_val_score(est, X, y, cv=cv)
                for est in estimators
    }

Users might not realize it, but **the estimators will be evaluated on
different splits**, even though they think they've set the random state by
passing a carefuly crafted instance. This is because `cv.split` was called
multiple times, yet these calls were hidden inside of `cross_val_score`. On
top of impossible fold-to-fold comparison, comparing the average scores is
also not ideal if the number of folds or samples is small.

As a consequence, before users can safely report score comparisons, **they
need to know how many times** `split` **is called**, which should just be an
implementation detail. While the above example is already error-prone, things
get harder in more complex tools like `GridSearchCV`: how are users supposed
to know that `split` is called only once in `GridSearchCV.fit`?

.. note::
    This implementation detail about `GridSearchCV.fit` is in fact
    documented, but only at the very end of the cross-validation `User Guide
    <https://scikit-learn.org/stable/modules/cross_validation.html#a-note-on-shuffling>`_.
    It is not documented where it shoud be, that is, in the hyper-parameter
    tuning User Guide or in the docstrings.

.. note::
    In `#18363 <https://github.com/scikit-learn/scikit-learn/pull/18363>`_,
    we recommend users to use integers for CV splitters' `random_state`,
    effectively making them stateless.

.. note::
    Fixing how `random_state` is handled in the splitters is one of the
    entries in the `Roadmap <https://scikit-learn.org/dev/roadmap.html>`_.

.. _bugs:

Bugs
----

Many bugs have happened over the years because of RandomState instances and
None. Quoting Andreas Müller from `#14042
<https://github.com/scikit-learn/scikit-learn/issues/14042>`_:

    There have been countless bugs because of this

("*This*" = RandomState instances and the implied statefulness of the
estimators).

Bugs caused by estimators statefulness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These bugs are often hard to find, and some of them are actual data leaks,
e.g. `#14034 <https://github.com/scikit-learn/scikit-learn/issues/14034>`_.

They arise because the estimators are stateful across calls to `fit`. Fixing
them usually involves forcing the estimator to be (at least partially)
stateless. A classical bug that happened multiple times is that the
validation set may differ across calls to `fit` in a warm-start + early
stopping context. For example, `this fix
<https://github.com/scikit-learn/scikit-learn/pull/14999>`_ is to draw a
random seed once and to re-use that seed for data splitting when
early-stopping + warm start is used. It is *not* an obvious bug, nor an
obvious fix.

Making estimators stateless across calls to `fit` would prevent such bugs to
happen, and would keep the code-base cleaner.

Bugs caused by splitters statefulness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`#18431 <https://github.com/scikit-learn/scikit-learn/pull/18431>`_ is a bug
introduced in `SequentialFeatureSelection` that perfectly illustrates the
previous section :ref:`cv_splitters_issues`. The bug was that splitter
statefulness would lead to comparing average scores of candidates that have
been evaluated on different splits. Here again, the fix is to enforce
statelessness of the splitter, e.g.
`KFolds(5, shuffle=True, random_state=None)` is forbidden.

.. note::
    This bug was introduced by Nicolas Hug, who is this SLEP's author: it's
    very easy to let these bugs sneak in, even when you're trying hard not
    to.

Other potential bugs can happen in the parameter search estimators. When a
third-party library wants to implement its own parameter search strategy, it
needs to subclass `BaseSearchCV` and call a built-in function
`evaluate_candidates(candidates)` once, or multiple times.
`evaluate_candidates` internally calls `split` once. If
`evaluate_candidates` is called more than once, this means that **the
candidate parameters are evaluated on different splits each time**.

This is a quite subtle issue that third-party developers might easily
overlook. Some core devs (Joel Nothman and Nicolas Hug) kept forgetting and
re-discovering this issue over and over in the `Successive Halving PR 
<https://github.com/scikit-learn/scikit-learn/pull/13900>`_.

At the very least, this makes fold-to-fold comparison between candidates
impossible whenever the search strategy calls `evaluate_candidates` more
than once. This can however cause bigger bugs in other scenarios, e.g. if we
implement successive halving + warm start (details ommitted here, you may
refer to `this issue
<https://github.com/scikit-learn/scikit-learn/issues/15125>`_).

.. note::
    In order to prevent any potential future bug and to prevent users from
    making erroneous comparisons between the candidates scores, the
    `Successive Halving implementation
    <https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV>`_
    forbids users from using stateful splitters, just like
    `SequentialFeatureSelection` (see the note in the docstring for the `cv`
    parameter).

Other issues
------------

Fit idempotency isn't respected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quoting our `Developer Guidelines
<https://scikit-learn.org/stable/developers/develop.html#fitting>`_:

    When fit is called, any previous call to fit should be ignored.

This means that ideally, calling `est.fit(X, y)` should yield the same model
twice. We have a check for that in the `check_estimator()` suite:
`check_fit_idempotent()`. Clearly, this fit-idempotency property is violated
when RandomState instances are used.

`clone` 's behaviour is implicit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Much like CV procedures and meta-estimators, what `clone` returns implicitly
depends on the estimators' `random_state`: it may return an exact clone or a
statistical clone. Statistical clones share a common RandomState instance and
thus are inter-dependent, as detailed in `#18363
<https://github.com/scikit-learn/scikit-learn/pull/18363>`_. This makes
debugging harder, among other things. Until `#18363
<https://github.com/scikit-learn/scikit-learn/pull/18363>`_, the distinction
between exact and statistical clones had never been documented and was up to
users to figure out.

.. note::
    While less user-facing, this issue is actually part of the root cause for
    the aforementioned issues related to estimators in CV procedures and in
    meta-estimators.

.. _key_use_cases:

Key use-cases and requirements
==============================

There are a few use-cases that are made possible by the legacy design and
that we will want to keep supporting in the new design. We will refer to
these use-cases in the rest of the document:

- A. Allow for consistent results across executions. This is a natural
  requirement for any implementation: we want users to be able to get
  consistently reproducible results across multiple program executions. This
  is currently supported by removing any use of `random_state=None`.
- B. Allow for maximum variability and different results across executions.
  This is currently supported by passing `random_state=None`: multiple
  program executions yield different results each time.
- C. CV procedures where the estimator's RNG is exactly the same on each
  fold. This is useful when one wants to make sure that a given seed will
  work well on new data (whether users should actually do this is
  debatable...). This is currently supported by passing an int to the
  `random_state` parameter of the estimator.
- D. CV procedures where the estimator's RNG is different for each fold. This
  is useful to increase the statistical significance of CV results. This is
  currently supported by passing a RandomState instance or None to the
  `random_state` parameter of the estimator.
- E. Obtain *strict* clones, i.e. clones that will yield exactly the same
  results when called on the same data. This is currently supported by
  calling `clone` if the estimator's `random_state` is an int.
- F. Obtain *statistical* clones, i.e. estimators
  that are identical but that will yield different results, even when called
  on the same data. This is currently supported by calling `clone` if
  the estimator's `random_state` is an instance or None.
- G. Easily obtain different splits with the same characteristics from a
  splitter. By "same characteristics", we mean same number of splits, same
  choice of stratification, etc. This is useful to implement e.g.
  boostrapping. This is currently supported by calling `split` multiple times
  on the same `cv` instance, if `cv.random_state` is an instance, or None.
  This can also be achieved by just creating multiple splitter instances with
  different `random_state` values (which could be integers).

.. note::
    C and E are very related: in `cross_val_score`, C is supported by
    creating strict clones (E). Similarly, D is supported by creating
    statistical clones (F).

    In the legacy design, C and D are mutually exclusive: a given estimator
    can only do C and not D, or only D and not C. Also, a given estimator can
    only produce strict clones or only statistical clones, but not both. In
    the proposed design, all estimators will support both C and D. Similarly,
    strict and statistical clones can be obtained from any estimator.

Proposed Solution
=================

.. note::
    This proposed solution is a work in progress and there is room for
    improvement. Feel free to suggest any.

We want to make estimators and splitter stateless, while also avoiding the
ambiguity of CV procedures and meta-estimators. We also want to keep
supporting all the aforementioned use-cases in some way.

.. note::
    A toy implementation of the proposed solution with illustration snippets
    is available in `this notebook
    <https://nbviewer.jupyter.org/gist/NicolasHug/2db607b01482988fa549eb2c8770f79f>`_.

The proposed solution is to sample a seed from `random_state` at `__init__`
in estimators and splitters::

    def _sample_seed(random_state):
        # sample a random seed to be stored as the random_state attribute
        # ints are passed-through
        if isinstance(random_state, int):
            return random_state
        else:
            return check_random_state(random_state).randint(0, 2**32)

    class MyEstimator(BaseEstimator):
        def __init__(self, random_state=None):
            self.random_state = _sample_seed(random_state)
        
        def set_params(self, random_state=None):
            self.random_state = _sample_seed(random_state)
          
    class MySplitter(BaseSplitter):
        def __init__(self, random_state=None):
            self.random_state = _sample_seed(random_state)

`fit`, `split`, and `get_params` are unchanged.

In order to **explicitly** support use-cases C and D, CV procedures like
`cross_val_score` should be updated with a new `use_exact_clones` parameter::

    def _check_statistical_clone_possible(est):
      if 'random_state' not in est.get_params():
          raise ValueError("This estimator isn't random and can only have exact clones")

    def cross_val_score(est, X, y, cv, use_exact_clones=True):
        # use_exact_clones:
        # - if True, the same estimator RNG is used on each fold (use-case C) 
        # - if False, the different estimator RNGs are used on each fold (use-case D) 
        
        if use_exact_clones:
            statistical = False
        else:
            # need a local RNG so that clones have different random_state attributes
            _check_statistical_clone_possible(est)
            statistical = np.random.RandomState(est.random_state)
            
        return [
            clone(est, statistical=statistical)
            .fit(X[train], y[train])
            .score(X[test], y[test])
            for train, test in cv.split(X, y)
        ]

.. note::
    The name `use_exact_clones` is just a placeholder for now, and the final
    name is up for discussion. A more descriptive name for `cross_val_score`
    could be e.g. `estimator_randomness={'constant', 'splitwise'}`. The name
    doesn't have to be the same throughout all CV procedures, and itshould
    accurately describe the alternative strategies that are possible.

Meta-estimators should be updated in a similar fashion to make their two
alternative behaviors explicit.

As can be seen from the above snippet, the `clone` function needs to be
updated so that one can explicitly request exact or statistical clones
(use-cases E and F)::

    def clone(est, statistical=False):
        # Return a strict clone or a statistical clone.
        
        # statistical parameter can be:
        # - False: a strict clone is returned
        # - True: a statistical clone is returned. Its RNG is seeded from `est`
        # - None, int, or RandomState instance: a statistical clone is returned.
        #   Its RNG is seeded from `statistical`. This is useful to
        #   create multiple statistical clones that don't have the same RNG
        
        params = est.get_params()
        
        if statistical is not False:
            # A statistical clone is a clone with a different random_state attribute
            _check_statistical_clone_possible(est)
            rng = params['random_state'] if statistical is True else statistical
            params['random_state'] = _sample_seed(check_random_state(rng))
            
        return est.__class__(**params)

Note how one can pass a RandomState instance as the `statistical` parameter,
in order to obtain a sequence of estimators that have different RNGs. This is
used in particular in the above `cross_val_score`.

Use-cases support
-----------------

Use-cases A and B are supported just like before.

Use-cases C, D, E, F are explicitly supported *and* can be achieved by any
estimator, no matter its `random_state`. The legacy design can only
(implicitly) support either C and E or D and F.

Use-case G can be explicitly supported by creating multiple CV instances,
each with a different `random_state`::

    rng = np.RandomState(0)
    cvs = [KFold(random_state=rng) for _ in range(n_bootstrap_iterations)]
  
CV instances are extremely cheap to create and to store. Alternatively, we
can introduce the notion of statistical clone for splitters, and let `clone`
support splitters as well. This is however more involved.

Advantages
----------

- Users do not need to know the internals of CV procedures or meta estimators
  anymore. Any potential ambiguity is now made explicit and exposed to them
  by a new parameter, which will also make documentation easier.
  *Removing ambiguity is the main point of this SLEP*.

- The mental model is simpler: no matter what is passed as `random_state`
  (int, RandomState instances, or None), results are constant across calls to
  `fit` and `split`. The RandomState instance is mutated **once** (and only
  once) in `__init__`. Bugs will be less likely to sneak in.

- Estimators and splitters are stateless, and `fit` is now properly
  idempotent.

- Users now have explicit control on the CV procedures and meta-estimators,
  instead of implicitly relying on the estimator's `random_state`. These
  procedures are not ambiguous anymore.

- Since CV splitters always yield the same splits, the chances of performing
  erroneous score comparison is limited.

- `fit`, `split`, and `get_params` are unchanged.

Drawbacks
---------

- We break our convention that `__init__` should only ever store attributes,
  as they are passed in (for integers, this convention is still respected).
  Note however that the reason we have this convention is that we want the
  semantic of `__init__` and `set_params` to be the same. **This is still
  respected here.** So this isn't really an issue.

- CV procedures and meta-estimators must be updated. There is however no way
  around this, if we want to explicitly expose the different possible
  strategies.

Backward compatibility and possible roadmap for adoption
--------------------------------------------------------

There are two main changes proposed here:

1. Making estimators and splitters stateless
2. Introducing the `use_exact_clones` (or alternative names) parameter to CV
   procedures and meta-estimators.

A possible deprecation path for 1) would be to warn the user the second time
`fit` (or `split`) is called, when `random_state` is an instance. There is no
need for a warning if `random_state` is an integer since estimators and
splitters are already stateless in this case. The warning would suggest users
to explicitly create a statistical clone (using `clone`) instead of calling
`fit` twice on the same instance. For splitters, we would just suggest the
creation of a new instance.

However, making estimators stateless without supporting item 2) would mean
dropping support for use-case D. So item 2) should be implemented before item
1).

Item 2) can be implemented in a backward-compatible fashion by introducing
the `use_exact_clones` parameter where relevant with a default 'warn', which
would tell users that they should be setting this parameter explictly from
now on::

    def cross_val_score(..., use_exact_clones='warn'):
        if use_exact_clones == 'warn':
            warn("The use_exact_clones parameter was introduced in ..."
                 "you should explicitly set it to True or False...")
            # proceed as before in a backward compatible way
        elif use_exact_clones is True:
            # build exact clones (probably by sampling a seed from the estimator's random_state)
        else:
            # build statistical clones (probably by creating a local RandomState instance)

However if 1) isn't implemented yet, this is not fully satisfactory: if
estimators are still stateful, this means for example that the seed drawn in
the `elif` block would be different across calls to `cross_val_score`, since
the estimator's `random_state` would have been mutated. This is not
desirable: once 1) is implemented and the estimators become stateless,
multiple calls to `cross_val_score` would result in consistent results (as
expected).

As a result, 1) must be implemented before 2). But since 2) must also be
implemented before 1)... we have a chicken and egg problem.

The path with the least amount of friction would likely be to implement both
1) and 2) at the same time, and accept the fact that `cross_val_score` and
similar procedures might temporarily give different results across calls, in
cases where the estimator is still stateful (i.e. if its `random_state` is an
instance). The number of user codes where results may change once the
estimators become stateless is likely to be quite small. This may be
acceptable with a reasonable amount of outreach.

An easier but more brutal adoption path is to just break backward
compatibility and release this in a new major version, e.g. 1.0 or 2.0.

Alternative solutions
=====================

Change the default of all random_state from `None` to a hardcoded int
---------------------------------------------------------------------

This doesn't solve much: it might limit pitfalls in users code, but does not
address the statefulness issues, nor the ambiguity of CV procedures and
meta-estimators.

Stop supporting RandomState instances
-------------------------------------

We would not be able to support use-cases D, F, and G, except by passing
`None`, but then it would be impossible to get reproducible results across
executions (use-case A). This doesn't address either the statefulness and
ambiguity issues previousl mentioned.

Store a seed in fit/split instead of in init
---------------------------------------------

Instead of storing a seed in `__init__`, we could store it the first time
`fit()` is called. For example::

    def fit(self):  # or split()
        self._random_state = check_random_state(self.random_state).randint(2*32)
        rng = self._random_state
        # ...

The advantage is that we respect our convention with `__init__`.

However, `fit` idempotency isn't respected anymore: the first call to `fit`
clearly influences all the other ones. The mental model is also not as clear
as the one of the proposed solution: users don't really know when that seed
is going to be drawn unless they know the internals of the procedures they
are using.

This also introduces a private attribute, so we would need more intrusive
changes to `set_params`, `get_params`, and `clone`.

Add a new parameter to estimators instead of adding a parameter to CV procedures and meta-estimators
----------------------------------------------------------------------------------------------------

Instead of updating each CV procedure and meta-estimator with the
`use_exact_clones` parameter, we could instead add this parameter to the
estimators that have a `random_state` attribute, and let `clone` detect what
kind of clone it needs to output depending on the estimators' corresponding
attribute.

However, the strategy used by CV procedures and estimators would still be
somewhat implicit, and making these strategies explicit is one of the main
goal of this SLEP. It is also easier to document and to expose the different
possible strategies of a CV procedure when there is a dedicated parameter in
that procedure.

.. References and Footnotes
.. ------------------------

.. .. [1] Each SLEP must either be explicitly labeled as placed in the public
..    domain (see this SLEP as an example) or licensed under the `Open
..    Publication License`_.

.. .. _Open Publication License: https://www.opencontent.org/openpub/


.. Copyright
.. ---------

.. This document has been placed in the public domain. [1]_
