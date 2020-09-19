.. _slep_011:

======================================
SLEP011: Fixing randomness ambiguities
======================================

:Author: Nicolas Hug
:Status: Under review
:Type: Standards Track
:Created: 2019-11-27

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
    implicity affects seemingly unrelated procedures.

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
  meta-estimators, by introducing new parameters, where relevant. A candidate
  parameter name is `use_exact_clones` (name open for discussion).
  
  This will **explicitly** expose the choice of the CV and meta-estimator
  strategy to users, instead of **implicitly** relying on the type of the
  estimator's `random_state`. Moreover, unlike in the legacy design, both
  all strategies will be supported by any estimator, no matter the type of
  the `random_state` attribute.

.. note::
    The *legacy* design refers to the design as used in the current
    scikit-learn master branch, i.e. `0.24.dev`.

.. note::
    Since passing `random_state=None` is equivalent to passing the global
    `RandomState` instance from `numpy`
    (`random_state=np.random.mtrand._rand`), we will not explicitly mention
    None here, but everything that applies to instances also applies to
    using None.

Issues with the legacy design
=============================

Introduction: a complex mental model
------------------------------------

The rules that dictate how `random_state` impacts the behaviour of estimators
and CV splitters are seemingly simple: use an int and get the same results
across calls, use an instance and get different results. Things get however
much more complicated when these estimators and splitters are used within
other procedures like CV procedures or meta-estimators.

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
will vary.

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

Unfortunately, there is now way for users to figure out what strategy is used
until they look at the code. It is not just a documentation problem. The core
problem here is that **the behaviour of the CV procedure is implicitly
dictated by the estimator's** `random_state`.

There are similar issues in meta-estimators, like `RFE` or
`SequentialFeatureSelection`: these are iterative feature selection
algorithms that will use either *exact* or *statistical* clones depending on
the estimator's `random_state`, which leads to two significantly different
strategies. Here again, **how they behave is only implicitly dictated by the
estimator's** `random_state`.

In addition, since the estimator's `random_state` type dictates the strategy,
users are bound to one single strategy once the estimator has been created:
it is for example impossible for an estimator to use a different RNG across
folds if that estimator was initialized with an integer.

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
`random_state` is a `RandomState` instance. This means that the following
code doesn't allow fold-to-fold comparison of scores::

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
    documented, at the very end of the `User Guide
    <https://scikit-learn.org/stable/modules/cross_validation.html#a-note-on-shuffling>`_.

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
estimator).

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
`evaluate_candidates()` internally calls `split` once. If
`evalute_candidates()` is called more than once, this means that **the
candidate parameters are evaluated on different splits each time**.

This is a quite subtle issue that third-party developers might easily
overlook. Some core devs (Joel Nothman and Nicolas Hug) kept forgetting and
re-discovering this issue over and over in the `Successive Halving PR 
<https://github.com/scikit-learn/scikit-learn/pull/13900>`_.

At the very least, this makes fold-to-fold comparison between candidates
impossible whenever the search strategy calls `evaluate_candidates()` more
than once. This can however cause bigger bugs in other scenarios, e.g. if we
implement successive halving + warm start (details ommitted here, you may
refer to `this issue
<https://github.com/scikit-learn/scikit-learn/issues/15125>`_).

In order to prevent any potential future bug and to prevent users
from making erroneous comparisons, the `Successive Halving implementation
<https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV>`_
forbids users from using stateful splitters, just like
`SequentialFeatureSelection`.

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

Depending on the estimator's `random_state` parameter, `clone` will return
an exact clone or a statistical clone. Statistical clones share a common
RandomState instance and thus are inter-dependent, as detailed in `#18363
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

.. note::
    C and E are very related: C is supported via creating strict clones (E).
    Similarly, D is supported by creating statistical clones (F).

    In legacy, C and D are mutually exclusive: a given estimator can only do
    C and not D, or only D and not C. Also, a given estimator can only
    produce strict clones or only statistical clones, but not both. In the
    proposed design, all estimators will support both C and D. Similarly,
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
        # ints are passe-through
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
`cross_val_score` should be updated with a new `use_exact_clones` parameter
(name up for discussion)::

    def _check_statistical_clone_possible(est):
      if 'random_state' not in est.get_params():
          raise ValueError("This estimator isn't random and can only have exact clones")

    def cross_val_score(est, X, y, cv, use_exact_clones=True):
        # use_exact_clones:
        # - if True, the same estimator RNG is used on each fold (use-case C) 
        # - if False, the different estimator RNG are used on each fold (use-case D) 
        
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

Meta-estimators should be updated in a similar fashion to make their two
alternative behaviors explicit. The `clone` function needs to be updated so
that one can explicitly request exact or statistical clones::

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
  in the documentation. *This is the main point of this SLEP*.

- The mental model is simpler: no matter what is passed as `random_state`
  (int, RandomState instances, or None), results are constant across calls to
  `fit` and `split`. The RandomState instance is mutated **once** (and only
  once) in `__init__`. Bugs will be less likely to sneak in.

- Estimators and splitters are stateless, and `fit` is now properly
  idempotent.

- Users now have explicit control on the CV procedures and meta-estimators,
  instead of implicitly relying on the estimator's `random_state`.

- Since CV splitters always yield the same splits, the chances of performing
  erroneous score comparison is limited.

- It is possible to preserve full backward-compatibility of behaviors in
  cases where an int was passed.

- `fit`, `split`, and `get_params` are unchanged.

Drawbacks
---------

- We break our convention that `__init__` should only ever store attributes,
  as they are passed in. Note however that the reason we have this convention
  is that we want the semantic of `__init__` and `set_params` are the same,
  and we want to enable people to change public attributes without having
  surprising behaviour. **This is still respected here.** So this isn't
  really an issue.

- CV procedures and meta-estimators must be updated.

Backward compatibility
----------------------

When an integer was used as `random_state`, all backward compatibility in
terms of results can be preserved (this will depend on the default value of
the `use_exact_clones` parameter)

Similarly, when an instance was used, results may be backward compatible for
CV procedures and meta-estimators depending on the default. However,
successive calls to `fit` will not yield backward-compatible results.
Similarly for splitters, backward compatibility cannot be preserved.

A deprecation path could be to start introducing the `use_exact_clones`
parameters without introducing the seed drawing in `__init__` yet. This would
however require a temporary deprecation of RandomState instances as well.

The easiest way might be allow for a breaking change, which means that the
SLEP would be implemented for a new major version of scikit-learn.

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
executions (use-case A)

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

.. References and Footnotes
.. ------------------------

.. .. [1] Each SLEP must either be explicitly labeled as placed in the public
..    domain (see this SLEP as an example) or licensed under the `Open
..    Publication License`_.

.. .. _Open Publication License: https://www.opencontent.org/openpub/


.. Copyright
.. ---------

.. This document has been placed in the public domain. [1]_
