.. _slep_011:

=====================
SLEP011: random state
=====================

:Author: Nicolas Hug
:Status: Under review
:Type: Standards Track
:Created: 2019-11-23

Current status
==============

`random_state` parameters are used commonly in estimators, and in CV
splitters. They can be either int, RandomState instances, or None. The
parameter is stored as an attribute in init and never changes, as per our
convention.

`fit` and `split` usually look like this::

    def fit(self, X, y):  # or split(self, X, y)
        rng = check_random_state(self.random_state)
        ...
        rng.some_method()  # directly use instance, e.g. for sampling
        some_function_or_class(random_state=rng)  # pass it down to other tools
                                                  # e.g. if self is a meta-estimator

`check_random_state` behaves as follows::

    def check_random_state(x):
        if x is an int, return np.RandomState(seed=x)
        if x is a RandomState instance, return x
        if x is None, return numpy's RandomState singleton instance

Accepting instances or None comes with different complications, listed below.
`None` is the current default for all estimators / splitters.


Problems with passing RandomState instances or None
===================================================

The main issue with RandomState instances and None is that it makes the
estimator stateful accross calls to `fit`. Almost all the other issues are
consequences of this fact.

Statefulness and violation of fit idempotence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimators should be stateless. That is, any call to `fit` should forget
whatever happened to previous calls to `fit`, provided that no warm-start is
hapenning. Whether CV splitters should be stateful or not is debatable, and
that point is discussed below.

Another related convention is that the `fit` method should be idempotent:
calling `est.fit(X, y)` and then `est.fit(X, y)` again should yield the same
model both times.

These properties are key for enforcing reproducibility. We have a common
checks for them.

If a `RandomState` instance or None are used, the idemptency property may be
violated::

    rng = RandomState(0)
    est = Est(..., random_state=rng)
    est.fit(...)  # consume rng
    est.fit(...)

The model inferred the second time isn't the same as the previous one since
the rng has been consumed during the first called. The statefulness property
isn't respected either since the first call to `fit` has an impact on the
second.

The same goes with passing `None`.

A related issue is that the `rng` may be consumed outside of the estimator.
The estimator isn't "self contained" anymore and its behaviour is now
dependent on some stuff that happen outside.

Countless bugs
~~~~~~~~~~~~~~

Quoting @amueller from `14042
<https://github.com/scikit-learn/scikit-learn/issues/14042>`_, many bugs
have happened over the years because of RandomState instances and None.

These bugs are often hard to find, and some of them are actual data leaks,
see e.g. `14034
<https://github.com/scikit-learn/scikit-learn/issues/14034>`_. Some of these
bugs have been around forever and we just haven't discovered them yet.

The bug in `14034
<https://github.com/scikit-learn/scikit-learn/issues/14034>`_ is that the
validation subset for early-stopping was computed using `self.random_state`:
in a warm-starting context, that subset may be different accross multiple
calls to `fit`, since the RNG is getting consumed. We instead want the
subset to be the same for all calls to `fit`. The fix was to store a private
seed that was generated the first time `fit` is called.

This is a typical example of many other similar bugs that we need to
monkey-patch with potentially overly complex logic.

Cloning may return a different estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`my_clone = clone(est)` returns an *unfitted* estimator whose parameters are
(supposed to be) the same as those that were passed when `est` was
instanciated. Whether
*clone* is a good name for describing this process is another issue, but the
semantic of cloning is scikit-learn is as described above. We can think of
*cloning* as *reset with initial parameters*.

That semantic is not respected if `est` was instanciated with an instance or
with None::

    rng = RandomState(0)
    est = Est(..., random_state=rng)
    est.fit(X, y)  # consume RNG here
    my_clone = clone(est)
    my_clone.fit(X, y)  # not the same model!

`my_clone` isn't the same as what `est` was, since the RNG has been consumed
in-between. Fitting `my_clone` on the same data will not give the same model
as `est`. While this is not desirable when an instance is passed, one might
argue that this is the desired behaviour when None is passed.

In addition, `est` and `my_clone` are now interdependent because they share the
same `rng` instance.

Incompatibility with third-party estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the snippet above in the introduction, most meta-estimators will
directly pass down `self.random_state` to their sub-estimators. Some
third-party libraries only support integers as `random_state`, not instances
or None. See e.g. `15611
<https://github.com/scikit-learn/scikit-learn/issues/15611>`_

CV-Splitters statefulness
~~~~~~~~~~~~~~~~~~~~~~~~~

CV-splitters are stateful::

    rng = np.random.RandomState(0)
    cv = KFolds(shuffle=True, random_state=rng)
    a = cv.split(X, y)
    b = cv.split(X, y)  # different from a

`a` and `b` are different splits, because of how `split` is implemented (see
introduction above).

This behaviour is inconsistent for two reasons.

The first one is that if `rng` were an int, then `a` and `b` would have been
equal. Indeed, the behaviour of the CV splitter depends on the
**type** of the `random_state` parameter::

- int -> stateless, get the same splits each time you call split()
- None or instance -> stateful, get different splits each time you call split()

Concretely, we have a method (`split`) whose behaviour depends on the *type*
of a parameter that was passed to `init`. We can argue that this is a common
pattern in object-oriented design, but in the case of the `random_state`
parameter, this is potentially confusing.

The second inconsistency is that splitters are stateful by design, while we
want our estimators to be stateless. Granted, splitters aren't estimators.
But, quoting `@GaelVaroquaux
<https://github.com/scikit-learn/scikit-learn/pull/15177#issuecomment-548021786>`_,
consistency is one thing that we are really good at.
So it is important to have the splitters consistent with the estimators,
w.r.t. the statelessness property. The current behaviour is not necessarily
clear for users.

Note Fixing how random_state is handled in the splitters is one of the
entries in the `Roadmap <https://scikit-learn.org/dev/roadmap.html>`_. Some
users may rely on the current behaviour, `to implement e.g. bootstrapping,
<https://github.com/scikit-learn/scikit-learn/pull/15177#issuecomment-548021786>`_.
If we make the splitters stateless, the "old" behaviour can be easily
reproduced by simply creating new CV instances, instead of calling `split`
on the same instance. Instances are cheap to create.

Potential bugs in custom parameter searches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(This issue is a direct consequence of the splitters being stateful.)

We have a private API for subclassing BaseSearchCV and implementing custom
parameter search strategies. The contract is that the custom class should
override the `_run_search(evaluate_candidate, ...)` method which itself must
call the `evaluate_candidates()` closure, were `cv.split()` will be called.

Third-party developers may only *call* `evaluate_candidates()`, not change
its content. Now, since `cv.split()` is called in `evaluate_candadates()`,
that means that `evalute_candidates()` will potentially evaluate its
candidates parameters **on different splits** each time it is called.

This is a quite subtle issue that third-party developers might easily
overlook.

Depending on the intended behaviour of the parameter search, this may or may
not be a good thing. This is typically a bug if we implement successive
halving + warm start (details ommitted here, you may refer to `this issue
<https://github.com/scikit-learn/scikit-learn/issues/15125>`_ for some more
details).

Proposed Solution
=================

We need a solution that fixes the statefulness of the estimators and the
splitters.

A toy example of the proposed solution is implemented in this `notebook
<https://nbviewer.jupyter.org/gist/NicolasHug/1169ee253a4669ff993c947507ae2cb5>`_.

The proposed solution is to store the *state* of a RandomState instance in
`__init__`::

    def __init__(self, ..., random_state=None):
        self.random_state = check_random_state(random_state).get_state()

That `random_state` attribute is a tuple with about 620 integers, as returned
by `get_state()
<https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.get_state.html#numpy.random.get_state>`_.

That state is then used in `fit` or in `split` as follows::

    def fit(self, X, y):  # or split()
        rng = np.random.RandomState()
        rng.set_state(self.random_state)
        # ... use rng as before

Since `self.random_state` is a tuple that never changes, calling `fit` or
`split` on the same instance always gives the same results.

We want `__init__` and `set_params/get_params` to be consistent. To that end,
we will need to special-case these methods::

    def get_params(self):

        random_state = np.random.RandomState()
        random_state.set_state(self.random_state)
        return {'random_state': random_sate, ...}

    def set_params(self, ...):

        self.random_state = check_random_state(random_state).get_state()  # same as in init

`clone` does not need to be special-cased, because `get_params` does all the
work. Note that the following::

    est.set_params(random_state=est.get_params()['random_state'])

behaves as expected and does not change the `random_state` attribute of the
estimator. However, one should not use::

    est.set_params(random_state=est.random_state)

since `est.random_state` is neither an int, None or an instance: it is a tuple.
We can error with a decent message in that case.

Advantages:

- It fixes the statefullness issue. `fit` is now idempotent. Calling `split` on
  the same instance gives the same splits. In other words, it does what we
  want.

- It is relatively simple to implement, and not too intrusive.

- Backward compatibility is preserved between scikit-learn versions. Let A
  be a version with the current behaviour (say 0.22) and let B be a version
  where the new behaviour is implemented. The models and the splits obtained
  will be the same in A and in B. That property may not be respected with
  other solutions, see below.

- Both RandomState instances and None are still supported. We don't need to
  deprecate the use of any of them.

- As a bonus, the `self.random_state` attribute is an *actual* random state:
  it is the state of some RNG. What we currently call `random_state` is not
  a state but a RNG (though this is numpy's fault.)

Drawbacks:

- We break our convention that `__init__` should only ever store attributes, as
  they are passed in. Note however that the reason we have this convention
  is that we want the semantic of `__init__` and `set_params` are the same,
  and we want to enable people to change public attributes without having
  surprising behaviour. **This is still respected here.** So this isn't
  really an issue.

- There is a subtelty that occurs when passing `None`. `check_random_state`
  will return the singleton `np.random.mtrand._rand`, and we will call
  `get_state()` on the singleton. The thing is, its state won't change
  accross multiple instances unless the singleton is consumed. So if we do
  `a = Est(random_state=None); b = Est(random_state=None)`, a and b actually
  have exactly the same `random_state` attribute, since the state of the
  singleton wasn't changed. To circumvent this, the logic in `__init__` and
  `set_params` involves a private helper that makes sure the singleton's RNG is
  consumed. Please refer to the notebook.

- The `__repr__()` will need to special-case the `random_state` attribute to
  avoid printing a long tuple.

- We need to store about 620 integers. This is however negligible w.r.t. e.g.
  the size of a typical dataset

- It does not fix the issue about third-party packages only accepting integers.
  This can however be fixed in meta-estimator, independently.

Alternative solutions
=====================

Store a seed instead of a state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of storing a state from `get_state()`, we could store a randomly
generated seed::

    def __init__(self, ..., random_state=None):
        self.random_state = check_random_state(random_state).randint(0, BIG_INT)

Then instead of using `set_state` we could just use
`rng = RandomState(seed=self.random_state)` in `fit` or `split`.

Advantages:

- It also fixes the third-party estimators issue, since we would be passing
  self.random_state which is an int
- It's cheaper than storing 620 ints
- We don't need to artificially consume the singleton's RNG since it is
  de-facto consumed anyway.

Drawbacks:

- We want that
  `est.set_params(random_state=est.get_params()['random_state'])` does not
  change `est.random_state`. With this logic, this is not possible to enforce.
  Also, clone will not work was expected.

- It is not backward compatible between versions. For example if you passed
  an int in version A (say 0.22), then in version B (with the new
  behaviour), your estimator will not start with the same RNG when `fit` is
  called the first time. Same for splitters.

Store the state in fit/split instead of in init
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of storing the output of `get_state()` in `__init__`, we could store it
the first time `fit()` is called. For example::

    def fit(self):  # or split()
        self._random_state = getattr(self, '_random_state', check_random_state(self.random_state).get_state())
        rng = np.random.RandomState()
        rng.set_state(self._random_state)
        # ...

The advantage is that we respect our convention with `__init__`.

However, `fit` idempotency isn't respected anymore: the first call to `fit`
clearly influences all the other ones. This also introduces a private
attribute, so we would need more intrusive changes to `set_params`,
`get_params`, and `clone`.


Execution
=========

That change might be a 1.0 thing.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
