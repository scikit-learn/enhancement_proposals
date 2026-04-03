.. _slep_023:

=====================
SLEP023: Callback API
=====================

:Author: Jérémie du Boisberranger
:Status: Draft
:Type: Standards Track
:Created: 2024-03-01

Abstract
--------

This SLEP proposes an API to allow users to register callbacks to be called at
specific points during the training of scikit-learn estimators.

Motivation
----------

The current scikit-learn API provides a very limited way for users to inspect the steps
of the training process (`fit`) of an estimator, and even less so when several
estimators and meta-estimators are composed. Setting a verbosity level provides some
information and only a few estimators expose public attributes containing some
information accumulated during training
(e.g. `HistGradientBoostingClassifier.train_score_`), and that's pretty much it.

Some use cases that have been requested many times on the scikit-learn issue tracker
which are not possible to achieve with the current API are for instance progress bars,
structured logging, metric monitoring, snapshots, etc. A callback API could also provide
a generic and consistent API for early stopping, which is currently implemented
differently in only a few estimators. For such widely requested features, scikit-learn
will provide built-in callbacks, but users would also be able to implement their own
callbacks for less common use cases.

By providing a way to gather information during the training process, effectively making
the scikit-learn implementation of machine learning algorithms more transparent, the
callback API would also bring a lot of value for educational, testing, and debugging
purposes.

Public interfaces
-----------------

This section describes the proposed public interfaces for callbacks. They are divided
into three subsets which target different kinds of users:

- the end users of scikit-learn
- the scikit-learn and third-party developers implementing estimators
- the scikit-learn and third-party developers implementing callbacks

Using callbacks
~~~~~~~~~~~~~~~

Callbacks are objects that can be registered on an estimator to be called at specific
points during fit, gathering information about the training process. scikit-learn
provides a set of built-in callbacks for common use cases, exposed in the
`sklearn.callback` module.

Callbacks can be registered on an estimator using its `set_callbacks` method:

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.callback import ProgressBar

    callback = ProgressBar()
    clf = LogisticRegression()
    clf.set_callbacks(callback)
    clf.fit(X, y)

When composing estimators and meta-estimators, some callbacks are intuitively expected
to only be applied to some of them (e.g. early stopping on inner estimators) while
others are expected to be applied to all of them (e.g. progress bars). This SLEP
proposes that some callbacks (referred to as "auto-propagated") have the property of
being automatically propagated down to sub-estimators, such that they only need to be 
registered on the outermost meta-estimator:

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.callback import EarlyStopping, ProgressBar

    clf = LogisticRegression().set_callbacks(EarlyStopping())
    gs = GridSearchCV(clf, param_grid)
    gs.set_callbacks(ProgressBar(max_propagation_depth=1))
    gs.fit(X, y)

Callback support in estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add callback support to an estimator, scikit-learn provides three components:

- `CallbackContext`: a class that is used to manage the callbacks during fit.
  Instances of this class represent contexts of tasks being executed, where tasks are
  units of work defined by the estimator. There is one `CallbackContext` for each task.
  Tasks (and therefore callback contexts) have a natural tree structure, where each task
  can be decomposed into subtasks and so on, with the root task being the whole fit
  task. Usually tasks correspond to iterations of loops during fit and nested loops
  correspond to nested tasks, but in general a task can be any unit of work defined by
  the estimator.

  The `CallbackContext` class exposes the following methods:

  - `call_on_fit_task_begin`, to call the callbacks at the beginning of the task.
  - `call_on_fit_task_end`, to call the callbacks at the end of the task.
  - `subcontext`, to create a child `CallbackContext` for a subtask.
  - `propagate_callbacks`, to propagate the callback context and the auto-propagated
    callbacks from a meta-estimator to its sub-estimators.

  When a callback is called for a given task, it receives the corresponding context
  which exposes attributes that provide information about the task being executed to be
  used by the callback to adapt its behavior. It also receives all the information that
  the estimator is able to provide about the state of the fitting process at this task.

- `CallbackSupportMixin`: a class that the estimator should inherit from. This mixin
  exposes the following methods:

  - `set_callbacks`, to register callbacks on the estimator.
  - `_init_callback_context`, to create the root callback context for the estimator
    and set up the callbacks for the estimator.

- `with_callbacks`: a decorator that the estimator should use to decorate the fit
  method. It runs fit in a try-finally block to ensure that callbacks are properly
  torn down no matter what happens during fit.

A typical implementation of callback support in an estimator would look like this:

.. code-block:: python

    from sklearn.base import BaseEstimator
    from sklearn.callback import CallbackSupportMixin, with_callbacks

    class MyEstimator(CallbackSupportMixin, BaseEstimator):

        @with_callbacks
        def fit(self, X, y):
            callback_ctx = self._init_callback_context(
                task_name="fit", max_subtasks=self.n_iter
            )
            callback_ctx.call_on_fit_task_begin(X=X, y=y)

            for i in range(self.n_iter):
                callback_subctx = callback_ctx.subcontext(
                    task_name=f"iteration {i}", task_id=i
                ).call_on_fit_task_begin(X=X, y=y)

                # <computation part of the estimator>

                callback_subctx.call_on_fit_task_end(X=X, y=y)

            callback_ctx.call_on_fit_task_end(X=X, y=y)

            return self

The callback protocol
~~~~~~~~~~~~~~~~~~~~~

Callbacks must implement the following
`protocol <https://typing.python.org/en/latest/spec/protocol.html>`__:

.. code-block:: python

    class FitCallback(Protocol):
        def setup(self, estimator, context): ...
        def on_fit_task_begin(self, estimator, context, *, X=None, y=None, metadata=None, fitted_estimator=None): ...
        def on_fit_task_end(self, estimator, context, *, X=None, y=None, metadata=None, fitted_estimator=None): ...
        def teardown(self, estimator, context): ...

The 4 protocol methods are referred to as callback hooks in the rest of this SLEP. The
`setup` and `teardown` hooks are called at the beginning and end of fit and should
be used to set up and tear down the callback for the estimator. The `on_fit_task_begin`
and `on_fit_task_end` hooks are called at the beginning and end of each task performed
during fit, including the root task.

The keyword arguments received by the `on_fit_task_begin` and `on_fit_task_end`
hooks contain all the available information provided by the estimator about the state of
the fitting process at this task:

- "X": the training data.

- "y": the training target.

- "metadata": a dictionary containing training and validation metadata, e.g. sample
  weights, `X_val`, `y_val`, etc.

- "fitted_estimator": an estimator instance that is ready to predict, transform, etc.
  as if fit stopped at the end of this task.

Note that some estimators may not be able to provide values for all of these keys for
every task.

Auto-propagated callbacks must implement a small extension of this protocol:

.. code-block:: python

    class AutoPropagatedCallback(FitCallback, Protocol):
        @property
        def max_propagation_depth(self): ...

`max_propagation_depth` defines the maximum nesting level of sub-estimators to propagate
the callback to.

Example traces of hook calls
----------------------------

This section gives example traces of callback hook calls in different scenarios. First,
consider a callback registered on an estimator with a single level of iterations:

.. code-block:: python

    >>> estimator = MyEstimator(n_iter=10).set_callbacks(MyCallback())
    >>> estimator.fit(X, y)
    setup by MaxIterEstimator for fit
    on_fit_task_begin by MaxIterEstimator for fit
        on_fit_task_begin by MaxIterEstimator for iteration 0
        on_fit_task_end by MaxIterEstimator for iteration 0
        ...  # iterations 1 to 8
        on_fit_task_begin by MaxIterEstimator for iteration 9
        on_fit_task_end by MaxIterEstimator for iteration 9
    on_fit_task_end by MaxIterEstimator for fit
    teardown by MaxIterEstimator for fit

It doesn't matter whether `MyCallback` is auto-propagated or not since there is no
sub-estimator to propagate it to in that case. Next, consider the case where the
estimator from the previous example is wrapped in a meta-estimator that fits clones of
that sub-estimator for different cross-validation folds:

.. code-block:: python

    >>> estimator = MyEstimator(n_iter=10).set_callbacks(MyCallback())
    >>> MetaEstimatorCV(estimator, cv=5).fit(X, y)
    setup by MaxIterEstimator for fit (MetaEstimatorCV fold 0)
    on_fit_task_begin by MaxIterEstimator for fit (MetaEstimatorCV fold 0)
        on_fit_task_begin by MaxIterEstimator for iteration 0
        on_fit_task_end by MaxIterEstimator for iteration 0
        ...  # iterations 1 to 8
        on_fit_task_begin by MaxIterEstimator for iteration 9
        on_fit_task_end by MaxIterEstimator for iteration 9
    on_fit_task_end by MaxIterEstimator for fit (MetaEstimatorCV fold 0)
    teardown by MaxIterEstimator for fit (MetaEstimatorCV fold 0)
    ...  # folds 1 to 3
    setup by MaxIterEstimator for fit (MetaEstimatorCV fold 4)
    on_fit_task_begin by MaxIterEstimator for fit (MetaEstimatorCV fold 4)
        on_fit_task_begin by MaxIterEstimator for iteration 0
        on_fit_task_end by MaxIterEstimator for iteration 0
        ...  # iterations 1 to 8
        on_fit_task_begin by MaxIterEstimator for iteration 9
        on_fit_task_end by MaxIterEstimator for iteration 9
    on_fit_task_end by MaxIterEstimator for fit (MetaEstimatorCV fold 4)
    teardown by MaxIterEstimator for fit (MetaEstimatorCV fold 4)

The trace is similar to the first example, except that it is repeated for each fold. In
particular, the `setup` and `teardown` hooks are called for every fit of the
sub-estimator clone. Finally, consider the same composition with an auto-propagated
callback registered on the meta-estimator:

.. code-block:: python

    >>> estimator = MyEstimator(n_iter=10)
    >>> MetaEstimatorCV(estimator, cv=5).set_callbacks(MyAutoPropagatedCallback()).fit(X, y)
    setup by MetaEstimatorCV for fit
    on_fit_task_begin by MetaEstimatorCV for fit
        on_fit_task_begin by MaxIterEstimator for fit (MetaEstimatorCV fold 0)
            on_fit_task_begin by MaxIterEstimator for iteration 0
            on_fit_task_end by MaxIterEstimator for iteration 0
            ...  # iterations 1 to 8
            on_fit_task_begin by MaxIterEstimator for iteration 9
            on_fit_task_end by MaxIterEstimator for iteration 9
        on_fit_task_end by MaxIterEstimator for fit (MetaEstimatorCV fold 0)
        ...  # folds 1 to 3
        on_fit_task_begin by MaxIterEstimator for fit (MetaEstimatorCV fold 4)
            on_fit_task_begin by MaxIterEstimator for iteration 0
            on_fit_task_end by MaxIterEstimator for iteration 0
            ...  # iterations 1 to 8
            on_fit_task_begin by MaxIterEstimator for iteration 9
            on_fit_task_end by MaxIterEstimator for iteration 9
        on_fit_task_end by MaxIterEstimator for fit (MetaEstimatorCV fold 4)
    on_fit_task_end by MetaEstimatorCV for fit
    teardown by MetaEstimatorCV for fit

This time, the callback hooks are also called for the tasks of the meta-estimator. Note
that the `setup` and `teardown` hooks are called only once, by the meta-estimator.

Considerations
--------------

This section discusses the main challenges and the proposed solutions for the callback
API to be generic and flexible enough despite the wide diversity of scikit-learn
estimators and meta-estimators.

Protocol extensions
~~~~~~~~~~~~~~~~~~~

To keep the scope of this SLEP reasonable, it only considers callbacks called during
fit but the API could be extended to other methods (e.g. `predict` or `transform`) or
unbound functions (e.g. `cross_validate`) in the future.
PR `#33404 <https://github.com/scikit-learn/scikit-learn/pull/33404>`__, for instance,
proposes a new protocol for callback support in unbound functions.

The setup and teardown hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `setup` and `teardown` hooks are not absolutely necessary at the time of writing
since `on_fit_task_begin` and `on_fit_task_end` are also called at the beginning and end
of fit as well, so they could in principle be used to set up and tear down the callback.
The main motivations for including them are:

- in anticipation for future extensions of the protocol to other methods than fit;
- to separate technical concerns of resource management from semantic concerns related
  to reacting to specific events during fit.

Task granularity
~~~~~~~~~~~~~~~~

The smallest task granularity considered in this SLEP is tasks dealing with the full
dataset. For instance, the `on_fit_task_end` hook is called at the end of a loop
iterating over the full dataset but not at the end of each step of such a loop. A
smaller granularity would not allow the same level of flexibility and consistency across
estimators.

Performance
~~~~~~~~~~~

It's inevitable that callbacks will have a performance cost, especially when called
within Cython nogil code. The most important thing is to make sure that when no
callbacks are registered, the performance is not affected (not acquiring the GIL for
instance).

Moreover, some information given to the hooks might be expensive to compute but is not
needed by all callbacks. This SLEP proposes that callbacks must explicitly request the
information which is expensive to compute, and that the estimator passes it to the hooks
in a lazy way.

Parallelism
~~~~~~~~~~~

Many scikit-learn estimators use multiprocessing or multithreading which can make the
design of callbacks more complex, because callbacks don't share their state between
processes. The file system or `multiprocessing.Manager` objects for instance should be
used to overcome this issue.

Non-callback-aware meta-estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Callbacks can be registered on estimators that are passed to meta-estimators or unbound
functions that do not support callbacks. In this case, callbacks should not break the
workflow but might not provide full value and may perform suboptimally.

Additional dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Some callbacks may require additional dependencies (e.g. `rich` for progress bars).
Such dependencies must be optional and only imported when the corresponding callback is
used, similarly to the display objects.

Implementation
--------------

An implementation of this SLEP is being developed in the `callbacks` feature branch.
PR `#33322 <https://github.com/scikit-learn/scikit-learn/pull/33322>`__ keeps an updated
diff against the `main` branch. It currently contains the callback framework and the
progress bar callback.

Discussion
----------

The goal of this SLEP is to provide a common solution for a wide range of use cases
discussed in many long-standing issues and PRs, going back to issue
`#78 <https://github.com/scikit-learn/scikit-learn/issues/78>`__. Here are some related
issues and PRs: `#4863 <https://github.com/scikit-learn/scikit-learn/issues/4863>`__,
`#7574 <https://github.com/scikit-learn/scikit-learn/issues/7574>`__,
`#8433 <https://github.com/scikit-learn/scikit-learn/issues/8433>`__,
`#8994 <https://github.com/scikit-learn/scikit-learn/issues/8994>`__,
`#9136 <https://github.com/scikit-learn/scikit-learn/issues/9136>`__,
`#10489 <https://github.com/scikit-learn/scikit-learn/issues/10489>`__,
`#10973 <https://github.com/scikit-learn/scikit-learn/issues/10973>`__,
`#12325 <https://github.com/scikit-learn/scikit-learn/issues/12325>`__,
`#14338 <https://github.com/scikit-learn/scikit-learn/issues/14338>`__,
`#14531 <https://github.com/scikit-learn/scikit-learn/issues/14531>`__,
`#16118 <https://github.com/scikit-learn/scikit-learn/issues/16118>`__,
`#18507 <https://github.com/scikit-learn/scikit-learn/issues/18507>`__,
`#18748 <https://github.com/scikit-learn/scikit-learn/issues/18748>`__,
`#18773 <https://github.com/scikit-learn/scikit-learn/issues/18773>`__,
`#20127 <https://github.com/scikit-learn/scikit-learn/issues/20127>`__,
`#20668 <https://github.com/scikit-learn/scikit-learn/issues/20668>`__,
`#23156 <https://github.com/scikit-learn/scikit-learn/issues/23156>`__,
`#24524 <https://github.com/scikit-learn/scikit-learn/issues/24524>`__,
`#25187 <https://github.com/scikit-learn/scikit-learn/issues/25187>`__,
`#26395 <https://github.com/scikit-learn/scikit-learn/issues/26395>`__,
`#26494 <https://github.com/scikit-learn/scikit-learn/issues/26494>`__,
`#26532 <https://github.com/scikit-learn/scikit-learn/issues/26532>`__,
`#1171 <https://github.com/scikit-learn/scikit-learn/pull/1171>`__,
`#3817 <https://github.com/scikit-learn/scikit-learn/pull/3817>`__,
`#8317 <https://github.com/scikit-learn/scikit-learn/pull/8317>`__,
`#16925 <https://github.com/scikit-learn/scikit-learn/pull/16925>`__,
`#22000 <https://github.com/scikit-learn/scikit-learn/pull/22000>`__.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open Publication
   License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

Copyright
---------

This document has been placed in the public domain. [1]_
