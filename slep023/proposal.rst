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
of the training process (`fit`) of an estimator, even less a composition of several
estimators and meta-estimators. Setting a verbosity level provides some information and
only a few estimators expose public attributes containing some information accumulated
during training (e.g. `HistGradientBoostingClassifier.train_score_`), and that's pretty
much it.

Some use cases that have been requested many times on the scikit-learn issue tracker
which are not possible to achieve with the current API are for instance progress bars,
structured logging, metric monitoring, snapshots, etc. The callback API could also
provide a generic and consistent API for early stopping, which is currently implemented
differently in only a few estimators. For such widely requested features, scikit-learn
will provide builtin callbacks, but users would also be able to implement their own
callbacks for less common use cases.

By providing a way to gather information during the training process, effectivley making
scikit-learn implementation of machine learning algorithms more transparent, the
callback API would also bring a lot of value for educational, testing and debugging
purposes.

Public interfaces
-----------------

This sections describes the proposed public interfaces for the callbacks. They are
divided into three subsets which target different kinds of users:

- the end users of scikit-learn
- the scikit-learn and third-party developers implementing estimators
- the scikit-learn and third-party developers implementing callbacks

Using callbacks
~~~~~~~~~~~~~~~

Callbacks are objects that can be registered to an estimator to be called at specific
points during `fit`, gathering information about the training process. scikit-learn
provides a set of builtin callbacks for common use cases, exposed in the
`sklearn.callbacks` module.

Callbacks can be registered to an estimator using its `set_callbacks` method:

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.callbacks import ProgressBar

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
    from sklearn.callbacks import EarlyStopping, ProgressBar

    clf = LogisticRegression().set_callbacks(EarlyStopping())
    clf = GridSearchCV(clf, param_grid).set_callbacks(ProgressBar())
    clf.fit(X, y)

Callback support in estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add callback support to an estimator, scikit-learn provides three components:

- a `CallbackContext` class that is used to manage the callbacks during `fit`. Instances
  of this class represent contexts of tasks being executed, where tasks are unit of work
  defined by the estimator. There is one `CallbackContext` for each task. Tasks (and
  therefore callback contexts) have a natural tree structure, where each task can be
  decomposed into subtasks and so on, with the root task being the whole `fit` task.
  Usually tasks correspond to iterations of loops during `fit` and nested loops
  correspond to nested tasks, but in general a task can be any unit of work defined by
  the estimator.

  The `CallbackContext` class exposes the following methods:

  - `eval_on_fit_task_begin`, to evaluate the callbacks at the beginning of the task.
  - `eval_on_fit_task_end`, to evaluate the callbacks at the end of the task.
  - `subcontext`, to create a child `CallbackContext` for a subtask.
  - `propagate_callbacks`, to propagate the callback context and the auto-propagated
    callbacks from a meta-estimator to its sub-estimators.

  When a callback is evaluated for a given task, it receives the corresponding context
  which exposes attributes that provide information about the task being executed to be
  used by the callback to adapt its behavior. It also receives all the information that
  the estimator is able to provide about the state of the fitting process at this task.

- a `CallbackSupportMixin` class that the estimator should inherit from. This mixin
  exposes the following methods:

  - `set_callbacks`, to register callbacks to the estimator.
  - `_init_callback_context`, to create the root callback context for the estimator
    and setup the callbacks for the estimator.

- a `with_callbacks` decorator that the estimator should use to decorate the `fit`
  method. It runs `fit` in a try-finally block to ensure that callbacks are properly
  torn down no matter what happens during `fit`.

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
            callback_ctx.eval_on_fit_task_begin()

            for i in range(self.n_iter):
                subcontext = callback_ctx.subcontext(
                    task_name=f"iteration {i}", task_id=i
                ).eval_on_fit_task_begin(
                  data={"X_train": X, "y_train": y},
                )

                # <computation part of the estimator>

                subcontext.eval_on_fit_task_end(
                    data={"X_train": X, "y_train": y},
                )

            callback_ctx.eval_on_fit_task_end()

            return self

The callback protocol
~~~~~~~~~~~~~~~~~~~~~

Callbacks must implement the following protocol:

.. code-block:: python

    class FitCallback(Protocol):
        def setup(self, context): ...
        def on_fit_task_begin(self, context, *, data=None, metadata=None, fitted_estimator=None): ...
        def on_fit_task_end(self, context, *, data=None, metadata=None, fitted_estimator=None): ...
        def teardown(self, context): ...

The `setup` and `teardown` hooks are called at the beginning and end of `fit` and should
be used to setup and tear down the callback for the estimator. The `on_fit_task_begin`
and `on_fit_task_end` hooks are called at the beginning and end of each task performed
during `fit`, including the root task.

The keyword arguments received by the `on_fit_task_begin` and `on_fit_task_end`
hooks contain all the available information provided by the estimator about the state of
the fitting process at this task:

- "data": a dictionary containing the training and validation data.

- "metadata": a dictionary containing training and validation metadata, e.g. sample
  weights.

- "fitted_estimator": an estimator instance that is ready to predict, transform, etc.
  as if `fit` stopped at the end of this task.

Note that some estimators may not be able to provide all of these information for every
task. That list is likely to be extended in the future so callbacks hooks should be
tolerant to missing keys.

Auto-propagated callbacks must implement a small extension of this protocol:

.. code-block:: python

    class AutoPropagatedCallback(FitCallback, Protocol):
        @property
        def max_propagation_depth(self): ...

`max_propagation_depth` defines the maximum nesting level of sub-estimators to propagate
the callback to.

Considerations
--------------

This section discusses the main challenges and the proposed solutions for the callback
API to be generic and flexible enough despite the wide diversity of scikit-learn
estimators and meta-estimators.

Protocol extensions
~~~~~~~~~~~~~~~~~~~

To keep the scope of this SLEP reasonable, it only considers callbacks evaluated during
`fit` but the API could be extended to other methods (e.g. `predict` or `transform`) or
unbound functions (e.g. `cross_validate`) in the future.
PR `#33404<https://github.com/scikit-learn/scikit-learn/pull/33404>`_ for instance
proposes a new protocol for callback evaluation in unbound functions.

Task granularity
~~~~~~~~~~~~~~~~

The smallest task granularity considered in this SLEP is tasks dealing with the full
dataset. For instance the `on_fit_task_end` hook is called at the end of a loop
iterating over the full dataset but not at the end of each step of such loop. A smaller
granularity would not allow the same level of flexibility.

Performance
~~~~~~~~~~~

It's inevitable that callbacks will have a performance cost, especially when called
within cython nogil code. The most important thing is to make sure that when no
callbacks are registered the performance is not affected (not acquiring the GIL for
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
workflow but might not provide full value and perform their work sub-optimally.

Additional dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Some callbacks may require additional dependencies (e.g. `rich` for progress bars).
Such dependencies must be optional and only imported when the corresponding callback is
used, similarly to the display objects.

Implementation
--------------

An implementation of this SLEP is being developed in the `callbacks` feature branch.
PR `#33322<https://github.com/scikit-learn/scikit-learn/pull/33322>`_ keeps an updated
diff againt the `main` branch. It currently contains the callback framework and the
progress bar callback.

Discussion
----------

The goal of this SLEP is to provide a common solution for a wide range of use cases
discussed in many long-standing issues and PRs, going back to issue
`#78<https://github.com/scikit-learn/scikit-learn/issues/78>`_. Here are some related
issues and PRs: `#4863<https://github.com/scikit-learn/scikit-learn/issues/4863>`_,
`#7574<https://github.com/scikit-learn/scikit-learn/issues/7574>`_,
`#8433<https://github.com/scikit-learn/scikit-learn/issues/8433>`_,
`#8994<https://github.com/scikit-learn/scikit-learn/issues/8994>`_,
`#9136<https://github.com/scikit-learn/scikit-learn/issues/9136>`_,
`#10489<https://github.com/scikit-learn/scikit-learn/issues/10489>`_,
`#10973<https://github.com/scikit-learn/scikit-learn/issues/10973>`_,
`#12325<https://github.com/scikit-learn/scikit-learn/issues/12325>`_,
`#14338<https://github.com/scikit-learn/scikit-learn/issues/14338>`_,
`#14531<https://github.com/scikit-learn/scikit-learn/issues/14531>`_,
`#16118<https://github.com/scikit-learn/scikit-learn/issues/16118>`_,
`#18507<https://github.com/scikit-learn/scikit-learn/issues/18507>`_,
`#18748<https://github.com/scikit-learn/scikit-learn/issues/18748>`_,
`#18773<https://github.com/scikit-learn/scikit-learn/issues/18773>`_,
`#20127<https://github.com/scikit-learn/scikit-learn/issues/20127>`_,
`#20668<https://github.com/scikit-learn/scikit-learn/issues/20668>`_,
`#23156<https://github.com/scikit-learn/scikit-learn/issues/23156>`_,
`#24524<https://github.com/scikit-learn/scikit-learn/issues/24524>`_,
`#25187<https://github.com/scikit-learn/scikit-learn/issues/25187>`_,
`#26395<https://github.com/scikit-learn/scikit-learn/issues/26395>`_,
`#26494<https://github.com/scikit-learn/scikit-learn/issues/26494>`_,
`#26532<https://github.com/scikit-learn/scikit-learn/issues/26532>`_,
`#1171<https://github.com/scikit-learn/scikit-learn/pull/1171>`_,
`#3817<https://github.com/scikit-learn/scikit-learn/pull/3817>`_,
`#8317<https://github.com/scikit-learn/scikit-learn/pull/8317>`_,
`#16925<https://github.com/scikit-learn/scikit-learn/pull/16925>`_,
`#22000<https://github.com/scikit-learn/scikit-learn/pull/22000>`_.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open Publication
   License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

Copyright
---------

This document has been placed in the public domain. [1]_
