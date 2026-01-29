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
only a few estimators expose a few public attributes containing some information
accumulated during training (e.g. `HistGradientBoostingClassifier.train_score_`), and
that's pretty much it.

Some use cases that have been requested many times on the scikit-learn issue tracker
which are not possible to achieve with the current API are for instance Progress bars,
Structured logging, Monitoring, Snapshots, etc. It could also provide a generic and
consistent API for early stopping, which is currently implemented differently in only a
few estimators. For such widely requested features, scikit-learn would provide builtin
callbacks, but users would also be able to implement their own callbacks for less common
use cases.

By providing a way to gather information during the training process, effectivley making
scikit-learn implementation of machine learning algorithms more transparent, the
callback API would also bring a lot of value for educational, testing and debugging
purposes.

Public interfaces
-----------------

This sections describes the proposed public interfaces for the callback API. They are
divided into three subsets which target different kind of users: the end users of
scikit-learn, the third-party developers implementing custom estimators and the
third-party developers implementing custom callbacks.

Using callbacks
~~~~~~~~~~~~~~~

Callbacks are objects exposed in the `sklearn.callbacks` module. They can be registered
to an estimator using their `set_callbacks` method::

    from sklearn.linear_model import LogisticRegression
    from sklearn.callbacks import ProgressBar

    clf = LogisticRegression()
    callback = ProgressBar()
    clf.set_callbacks(callback)
    clf.fit(X, y)

When composing estimators and meta-estimators, some callbacks are intuitively expected
to be applied to all of them. For instance progress bars should be able to report the
progress of all parts of such compositions. This SLEP proposes that some callbacks have
the property of being automatically propagated to inner estimators, such that they only
need to be registered on the outermost estimator.

The callback protocol
~~~~~~~~~~~~~~~~~~~~~

Callbacks must implement the following protocol::

    class Callback(Protocol):
        def on_fit_begin(self, estimator)
        def on_fit_task_end(self, estimator, context, **kwargs)
        def on_fit_end(self, estimator, context)

The `on_fit_begin` hook is called at the beginning of `fit` and can be used to
initialize the callback. The `on_fit_end` hook is called at the end of `fit` and can be
used to cleanup the callback. The `on_fit_task_end` hook is called at the end of each
task performed during `fit`. Usually tasks correspond to iterations of loops during
`fit` and nested loops correspond to nested tasks, but in general a task can be any
unit of work defined by the estimator.

The `context` parameter is an instance of `CallbackContext`, which provides information
about the current state of the computation, described in more details below.

Callback support in estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add callback support to an estimator, scikit-learn provides three components:

- a `CallbackSupportMixin` class that the estimator should inherit from. This mixin
  exposes the `set_callbacks` method.

- a `with_callback_context` decorator to wrap the `fit` method of the estimator. This
  decorator takes care of the setup and teardown of callbacks and makes sure they are
  properly executed even if an exception is raised during `fit`.

- a `CallbackContext` class that is used to manage the callbacks during `fit`. An
  instance of this class is created by the `with_callback_context` decorator at the
  beginning of `fit` and can be accessed as the `_callback_fix_ctx` of the estimator.

  It exposes three methods responsible for calling the hooks of the registered
  callbacks: `eval_on_fit_begin`, `eval_on_fit_task_end` and `eval_on_fit_end`. In
  addition, it exposes a `propagate_callbacks` method to propagate callbacks from a
  meta-estimator to its inner estimators. Finally it exposes a `subcontext` method to
  create a `CallbackContext` instance for a subtask.

A typical implementation of callback support in an estimator would look like this::

    from sklearn.base import BaseEstimator
    from sklearn.callback import CallbackSupportMixin, with_callback_context

    class MyEstimator(CallbackSupportMixin, BaseEstimator):

        @with_callback_context
        def fit(self, X, y):
            callback_ctx = self._callback_fit_ctx
            callback_ctx.eval_on_fit_begin(estimator=self)
            for i in range(self.n_iter):
                subcontext = callback_ctx.subcontext(task_id=i)

                <computation part of the estimator>

                subcontext.eval_on_fit_task_end(
                    estimator=self,
                    context=subcontext,
                )

            return self

Design considerations
---------------------

This section discusses the main challenges and the proposed solutions for the callback
API to be generic and flexible enough despite the wide diversity of scikit-learn
estimators and meta-estimators.

Callback hooks
~~~~~~~~~~~~~~

To keep the scope of this SLEP reasonable, it only considers callbacks acting during
`fit` but the API could be extended to other methods in the future.

In addition, the smallest task granularity considered in this SLEP are tasks dealing
with the full dataset. For instance the `on_fit_task_end` hook is called at the end of
a loop iterating over the full dataset but not at the end of each step of such loop.

Performance
~~~~~~~~~~~

It's inevitable that callbacks will have a performance cost, especially when called
within cython nogil code. The most important thing is to make sure that when no
callbacks are registered the performance is not affected (not acquiring the GIL for
instance).

Moreover, some information given to the `on_fit_task_end` hook might be expensive to
compute but is not needed by all callbacks. This SLEP proposes that callbacks must
explicitly request the information which is expensive to compute, and that the estimator
gives this information to the hook in a lazy way.

Parallelism
~~~~~~~~~~~

Many scikit-learn estimators use multiprocessing or multithreading which can make the
design of callbacks more complex, because callbacks don't share their state between
processes. The file system or `multiprocessing.Manager` objects  for instance should be
used to overcome this issue.

Implementation
--------------

An implementation of this SLEP is being developed in :pr:`28760`, which targets the
`callbacks` feature branch. In this PR, only the base infrastructure and one builtin
callback (progress bars) are implemented. Other callbacks and documention will be
added incrementally in follow-up PRs.

Discussion
----------

The goal of this SLEP is to provide a common solution for a wide range of use cases
discussed in many long-standing issues and PRs, going back to :issue:`78`. Here are
some related issues and PRs: :issue:`4863`, :issue:`7574`, :issue:`8433`,
:issue:`8994`, :issue:`9136`, :issue:`10489`, :issue:`10973`, :issue:`12325`,
:issue:`14338`, :issue:`14531`, :issue:`16118`, :issue:`18507`, :issue:`18748`,
:issue:`18773`, :issue:`20127`, :issue:`20668`, :issue:`23156`, :issue:`24524`,
:issue:`25187`, :issue:`26395`, :issue:`26494`, :issue:`26532`, :pr:`1171`, :pr:`3817`,
:pr:`8317`, :pr:`16925`, :pr:`22000`.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open Publication
   License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

Copyright
---------

This document has been placed in the public domain. [1]_
