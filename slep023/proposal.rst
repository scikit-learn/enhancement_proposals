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
Structured logging, Monitoring, Snapshots, etc. It would also provide a generic and
consistent API for early stopping, which is currently implemented differently in only a
few estimators. For such widely requested features, scikit-learn would provide builtin
callbacks, but users would also be able to implement their own callbacks for less common
use cases.

By providing a way to gather information during the training process, effectivley making
scikit-learn implementation of machine learning algorithms more transparent, the
callback API would also bring a lot of value for educational, testing and debugging
purposes.

Detailed description
--------------------

This section discusses the main challenges and the proposed solutions for the callback
API to be generic and flexible enough despite the wide diversity of scikit-learn
estimators and meta-estimators.

Callbacks and hooks
~~~~~~~~~~~~~~~~~~~

This SLEP proposes to define a callback as an object containing methods called at
specific points of `fit` (hooks). To keep the scope of this SLEP reasonable, only hooks
called during `fit` are discussed here, but the API could be extended to other methods
in the future.

One hook is called at the beginning of `fit` (`on_fit_begin`) and one at the
end (`on_fit_end`) to allow callback initialization and cleanup. A third hook is called
at the end of each iteration (`on_fit_iter_end`), where an iteration is roughly defined
as any loop step from the outermost loop in the `fit` method down to a loop over the
full dataset or batch for online algorithms (e.g. `MiniBatchKMeans`).

Due to the diversity of the estimators in scikit-learn, it's not realistic to define
specific hooks for each of these kind of iteration, which would end up with estimator
specific hooks, defeating the purpose of a generic API.

Computation tree
~~~~~~~~~~~~~~~~

The computation steps of an estimator can be seen as a tree, where the root is the
beginning of `fit` and nested loops are the different depth levels, each loop step
being a node. The leaves are the steps of the innermost loop over the full dataset.
Based on the above section, the `on_fit_iter_end` hook is thus called at each node
of this tree.

Callbacks need to act differently depending on the depth they are called at. This SLEP
thus proposes that every estimator hold its computation tree as an attribute, such that
callbacks can know at which node they are called and act accordingly. Because the tree
depends on the value of the estimator's hyperparameters, this tree structure is created
dinamically at the beginning of `fit`. When estimators are nested, the trees are merged,
the root of the inner estimator being a leaf of the outer estimator.

Performance
~~~~~~~~~~~

It's inevitable that callbacks will have a performance cost, especially when called
within cython nogil code. The most important thing is to make sure is that when no
callbacks are registered the performance is not affected (not acquiring the GIL for
instance).

In addition, some information given to the `on_fit_iter_end` hook might be expensive to
compute but is not needed by all callbacks. This SLEP proposes that callbacks must
explicitly request the information which is expensive to compute, and that the estimator
gives this information to the hook in a lazy way.

Callback propagation
~~~~~~~~~~~~~~~~~~~~

When composing estimators and meta-estimators, some callbacks are intuitively expected
to be applied to all of them. For instance progress bars should be able to report the
progress of all parts of such compositions. This SLEP proposes that some callbacks have
the property of being automatically propagated to inner estimators, such that they only
need to be registered on the outermost estimator.

Parallelism
~~~~~~~~~~~

Many scikit-learn estimators use multiprocessing or multithreading which can make the
design of callbacks more complex, because callbacks don't share their state between
processes. The file system or `multiprocessing.Manager` objects  for instance should be
used to overcome this issue.

Implementation
--------------

An implementation of this SLEP is being developed in :pr:`27663`, which targets the
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
