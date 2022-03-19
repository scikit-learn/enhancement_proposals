==============
Clone Override
==============

:Author: Joel Nothman
:Status: Draft
:Type: Standards Track
:Created: 2022-03-19
:Resolution: (required for Accepted | Rejected | Withdrawn)

Abstract
--------

The ability to clone Scikit-learn estimators -- removing any state due to
previous fitting -- is essential to ensuring estimator configurations are
reusable across multiple instances in cross validation.
A centralised implementation of :func:`sklearn.base.clone` regards
an estimator's constructor parameters as the state that should be copied.
This proposal allows for an estimator class to perform further operations
during clone with a ``__sklearn_clone__`` method, which will default to
the current ``clone`` behaviour.

Detailed description
--------------------

Cloning estimators is one way that Scikit-learn ensures that there is no
data leakage across data splits in cross-validation: by only copying an
estimator's configuration, with no data from previous fitting, the
estimator must fit with a cold start.  Cloning an estimator often also
occurs prior to parallelism, ensuring that a minimal version of the
estimator -- without a large stored model -- is serialised and distributed.

Cloning is currently governed by the implementation of
:func:`sklearn.base.clone`, which recursively descends and copies the
parameters of the passed object. For an estimator, it constructs a new
instance of the estimator's class, passing to it cloned versions of the
parameter values returned by its ``get_params``. It then performs some
sanity checks to ensure that the values passed to the construtor are
identical to what is then returned by the clone's ``get_params``.

The current equivalence between constructor parameters and what is cloned
means that whenever an estimator or library developer deems it necessary
to have further configuration of an estimator reproduced in a clone,
they must include this configuration as a constructor parameter.

Cases where this need has been raised in Scikit-learn development include:

* ensuring metadata requests are cloned with an estimator
* ensuring parameter spaces are cloned with an estimator
* building a simple wrapper that can "freeze" a pre-fitted estimator

The current design also limits the ability for an estimator developer to
define an exception to the sanity checks (see :issue:`15371`).

This proposal empowers estimator developers to extend the base implementation
of ``clone`` by providing a ``__sklearn_clone__`` method, which ``clone`` will
delegate to when available. The default implementaton will match current
``clone`` behaviour. It will be provied through
``BaseEstimator.__sklearn_clone__`` but also
provided for estimators not inheriting from :obj:`~sklearn.base.BaseEstimator`.

This shifts the paradigm from ``clone`` being a fixed operation that
Scikit-learn must be able to perform on an estimator to ``clone`` being a
behaviour that each Scikit-learn compatible estimator must implement.
Developers are expected to be responsible in maintaintaining the fundamental
properties of cloning.

Implementation
--------------

Implementing this SLEP will require:

1. Factoring out `clone_parametrized` from `clone`, being the portion of its
   implementation that handles objects with `get_params`.
2. Modifying `clone` to call ``__sklearn_clone__`` when available on an
   object with ``get_params``, or ``clone_parametrized`` when not available.
3. Defining ``BaseEstimator.__sklearn_clone__`` to call ``clone_parametrized``.
4. Documenting the above.

Backward compatibility
----------------------

No breakage.

Alternatives
------------

Insted of allowing estimators to overwrite the entire clone process,
the core clone process could be obligatory, with the ability for an
estimator class to customise additional steps.

One API would allow for an estimator class to provide
``__sklearn__post_clone__(self, source)`` for operations in addition
to the core cloning, or ``__sklearn__clone_attrs__`` could be defined
on a class to specify additional attributes that should be copied for
that class and its descendants.

Alternative solutions include continuing to force developers into providing
sometimes-awkward constructor parameters for any clonable material, and
Scikit-learn core developers having the exceptional ability to extend
the ``clone`` function as needed.

Discussion
----------

:issue:`5080` raised the proposal of polymorphism for ``clone`` as the right
way to provide an object-oriented API, and as a way to enable the
implementation of wrappers around estimators for model memoisation and
freezing.  Objections were based on the notion that ``clone`` has a simple
contract, and that "extension to it would open the door to violations of that
contract" [2]_.

The naming of ``__sklearn_clone__`` was further proposed and discussed in
:issue:`21838`.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.
.. _Open Publication License: https://www.opencontent.org/openpub/

.. [2] `Gael Varoquaux's comments on #5080 in 2015
   <https://github.com/scikit-learn/scikit-learn/issues/5080#issuecomment-127128808>`__


Copyright
---------

This document has been placed in the public domain. [1]_
