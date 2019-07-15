.. _slep_009:

===============================
SLEP009: Keyword-only arguments
===============================

:Author: Adrin Jalali
:Status: Draft
:Type: Standards Track
:Created: 2019-07-13

Abstract
--------

This proposal discusses the path to gradually forcing users to pass arguments,
or most of them, as keyword arguments only. It talks about the status-quo, and
the motivation to introduce the change. It shall cover the pros and cons of the
change. The original issue starting the discussion is located
[here](https://github.com/scikit-learn/scikit-learn/issues/12805).

Motivation
----------

At the moment `sklearn` accepts most of the arguments both as positional and
keyword arguments. For example, both the following are valid:

.. code-block:: python

    # positional arguments
    clf = svm.SVC(.1, 'rbf')
    # keyword arguments
    clf = svm.SVC(C=.1, kernel='rbf')


Using keyword arguments has a few benefits:

- it is more readable
- for models which accept many parameters, especially numerical, it is less
  error-prone than positional arguments. Compare these examples:

.. code-block:: python

    cls = cluster.OPTICS(
        min_samples=5, max_eps=inf, metric=’minkowski’, p=2,
        metric_params=None, cluster_method=’xi’, eps=None, xi=0.05,
        predecessor_correction=True, min_cluster_size=None, algorithm=’auto’,
        leaf_size=30, n_jobs=None)

    cls = cluster.OPTICS(5, inf, ’minkowski’, 2, None, ’xi’, None, 0.05,
                         True, None, ’auto’, 30, None)


- it allows us to add new parameters closer the other relevant parameters,
  instead of adding new ones at the end of the list. Right now all new
  parameters are added at the end for backward compatibility. We may even get
  some sections support in numpydoc, and have sections in the documentation for
  the parameters.

Solution
--------

A proposed solution is available at
[#13311](https://github.com/scikit-learn/scikit-learn/pull/13311), which
deprecates the usage of positional arguments for most arguments on certain
functions. It uses a decorator, and removing the decorator would result in an
error if the function is called with positional arguments. Examples (borrowing
from the PR):

.. code-block:: python

    @warn_args
    def dbscan(X, eps=0.5, *, min_samples=4, metric='minkowski'):
        pass


    class LogisticRegression:

        @warn_args
        def __init__(self, penalty='l2', *, dual=False):

            self.penalty = penalty
            self.dual = dual


Calling ``LogisticRegression('l2', True)`` will result with a
``DeprecationWarning``:

.. code-block:: bash

    Should use keyword args: dual=True


Once the deprecation period is over, we'd remove the decorator and calling
the function/method with the positional arguments after `*` would fail.

Challenges
----------

The official supported way to have keyword only arguments is:

.. code-block:: python

    def func(arg1, arg2, *, arg3, arg4)

Which means the function can only be called with `arg3` and `arg4` specified
as keyword arguments:

.. code-block:: python

    func(1, 2, arg3=3, arg4=4)

The feature was discussed and the related PEP
([PEP3102](https://www.python.org/dev/peps/pep-3102/)) was accepted and
introduced in Python 3.0, in 2006. However, partly due to the fact that the
Scipy/PyData was supporting Python 2 until recently, the feature (among other
Python 3 features) has seen limited adoption and the users may not be used to
seeing the syntax. For instance, for the above function, defined in VSCode, the
hint would be shown as (these outputs are without the usage of the decorator):

.. code-block:: python

               func(arg1, arg2, *, arg3, arg4)

               param arg3
    func(1, 2, |)

The good news is that the IDE understands the syntax and tells the user it's
the ``arg3``'s turn. But it doesn't say it is a keyword only argument.

`ipython` would show:

.. code-block:: python

    In [1]: def func(arg1, arg2, *, arg3, arg4): pass               

    In [2]: func( 
      abs()                          arg3=                           
      all()                          arg4=                           
      any()                          ArithmeticError                >
      arg1=                          ascii()                         
      arg2=                          AssertionError                  

However, with the decorator, ``ipython`` shows:

.. code-block:: python

    In [2]: func( 
      arg1=                          ArithmeticError                 
      abs()                          ascii()                         
      all()                          AssertionError                 >
      any()                          AttributeError                  

The parameters are still all there, but a bit more hidden and in a different
order. The hint shown by VSCode seems unaffected.

Scope
-----

The main question is, which functions/methods should follow this pattern. We
have a few categories here:

- The ``__init__`` parameters
  - all arguments
  - less commonly used arguments only (For instance, ``C`` and ``kernel`` in
    ``SVC`` could be positional, the rest keyword only).
- Main methods of the API, _i.e._ ``fit``, ``transform``, etc.
  - all arguments
  - less commonly used arguments only (For instance, ``X`` and ``y`` in
    ``fit`` could be positional, the rest keyword only).
- Functions
  - all arguments
  - less commonly used arguments only (For instance, ``score_func`` in
    ``make_scorer`` could be positional, the rest keyword only).

The change can also be a gradual one in the span of two or three releases. But
I'm not sure if that's a better idea than telling people they should do keyword
only, always, once.

Notes
-----

Some conversations with the users of `sklearn` who have been using the package
for a while, shows the feedback is positive for this change.

It is also worth noting that ``matplotlib`` has introduced a decorator for this
purpose in versoin 3.1, and the related PRs can be found
[here](https://github.com/matplotlib/matplotlib/pull/14130) and
[here](https://github.com/matplotlib/matplotlib/pull/14130).

