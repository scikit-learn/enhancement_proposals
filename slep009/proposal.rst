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

At the moment `sklearn` accepts all arguments both as positional and
keyword arguments. For example, both of the following are valid:

.. code-block:: python

    # positional arguments
    clf = svm.SVC(.1, 'rbf')
    # keyword arguments
    clf = svm.SVC(C=.1, kernel='rbf')


Using keyword arguments has a few benefits:

- It is more readable.
- For models which accept many parameters, especially numerical, it is less
  error-prone than positional arguments. Compare these examples:

.. code-block:: python

    cls = cluster.OPTICS(
        min_samples=5, max_eps=inf, metric=’minkowski’, p=2,
        metric_params=None, cluster_method=’xi’, eps=None, xi=0.05,
        predecessor_correction=True, min_cluster_size=None, algorithm=’auto’,
        leaf_size=30, n_jobs=None)

    cls = cluster.OPTICS(5, inf, ’minkowski’, 2, None, ’xi’, None, 0.05,
                         True, None, ’auto’, 30, None)


- It allows adding new parameters closer the other relevant parameters,
  instead of adding new ones at the end of the list without breaking backward
  compatibility. Right now all new parameters are added at the end of the
  signature. Once we move to a keyword only argument list, we can change their
  order and put related parameters together. Assuming at some point numpydoc
  would support sections for parameters, these groups of parameters would be
  in different sections for the documentation to be more readable.

Solution
--------

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

Challenges
----------



Scope
-----

An important open question is which functions/methods and/or parameters should
follow this pattern, and which parameters should be keyword only. We can
identify the following categories and the corresponding options we have for
each of them:

- The ``__init__`` parameters
  * All arguments
  * Less commonly used arguments only (For instance, ``C`` and ``kernel`` in
  ``SVC`` could be positional, the rest keyword only).
- Main methods of the API, *i.e.* ``fit``, ``transform``, etc.
  * All arguments
  * Less commonly used arguments only (For instance, ``X`` and ``y`` in
  ``fit`` could be positional, the rest keyword only).
- All other methods, *e.g.* ``SpectralBiclustering.get_submatrix``
  * All arguments (and this being the only option since these methods are more
  ad-hoc).
- Functions
  * All arguments
  * Less commonly used arguments only (For instance, ``score_func`` in
  ``make_scorer`` could be positional, the rest keyword only).

The term *commonly used* here can either refer to the parameters which are used
across the library, such as ``X`` and ``y``, or a parameter which is often used
when that method is used, such as ``C`` for ``SVC``. In the spirit of having a
similar interface across the library, we can go with the first definition, and
define the positional parameters independent of the estimator/function.

The change can also be a gradual one in the span of two or three releases,
*i.e.* we can start by changing the ``__init__``s, and continue later with the
other ones. But that may cause more confusion, and changing all the above
categories together may be a better option.

Deprecation Path
----------------

A proposed solution is available at
[#13311](https://github.com/scikit-learn/scikit-learn/pull/13311), which
deprecates the usage of positional arguments for most arguments on certain
functions and methods. It uses a decorator, and removing the decorator would
result in an error if the function is called with positional arguments.
Examples (borrowing from the PR):

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

However, with the decorator, ``ipython`` shows:

.. code-block:: python

    In [2]: func( 
      arg1=                          ArithmeticError                 
      abs()                          ascii()                         
      all()                          AssertionError                 >
      any()                          AttributeError                  

The parameters are still all there, but a bit more hidden and in a different
order. The hint shown by VSCode seems unaffected.

Notes
-----

Some conversations with the users of `sklearn` who have been using the package
for a while, shows the feedback is positive for this change.

It is also worth noting that ``matplotlib`` has introduced a decorator for this
purpose in versoin 3.1, and the related PRs can be found
[here](https://github.com/matplotlib/matplotlib/pull/13601) and
[here](https://github.com/matplotlib/matplotlib/pull/14130).

