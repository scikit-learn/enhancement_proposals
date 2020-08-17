.. _slep_009:

===============================
SLEP009: Keyword-only arguments
===============================

:Author: Adrin Jalali
:Status: Accepted
:Type: Standards Track
:Created: 2019-07-13
:Vote opened: 2019-09-11

Abstract
########

This proposal discusses the path to gradually forcing users to pass arguments,
or most of them, as keyword arguments only. It talks about the status-quo, and
the motivation to introduce the change. It shall cover the pros and cons of the
change. The original issue starting the discussion is located
`here <https://github.com/scikit-learn/scikit-learn/issues/12805>`__.

Motivation
##########

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


- It allows adding new parameters closer the other relevant parameters, instead
  of adding new ones at the end of the list without breaking backward
  compatibility. Right now all new parameters are added at the end of the
  signature. Once we move to a keyword only argument list, we can change their
  order and put related parameters together. Assuming at some point numpydoc
  would support sections for parameters, these groups of parameters would be in
  different sections for the documentation to be more readable. Also, note that
  we have previously assumed users would pass most parameters by name and have
  sometimes considered changes to be backwards compatible when they modified
  the order of parameters. For example, user code relying on positional
  arguments could break after a deprecated parameter was removed. Accepting
  this SLEP would make this requirement explicit.

Solution
########

The official supported way to have keyword only arguments is:

.. code-block:: python

    def func(arg1, arg2, *, arg3, arg4)

Which means the function can only be called with `arg3` and `arg4` specified
as keyword arguments:

.. code-block:: python

    func(1, 2, arg3=3, arg4=4)

The feature was discussed and the related PEP
`PEP3102 <https://www.python.org/dev/peps/pep-3102/>`_ was accepted and
introduced in Python 3.0, in 2006.

For the change to happen in ``sklearn``, we would need to add the ``*`` where
we want all subsequent parameters to be passed as keyword only.

Considerations
##############

We can identify the following main challenges: familiarity of the users with
the syntax, and its support by different IDEs.

Syntax
------

Partly due to the fact that the Scipy/PyData has been supporting Python 2 until
recently, the feature (among other Python 3 features) has seen limited adoption
and the users may not be used to seeing the syntax. The similarity between the
following two definitions may also be confusing to some users:

.. code-block:: python

    def f(arg1, *arg2, arg3): pass # variable length arguments at arg2 

    def f(arg1, *, arg3): pass # no arguments accepted at *

However, some other teams are already moving towards using the syntax, such as
``matplotlib`` which has introduced the syntax with a deprecation cycle using a
decorator for this purpose in version 3.1. The related PRs can be found `here
<https://github.com/matplotlib/matplotlib/pull/13601>`__ and `here
<https://github.com/matplotlib/matplotlib/pull/14130>`__. Soon users will be
familiar with the syntax.

IDE Support
-----------

Many users rely on autocomplete and parameter hints of the IDE while coding.
Here is how the hint looks like in two different IDEs. For instance, for the
above function, defined in VSCode, the hint would be shown as:

.. code-block:: python

               func(arg1, arg2, *, arg3, arg4)

               param arg3
    func(1, 2, |)

The good news is that the IDE understands the syntax and tells the user it's
the ``arg3``'s turn. But it doesn't say it is a keyword only argument.

`ipython`, however, suggests all parameters be given with the keyword anyway:

.. code-block:: python

    In [1]: def func(arg1, arg2, *, arg3, arg4): pass               

    In [2]: func( 
      abs()                          arg3=                           
      all()                          arg4=                           
      any()                          ArithmeticError                >
      arg1=                          ascii()                         
      arg2=                          AssertionError                  

Scope
#####

An important open question is which functions/methods and/or parameters should
follow this pattern, and which parameters should be keyword only. We can
identify the following categories of functions/methods:

- ``__init__``
- Main methods of the API, *i.e.* ``fit``, ``transform``, etc.
- All other methods, *e.g.* ``SpectralBiclustering.get_submatrix``
- Functions

With regard to the common methods of the API, the decision for these methods
should be the same throughout the library in order to keep a consistent
interface to the user.

This proposal suggests making only *most commonly* used parameters positional.
The *most commonly* used parameters are defined per method or function, to be
defined as either of the following two ways:

- The set defined and agreed upon by the core developers, which should cover
  the *easy* cases.
- A set identified as being in the top 95% of the use cases, using some
  automated analysis such as `this one
  <https://odyssey.readthedocs.io/en/latest/tutorial.html>`__ or `this one
  <https://github.com/Quansight-Labs/python-api-inspect>`__.

This way we would minimize the number of warnings the users would receive,
which minimizes the friction cause by the change. This SLEP does not define
these parameter sets, and the respective decisions shall be made in their
corresponding pull requests.

Deprecation Path
----------------

For a smooth transition, we need an easy deprecation path. Similar to the
decorators developed in ``matplotlib``, a proposed solution is available at
[#13311](https://github.com/scikit-learn/scikit-learn/pull/13311), which
deprecates the usage of positional arguments on selected functions and methods.
With the decorator, the user sees a warning if they pass the designated
keyword-only arguments as positional, and removing the decorator would result
in an error. Examples (borrowing from the PR):

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

The final decorator solution shall make sure it is well understood by most
commonly used IDEs and editors such as IPython, Jupiter Lab, Emacs, vim,
VSCode, and PyCharm.
