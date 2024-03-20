.. _slep_016:

=======================================
SLEP016: Parameter Spaces on Estimators
=======================================

:Author: Joel Nothman
:Status: Draft
:Type: Standards Track
:Created: 2021-11-30

Abstract
--------

This proposes to simplify the specification of parameter searches by allowing
the user to store candidate values for each parameter on the corresponding estimator.
The ``*SearchCV`` estimators would then have a setting to construct the
parameter grid or distribution from the supplied estimator.

Detailed description
--------------------

The ability to set and get parameters from deep within nested estimators using
``get_params`` and ``set_params`` is powerful, but the specification of
parameter spaces to search can be very unfriendly for users.
In particular, the structure of the parameter grid specification needs to
reflect the structure of the estimator, with every path explicitly specified by
``__``-separated elements.

For example, `one example <https://github.com/scikit-learn/scikit-learn/blob/d4d5f8c/examples/compose/plot_compare_reduction.py>`__
proposes searching over alternative preprocessing steps in a Pipeline and their
parameters, as well as the parameters of the downstream classifier.

::

    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.decomposition import PCA, NMF
    from sklearn.feature_selection import SelectKBest, chi2

    pipe = Pipeline(
        [
            # the reduce_dim stage is populated by the param_grid
            ("reduce_dim", "passthrough"),
            ("classify", LinearSVC(dual=False, max_iter=10000)),
        ]
    )

    N_FEATURES_OPTIONS = [2, 4, 8]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
        {
            "reduce_dim": [PCA(iterated_power=7)],
            "reduce_dim__n_components": N_FEATURES_OPTIONS,
            "classify__C": C_OPTIONS,
        },
        {
            "reduce_dim": [SelectKBest(chi2)],
            "reduce_dim__k": N_FEATURES_OPTIONS,
            "classify__C": C_OPTIONS,
        },
    ]

Here we see that in order to specify the search space for the 'k' parameter of
``SelectKBest``, the user needs to identify its fully qualified path from the
root estimator (``pipe``) that will be passed to the grid search estimator,
i.e. ``reduce_dim__k``.  To construct this fully qualified parameter name, the
user must know that the ``SelectKBest`` estimator resides in a ``Pipeline``
step named ``reduce_dim`` and that the Pipeline is not further nested in
another estimator. Changing the step identifier ``reduce_dim`` would entail
a change to 5 lines in the above code snippet.

We also see that the options for ``classify__C`` need to be specified twice.
It should be possible to specify it only once. The use of a list of two separate
dicts of parameter spaces is similarly cumbersome: the only reason the
parameter space is duplicated is to handle the alternation of one step in the
pipeline; for all other steps, it makes sense for the candidate parameter
space to remain constant regardless of whether ``reduce_dim`` is a feature
selector or a PCA.

This SLEP proposes to add a methods to estimators that allow the user
to specify candidates or distributions for local parameters on a specific
estimator estimator instance::

    svc = LinearSVC(dual=False, max_iter=10000).set_search_grid(C=C_OPTIONS)
    pipe = Pipeline(
        [
            # the reduce_dim stage is populated by the param_grid
            ("reduce_dim", "passthrough"),
            ("classify", svc),
        ]
    ).set_search_grid(reduce_dim=[
        PCA(iterated_power=7).set_search_grid(n_components=N_FEATURES_OPTIONS),
        SelectKBest().set_search_grid(k=N_FEATURES_OPTIONS),
    ])

With this use of ``set_search_grid``, ``GridSearchCV(pipe)`` would not need the
parameter grid to be specified explicitly. Instead, a recursive descent through
``pipe``'s parameters allows it to reconstruct exactly the grid used in the
example above.

Such functionality therefore allows users to:

* easily define a parameter space together with the estimator they relate to,
  improving code cohesion.
* establish a library of estimator configurations for reuse, reducing repeated
  code, and reducing setup costs for auto-ML approaches. As such, this change
  helps to enable :issue:`5004`.
* avoid work modifying the parameter space specification when a composite
  estimator's strucutre is changed, or a Pipeline step is renamed.
* more comfortably specify search spaces that include the alternation of a
  step in a Pipeline (or ColumnTransformer, etc.), creating a form of
  conditional dependency in the search space.

History
-------

:issue:`5082`, :issue:`7608` and :issue:`19045` have all raised associating
parameter search spaces directly with an estimator instance, while this
has been supported by third party packages [1]_, [2]_. :issue:`21784` proposed
a ``GridFactory``, but feedback suggested that methods on each estimator
was more usable than an external utility.

This proposal pertains to the Scikit-learn Roadmap entry "Better support for
manual and automatic pipeline building" dating back to 2018.

Implementation
--------------

Four public methods will be added to ``BaseEstimator``::

    def set_search_grid(self, **grid: List[object]):
        """Sets candidate values for parameters in a search

        These candidates are used in grid search when a parameter grid is not
        explicitly specified. They are also used in randomized search in the
        case where set_search_rvs has not been used for the corresponding
        parameter.

        Note that this parameter space has no effect when the estimator's own
        ``fit`` method is called, but can be used by model selection utilities.

        As with :meth:`set_params`, update semantics apply, such that
        ``set_search_grid(param1=['a', 'b'], param2=[1, 2]).set_search_grid(param=['a'])``
        will retain the candidates set for ``param2``. To reset the grid,
        each parameter's candidates should be set to ``[]``.

        Parameters
        ----------
        grid : Dict[Str, List[object]]
            Keyword arguments define the values to be searched for each
            specified parameter.

            Keywords must be valid parameter names from :meth:`get_params`.

        Returns
        -------
        self : Estimator
        """
        ...

    def get_grid(self):
        """Retrieves current settings for parameters where search candidates are set

        Note that this only reflects local parameter candidates, and a grid
        including nested estimators can be constructed in combination with
        `get_params`.

        Returns
        -------
        dict
            A mapping from parameter name to a list of values. Each parameter
            name should be a member of `self.get_params(deep=False).keys()`.
        """
        ...

    def set_search_rvs(self, **distribution):
        """Sets candidate values for parameters in a search

        These candidates are used in randomized search when a parameter
        distribution is not explicitly specified. For parameters where
        no distribution is defined and a grid is defined, those grid values
        will also be used.

        As with :meth:`set_params`, update semantics apply, such that
        ``set_search_rvs(param1=['a', 'b'], param2=[1, 2]).set_search_grid(param=['a'])``
        will retain the candidates set for ``param2``. To reset the grid,
        each parameter's candidates should be set to ``[]``.

        Parameters
        ----------
        distribution : mapping from str to RV or list
            Keyword arguments define the distribution to be searched for each
            specified parameter.
            Distributions may be specified either as an object with the method
            ``rvs`` (see :mod:`scipy.stats`) or a list of discrete values with
            uniform distribution.

            Keywords must be valid parameter names from :meth:`get_params`.

        Returns
        -------
        self : Estimator
        """
        ...
    
    def get_distribution(self):
        """Retrieves current settings for parameters where a search distribution is set

        Note that this only reflects local parameter candidates, and a joint distribution
        including nested estimators can be constructed in combination with
        `get_params`.

        For parameters where ``set_search_rvs`` has not been used, but ``set_search_grid``
        has been, this will return the corresponding list of values specified in
        ``set_search_grid``.

        Returns
        -------
        dict
            A mapping from parameter name to a scipy-compatible distribution
            (i.e. with ``rvs``` method) or list of discrete values. Each parameter
            name should be a member of `self.get_params(deep=False).keys()`.
        """
        ...

The current distribution and grid values will be stored in a private
attribute on the estimator, and ``get_grid`` may simply return this value,
or an empty dict if undefined, while ``get_distribution`` will combine the
stored parameter distributions with ``get_grid`` values.
The attribute will be undefined by default upon construction of the estimator,
though in the future we could consider default grids being specified for
some estimator classes.

Parameter spaces should be copied in :ojb:`sklearn.base.clone`, so that a user
can overwrite only one parameter's space without redefining everything.
To facilitate this (in the absence of a polymorphic implementation of clone),
we might need to store the candidate grids and distributions in a known instance
attribute, or use a combination of `get_grid`, `get_distribution`, `get_params`
and `set_search_grid`, `set_search_rvs` etc. to perform `clone`.

Search estimators in `sklearn.model_selection` will be updated such that the
currently required `param_grid` and `param_distributions` parameters will now default
to 'extract'. The 'extract' value instructs the search estimator to construct
a complete search space from the provided estimator's `get_grid` (respectively,
`get_distribution`) return value together with `get_params`.
It recursively calls `get_grid` (and `get_distribution`) on any parametrized
objects (i.e. those with `get_params`) with this method that are descendent
from the given estimator, including:
* values in ``estimator.get_params(deep=True)``
* elements of list values in ``x.get_grid()`` or ``x.get_distribution()``
  as appropriate (disregarding rvs) for any `x` descendant of the estimator.

See the implementation of ``build_param_grid`` in Searchgrid [1]_, which applies
to the grid search case. This algorithm enables the specification of searches
over components in a pipeline as well as their parameters.

If the search estimator perfoming the 'extract' algorithm extracts an empty
grid or distribution altogether for the given estimator, it should raise a
`ValueError`, indicative of likely user error.  Note that this allows a step in a
`Pipeline` to have an empty search space as long as at least one step of that
`Pipeline` defines a non-empty search space.

Backward compatibility
----------------------

Where the user specifies an explicit grid, but one is also stored on the estimator
using `set_search_grid`, we will adopt legacy behaviour, and search with the
explicitly provided grid, maintaining backwards compatibility, and allowing a
manual override of the new behaviour.  This behavior will be made clear in the
docuemntation of parameters like `param_grid` and `param_distributions`.

Alternatives
------------

The fundamental change here is to associate parameter search configuration with each atomic estimator object.

Alternative APIs to do so include:

* Provide a function ``set_search_grid`` as Searchgrid [1]_ does, which takes an
  estimator instance and a parameter space, and sets a private
  attribute on the estimator object. This avoids cluttering the estimator's
  method namespace.
* Provide a `GridFactory` (see :issue:`21784`) which allows the user to
  construct a mapping from atomic estimator instances (and potentially estimator
  classes as a fallback) to their search spaces.
  Aside from not cluttering the estimator's namespace, this may have
  theoretical benefit in allowing the user to construct multiple search spaces
  for the same composite estimator. There are no known use cases for this
  benefit. This approach cannot retain the parameter space for a cloned estimator,
  potentially leading to surprising behavior.
* In the vein of `GridFactory`, but without a new object-oriented API:
  Provide a helper function which takes a mapping of estimator instances
  (and perhaps classes as a fall-back) to a shallow parameter search space, and
  transforms it into a traditional parameter grid.
  This helper function could be public, or else this instance-space mapping would
  become a new, *additional* way of specifying a parameter grid to `*SearchCV`.
  Inputs in this format would automatically be converted to traditional parameter
  grids. This has similar benefits and downsides as `GridFactory`, while avoiding
  introducing a new API and instead relying on plain old Python dicts.
  Having multiple distinct dict-based representations of parameter spaces is

Another questionable design is the separation of ``set_search_grid`` and ``set_search_rvs``.
These could be combined into a single method, such that
:class:`~sklearn.model_selection.GridSearchCV` rejects a call to `fit` where `rvs`
appear. This would make it harder to predefine search spaces that could be used
for either exhaustive or randomised searches, which may be a use case in Auto-ML.
That is, if we had a single set_search_space, an AutoML library that is set up to call it
would have to choose between setting RVs and grids. But this presumes a lot about how an
AutoML library using this API might look.

Discussion
----------

There are several areas to extend upon the above changes, such as to allow
easier construction of pipelines with alternative steps to be searched (see
``searchgrid.make_pipeline``), and handling alternative steps having
non-uniform distribution for randomised search.

There are also several other limitatins of the proposed solution:

Limitation: tied parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our solution does not directly meet the need for conditional
dependencies within a single estimator raised in :issue:`9610`, e.g::

    param_grid = [
        {
            "kernel": ["rbf"],
            "gamma": [.001, .0001],
            "C": [1, 10],
        },
        {
            "kernel": ["linear"],
            "C": [1, 10],
        }
    ]

Using the proposed API, the user would need to search over multiple instances
of the estimator, setting the parameter grids that could be searched with
conditional independence.

That issue also raises a request to tie parameters across estimators. While
the current proposal does not support this use case, the algorithm translating
an estimator to its deep parameter grid/distribution could potentially be adjusted
to recognise a ``TiedParam`` helper.

Limitation: continued use of ``__`` for search parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While this proposal reduces the use of dunders (``__``) for specifying parameter
spaces, they will still be rendered in ``*SearchCV``'s ``cv_results_`` attribute.
``cv_results_`` is similarly affected by large changes to its keys when small
changes are made to the composite model structure. Future work could provide
tools to make ``cv_results_`` more accessible and invariant to model structure.

References and Footnotes
------------------------

.. [1] Joel Nothman (2017). *SearchGrid*. Software Release.
   https://searchgrid.readthedocs.io/

.. [2] Guillaume Chevalier, Alexandre Brilliant and Eric Hamel (2019).
   *Neuraxle - A Python Framework for Neat Machine Learning Pipelines*.
   DOI:10.13140/RG.2.2.33135.59043. Software at https://www.neuraxle.org/

Copyright
---------

This document has been placed in the public domain.
