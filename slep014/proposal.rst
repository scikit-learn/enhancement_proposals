.. _slep_014:

=======================================
SLEP014: Parameter Spaces on Estimators
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
another estimator. Changing the name of ``reduce_dim`` would entail a change to
5 lines in the above code snippet.

We also see that the options for ``classify__C`` need to be specified twice.
Were the candidate values for ``C`` something that belonged to the
``LinearSVC`` estimator instance, rather than part of a grid specification, it
would be possible to specify it only once. The use of a list of two separate
dicts of parameter spaces is altogether avoidable, as the only reason the
parameter space is duplicated is to handle the alternation of one step in the
pipeline; for all other steps, it makes sense for the candidate parameter
space to remain constant regardless of whether ``reduce_dim`` is a feature
selector or a PCA.

This SLEP proposes to allow the user to specify candidates or distributions for
local parameters on a specific estimator estimator instance::

    svc = LinearSVC(dual=False, max_iter=10000).set_grid(C=C_OPTIONS)
    pipe = Pipeline(
        [
            # the reduce_dim stage is populated by the param_grid
            ("reduce_dim", "passthrough"),
            ("classify", svc),
        ]
    ).set_grid(reduce_dim=[
        PCA(iterated_power=7).set_grid(n_components=N_FEATURES_OPTIONS),
        SelectKBest().set_grid(k=N_FEATURES_OPTIONS),
    ])

With this use of ``set_grid``, ``GridSearchCV(pipe)`` would not need the
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

Implementation
--------------

TODO

Four public methods will be added to ``BaseEstimator``::

    def set_grid(self, **grid: List[object]):
        """Sets candidate values for parameters in a search

        These candidates are used in grid search when a paameter grid is not
        explicitly specified. They are also used in randomized search in the
        case where set_distribution has not been used for the corresponding
        parameter.

        As with :meth:`set_params`, update semantics apply, such that
        ``set_grid(param1=['a', 'b'], param2=[1, 2]).set_grid(param=['a'])``
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


setter, getter for grid.
setter, getter for distribution.
Overwriting behaviour

Private attribute on estimator, dynamically allocated on request

Grid Search update to handle param_grid='extract' using the algorithm
and implementation from searchgrid [1]_. If an empty grid is extracted
an error should be raised.
Randomized Search update to handle param_distributions='extract', using ``get_grid``
only to update the results of ``get_distribution``.

Parameter spaces should be copied in clone, so that a user can overwrite only
one parameter's space without redefining everything.

expected behaviour when a parameter name with `__` is used.

Backward compatibility
----------------------

No concerns

Alternatives
------------

TODO

no methhods, but storing on est
GridFactory (:issue:`21784`)

Alternative syntaxes

one call per param?
duplicate vs single method for grid vs  distbn

make_pipeline alternative or extension to avoid declaring 'passthrough'

searchgrid [1]_, Neuraxle [2]_

Discussion
----------

raised in :issue:`19045`.

:issue:`9610`: our solution does not directly meet the need for conditional
dependencies within a single estimator, e.g::

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
    
That issue also raises a request to tie parameters across estimators. While
the current proposal does not support this use case, the algorithm translating
an estimator to its deep parameter grid/distribution could potentially be adjusted
to recognise a ``TiedParam`` helper.

searchgrid's implementation was mentioned in relation to
https://github.com/scikit-learn/scikit-learn/issues/7707#issuecomment-392298478

Not handled: ``__`` paths still used in ``cv_results_``

Non-uniform distributions on categorical values.

This section may just be a bullet list including links to any discussions
regarding the SLEP:

- This includes links to mailing list threads or relevant GitHub issues.


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