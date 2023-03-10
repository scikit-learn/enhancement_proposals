.. _slep_021:

==================================================
SLEP021: Unified API to compute feature importance
==================================================

:Author: Thomas J Fan, Guillaume Lemaitre
:Status: Draft
:Type: Standards Track
:Created: 2023-03-09

Abstract
--------

This SLEP proposes a common API for computing feature importance.

Detailed description
--------------------

Motivation
~~~~~~~~~~

Data scientists rely on feature importance when inspecting a trained model.
Feature importance is a measure of how much a feature contributes to the
prediction and thus gives insights on the model the predictions provided by
the model.

However, there is currently not a single method to compute feature importance.
All available methods are designed upon axioms or hypotheses that are not
necessarly respected in practice.

Some work in scikit-learn has been done to provide documentation to highlight
the limitations of some implemented methods. However, there is currently not
a common way to expose feature importance in scikit-learn. In addition, for
some historical reasons, some estimators (e.g. decision tree) provide a single
feature importance that could be used as the "method-to-use" to analyse the
model. It is problematic since there is not defacto standard to analyse the
feature importance of a model.

Therefore, this SLEP proposes an API for providing feature importance allowing
to be flexible to switch between methods and extensible to add new methods. It
is a follow-up of initial discussions from :issue:`20059`.

Current state
~~~~~~~~~~~~~

Available methods
^^^^^^^^^^^^^^^^^

The following methods are available in scikit-learn to provide some feature
importance:

- The function :func:`sklearn.inspection.permutation_importance`. It requests
  a fitted estimator and a dataset. Addtional parameters can be provided. The
  method returns a `Bunch` containing 3 attributes: all decrease in score for
  all repeatitions, the mean, and the standard deviation across the repeats.
  This method is therefore estimator agnostic.
- The linear estimators have a `coef_` attributes once fitted.
- The decision tree-based estimators have a `feature_importances_` attribute
  once fitted.

Use cases
^^^^^^^^^

The first usage of feature importance is to inspect a fitted model. Usually,
the feature importance will be plotted to visualize the importance of the
features::

   >>> tree = DecisionTreeClassifier().fit(X_train, y_train)
   >>> plt.barh(X_train.columns, tree.feature_importances_)

The analysis can be done further by checking the variance of the feature
importance. :func:`sklearn.inspection.permutation_importance` already provides
a way to do that since it repeats the computation several time. For the model
specific feature importance, the user can use cross-validation to get an idea
of the dispersion::

   >>> cv_results = cross_validate(tree, X_train, y_train, return_estimator=True)
   >>> feature_importances = [est.feature_importances_ for est in cv_results["estimator"]]
   >>> plt.boxplot(feature_importances, labels=X_train.columns)

The second usage is about the model selection. Meta-estimator such as
:class:`sklearn.model_selection.SelectFromModel` internally use an array of
length `(n_features,)` to select feature and retrain a model on this subset of
feature.

By default :class:`sklearn.model_selection.SelectFromModel` relies on the
estimator to expose `coef_` or `feature_importances_`::

   >>> SelectFromModel(tree).fit(X_train, y_train)  # `tree` exposes `feature_importances_`

For more flexbilibity, a string can be provided::

   >>> linear_model = make_pipeline(StandardScaler(), LogisticRegression())
   >>> SelectFromModel(
   ...     linear_model, importance_getter="named_steps.logisticregression.coef_"
   ... ).fit(X_train, y_train)

:class:`sklearn.model_selection.SelectFromModel` rely by default on
the estimator to expose a `coef_` or `feature_importances_` attribute. It is
also possible to provide a string corresponding the attribute name returning
the feature importance. It allows to deal with estimator embedded inside a
pipeline, for instance. Finally, a callable taking an estimator and returning
a NumPy array can also be provided.

Current pitfalls
~~~~~~~~~~~~~~~~

Solution
~~~~~~~~

Plotting
^^^^^^^^

Add a new :class:`sklearn.inspection.FeatureImportanceDisplay` class to
:mod:`sklearn.inspection`. Two methods could be useful for this display: (i)
:meth:`sklearn.inspection.FeatureImportanceDisplay.from_estimator` to plot a
a single estimate of feature importance and (ii)
:meth:`sklearn.inspection.FeatureImportanceDisplay.from_cv_results` to plot a
an estimate of the feature importance together with the variance.

The display should therefore be aware how to retrieve the feature importance
given the estimator.

Discussion
----------

Issues where some discussions related to feature importance has been discussed:
:issue:`20059`, :issue:`21170`.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open Publication
   License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain [1]_.
