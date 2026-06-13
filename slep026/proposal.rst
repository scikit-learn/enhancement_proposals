.. _slep_026:

***********************************************************************
SLEP026: Add ``scoring`` to the ``score`` Method and Remove the Default
***********************************************************************

:Author: Christian Lorentzen
:Status: Draft
:Type: Standards Track
:Created: 2026-02-26
:Resolution: <url> (required for Accepted | Rejected | Withdrawn)

.. contents:: Table of contents
   :depth: 2

1. Abstract
===========

This SLEP proposes to add a ``scoring`` parameter to the ``score`` method and to remove
the default value, for classifiers as well as regressors. The  current status is that
``classifier.score`` always uses *accuracy*, ``regressor.score`` most often uses *R2*.

2. Motivation
=============

The main motivation is to avoid the blind usage of a default score that, in many use
cases, is not a good default. This is one of the conclusions of the rejected SLEP 25
[3]_: *There is no good default score*. (Or at least, scikit-learn core contributors
could not agree on any.)

The ``score`` method of estimators is used to compare models and as default in cross
validation tools, for example in ``cross_validate``.
The result of using ``score`` may not be what a user wants or expects.
Classifiers, for instance, default to *accuracy* which is only a suitable metric if
false positives have about the same cost as false negatives.
Additionally, for classifiers, a user might favor ``predict_proba`` in her application
instead of the 50%-thresholded ``predict``. In this case a proper scoring rule like
log loss and Brier score is preferred from a methodological point of view.
One could say: *The default score is too easy to use*.

As applications of using a score are so wide-ranging, we acknowledge that
searching for a good default covering most of them is a futile quest. Instead, we would
like to foster a conscious decision by users and engagement with the subject.
Some guidelines are already provided in the user guide, see [2]_ subsection
"Which scoring function should I use?".

Note that ``score`` is redundant. The ``sklearn.metrics`` module provides all the
functionality needed.

3. Analysis
===========

3.1 The default ``score`` method
--------------------------------
Some desirable properties for a default ``score`` are:

- Comparability across different estimators
- Alignment with purpose of prediction
- Ease of explanation

With the status quo, the comparability for classifiers is given because all classifiers
inherit the accuracy score from ``ClassifierMixin``. Most regressors inherit the R2
score from ``RegressorMixin``; there are, however, exceptions.

The second point, purpose, is much harder to fulfil and often contradicts the
comparability.
If the purpose is model selection, e.g. (hyper-) parameter search via ``GridSearchCV``,
a good choice is the loss underlying the estimator. This is currently only the case for
the minority of classifiers.
For model comparison and assessment, e.g. comparing a ``DecisionTreeRegressor`` with a
``Lasso`` or assessing a final selected model, it is desirable if the score is aligned
with the application purpose of the model. This depends heavily on the use case at hand
and the variety of use cases is so large that a unified choice seems futile.
This can be seen by the simple distinction between ``predict`` and ``predict_proba``
of classifiers. Which method is applied in the end? For instance, the default accuracy
assess the 50%-thresholded ``predict``, and not ``predict_proba``. For different costs
of false positives and false negatives, the cost weighted misclassification error is
more appropriate (but not available in scikit-learn).
For assessment of predictions of the probability, ``predict_proba``, proper scoring
rules are preferred (at least, they are invented for that purpose).

While accuracy might be the simplest score to explain, R2 is already not very intuitive
and, evaluated out-of-sample (not on the training set), it even lacks its simple
interpretation in terms of variances.

**Conclusion** as stated above: *There is no good default score.*

3.2 (Mis-) Use cases
--------------------

1. Using R2 for quantile regression. Note that ``QuantileRegressor.score`` computes R2.
   Hint: This is bad practice because R2, like squared error, are optimized by the
   expected value while here one clearly wants a score that is optimized by the
   corresponding quantile.
2. Using a summary score like R2 or AUC with ``LeaveOneOut``, see
   https://github.com/scikit-learn/scikit-learn/issues/5097.
   A score is best computable per single sample/observation,
   ``scoring(y_true=[1], y_pred=[2.0])``.
3. Compare different scores/metrics as with ``GammaRegressor.score(X, y)`` and
   ``Lasso.score(X, y)``. The first returns D2 Gamma deviance, the latter R2.
   Note the trade-off because D2 Gamma is aligned with the underlying loss that
   ``GammaRegressor`` minimizes (on the training set).
4. Your application has very different costs for false positives versus false
   negatives, e.g. fraud detection in financial services or many medical applications,
   and you use (as of version 1.8)
   
   - ``FixedThresholdClassifier(estimator=.., threshold=0.9).fit(X, y).score(X_test, y_test)``; or
   - ``RidgeClassifierCV().fit(X, y)``; or
   - ``cross_validate(RandomForestClassifier(), X, y)``.
   
   All three examples default to using the ``score`` method returning accuracy. The
   first example with ``FixedThresholdClassifier`` is particularly misleading as the
   user is already "on the right track" (``score`` would better be the cost weighted
   misclassification error with cost ratio 0.9).
5. Early stopping is not aligned with the loss of the estimator as in
   ``SGDRegressor(loss="epsilon_insensitive", early_stopping=True).fit(X, y)``, because
   it uses ``estimator.score(X, y)`` returning R2 to track the improvement per
   iteration. Similar for ``SGDClassifier(loss="hinge")`` and other losses.


3.3 Details per estimator
-------------------------

The following tables provide an overview of scores/metrics used in different estimators
and functions. It is written as of scikit-learn version 1.9.

+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
| estimator                          | type      | what                  | score used                    | comment                                                          |
+====================================+===========+=======================+===============================+==================================================================+
| ``ClassifierMixin``                | method    | ``score``             | accuracy                      |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
| ``RegressorMixin``                 | method    | ``score``             | R2                            |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
| ``BaggingClassifier``              | attribute | ``oob_score_``        | accuracy                      |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
| ``BaggingRegressor``               | attribute | ``oob_score_``        | R2                            |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
| ``ExtraTreesClassifier``           | attribute | ``oob_score_``        | accuracy                      |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
| ``ExtraTreesRegressor``            | attribute | ``oob_score_``        | R2                            |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
| ``RandomForestClassifier``         | attribute | ``oob_score_``        | accuracy                      |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
| ``RandomForestRegressor``          | attribute | ``oob_score_``        | R2                            |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
| ``GradientBoostingClassifier``     | parameter | ``loss``              | default ``"log_loss"``        | early stopping uses ``loss``                                     |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | attribute | ``oob_scores_``       | given ``loss``                |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | attribute | ``train_score_``      | given ``loss``                |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``score``             | accuracy                      |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
| ``GradientBoostingRegressor``      | parameter | ``loss``              | default ``"squared_error"``   | early stopping uses ``loss``                                     |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | attribute | ``oob_scores_``       | given ``loss``                |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | attribute | ``train_score_``      | given ``loss``                |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
| ``HistGradientBoostingClassifier`` | parameter | ``loss``              | default ``"log_loss"``        | always log loss                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | parameter | ``scoring``           | default ``"loss"``            | for early stopping                                               |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``score``             | accuracy                      |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | attribute | ``train_score_``      |                               |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | attribute | ``validation_score_`` |                               |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``HistGradientBoostingRegressor``   | parameter | ``loss``              | default ``"squared_error"``   |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | parameter | ``scoring``           | default ``"loss"``            | for early stopping                                               |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``score``             | R2                            | R2 bad if loss is quantile loss, absolute error or Gamma         |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``LogisticRegressionCV``            | parameter | ``scoring``           | default accuracy              | Will change in 1.11 to (negative) log loss                       |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | attribute | ``scores_``           | given ``scoring``             |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``scores``            | given ``scoring``             |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``TweedieRegressor``                | parameter | ``power``             | Tweedie deviance              | Similar ``PoissonRegressor`` and ``GammaRegressor``              |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``score``             | D2 Tweedie deviance           | As given by ``power``                                            |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``SGDClassifier``                   | parameter | ``loss``              | given ``loss``                |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``score``             | accuracy                      | early stopping based on ``score``                                |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``SGDRegressor``                    | parameter | ``loss``              | given ``loss``                |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``score``             | R2                            | early stopping based on ``score``                                |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``QuantileRegressor``               | method    | ``score``             | R2                            | R2 bad if loss is quantile loss                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``MLPClassifier``                   | method    | ``score``             | accuracy                      | early stopping based on ``score``, fit minimizes log loss        |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``MLPRegressor``                    | parameter | ``loss``              | given ``loss``                |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``score``             | R2                            | early stopping based on ``score``                                |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``FixedThresholdClassifier``        | method    | ``score``             | accuracy                      | Given ``threshold`` implies a cost ratio different than accuracy |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``TunedThresholdClassifierCV``      | parameter | ``scoring``           | default ``balanced_accuracy`` |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``DecisionTreeClassifier``          | parameter | ``criterion``         | default ``"gini"``            | Gini is (from) Brier score                                       |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``score``             | accuracy                      | for all choices of ``criterion``                                 |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``DecisionTreeRegressor``           | parameter | ``criterion``         | default ``"squared_error"``   |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``score``             | R2                            | for all choices of ``criterion``                                 |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``ExtraTreeClassifier``             | parameter | ``criterion``         | default ``"gini"``            | Gini is (from) Brier score                                       |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``score``             | accuracy                      |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|``ExtraTreeRegressor``              | parameter | ``criterion``         | default ``"squared_error"``   |                                                                  |
+                                    +-----------+-----------------------+-------------------------------+------------------------------------------------------------------+
|                                    | method    | ``score``             | R2                            |                                                                  |
+------------------------------------+-----------+-----------------------+-------------------------------+------------------------------------------------------------------+

==========================  ==============  ================  ===========================================
CV tool                     parameter       score used        comment
==========================  ==============  ================  ===========================================
``GridSearchCV``            ``scoring``     default ``None``  ``None`` means estimator's ``score`` method
``HalvingGridSearchCV``     ``scoring``     default ``None``  ``None`` means estimator's ``score`` method
``HalvingRandomSearchCV``   ``scoring``     default ``None``  ``None`` means estimator's ``score`` method
``RandomizedSearchCV``      ``scoring``     default ``None``  ``None`` means estimator's ``score`` method
``cross_val_score``         ``scoring``     default ``None``  ``None`` means estimator's ``score`` method
``cross_validate``          ``scoring``     default ``None``  ``None`` means estimator's ``score`` method
``permutation_test_score``  ``scoring``     default ``None``  ``None`` means estimator's ``score`` method
``validation_curve``        ``scoring``     default ``None``  ``None`` means estimator's ``score`` method
``LearningCurveDisplay``    ``score_name``  default ``None``  ``None`` means estimator's ``score`` method
``ValidationCurveDisplay``  ``score_name``  default ``None``  ``None`` means estimator's ``score`` method
==========================  ==============  ================  ===========================================


4. Solution
===========
4.1. Add ``scoring`` to ``score`` and remove the default
--------------------------------------------------------
**The proposed solution is:**

- Add a ``scoring`` parameter to ``score`` such that a user can select the used score
  as it is already the case in different classes and functions.
  Keep the current default, now set explicitly.
- Deprecate and remove the default from the ``scoring`, for supervised classifiers,
  regressors and cross validation / grid search tools.
  Ideally, this deprecation adds an informative warning such that users can infer
  what to do and where to find more information.

4.1.1. Advantages
~~~~~~~~~~~~~~~~~

- No bad default score (no default at all) for your use case, remember:
  *There is no good default score*.
- An active choice of score/metric by the user is triggered as there is no more
  default.

4.1.1. Disadvantages
~~~~~~~~~~~~~~~~~~~~

- Calling ``score`` needs one more parameter to be set.
  Instead of::
  
      >>> estimator.score(X, y)
      
  one needs::
  
      >>> estimator.score(X, y, scoring="accuracy")
- Instantiating classes like ``GridSearchCV`` need one more explicitly set parameter.
  Same as above.

4.2. Backward compatibility
---------------------------

This change can be accomplished with the usual deprecation policy of scikit-learn.

4.3. Downstream packages
------------------------
Many packages inherit from ``ClassifierMixin`` and ``RegressorMixin``, some examples include
- LightGBM ``LGBMClassifier`` and ``LGBMRegressor``
- XGBoost ``XGBClassifier`` and ``XGBRegressor``
- sktorch ``NeuralNetClassifier`` and ``NeuralNetRegressor``

Examples of packages not inheriting:
- CatBoost ``CatBoostClassifier`` and ``CatBoostRegressor``

5. Alternatives
===============

5.1. Deprecate and then remove ``score`` from the API
-----------------------------------------------------
A possible solution is

- to deprecate the ``score`` method from the scikit-learn API; and
- to deprecate the default ``None`` of the ``scoring`` parameter of all CV tools; it
  must be explicitly set after the deprecation period.

Note that ``score`` is redundant. Every score is available in ``sklearn.metrics``.

5.1.1. Advantages
~~~~~~~~~~~~~~~~~

- No bad default score (no default at all) for your use case, remember:
  *There is no good default score*.
- An active choice of score/metric by the user is triggered as there is no more
  default.

5.1.1. Disadvantages
~~~~~~~~~~~~~~~~~~~~
- Disruption of the API.
- Very likely a major release for something not very marketable (a deprecation).
- More imports required and a bit longer code.
  Instead of::
  
      >>> estimator.score(X, y)
      
  one needs::
  
      >>> from sklearn.metrics import super_metric
      >>> super_metric(y_true=y, y_pred=estimator.predict(X))

- Deprecation: In principle, this deprecation can be accomplished with the usual
  deprecation policy of scikit-learn. It is, however, not backward compatible in the
  sense that, as with all deprecations, future versions won't work with code written
  for older versions: an ``AttributeError`` is raised if ``estimator.score(X, y)`` is
  called, a ``ValueError`` is raised if ``cv_tool(scoring=None)`` is called, a
  ``TypeError`` for ``cv_tool()`` (because ``scoring`` has no default value anymore).


5.2 Add ``scoring`` parameter to ``score``
------------------------------------------
Issue [4]_ proposes to add a ``scoring`` parameter to the ``score`` method:
``estimator.score(X, y, scoring=..)``.

This, however, is redundant. Every score is available in ``sklearn.metrics``.

5.3 Global config
-----------------
On top of adding a ``scoring`` argument of 5.1. a global config is added to
``sklearn.config_context`` to control the default score of ``score``.

Again, this is redundant. Every score is available in ``sklearn.metrics``.

5.4 SLEP 25: Change default of ``scoring`` parameter of ``score``
-----------------------------------------------------------------
A further extension of 5.1 is to additionally change the default parameter ``scoring``
to a new default. This corresponds to SLEP 25 [3]_ and was mainly rejected due to the
difficulty to agree on a new default, see also section 3.1.

The proposal included the new default ``scoring="d2_brier_score"`` for classifiers,
stay with for regressors ``scoring="r2"``.

After the discussion of SLEP 25 and its rejection, it seems pointless to the author of
this SLEP 26 to try with different default scores (e.g. mean squared error and mean
Brier score). The basic arguments against a change of defaults scores won't change.

5.5. Keep status quo
--------------------

Advantages:

- No change or breaking things for users
- No resources bound

Disadvantages:

- No change for users
- Potentially bad practice is continued
- Bad signal: scikit-learn community is resistant to change.


References and Footnotes
========================

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

.. [2] Scikit-Learn User Guide on "Metrics and Scoring"
    https://scikit-learn.org/stable/modules/model_evaluation.html

.. [3] :ref:`slep_025`:

.. [4] https://github.com/scikit-learn/scikit-learn/issues/28995

Copyright
=========

This document has been placed in the public domain. [1]_
