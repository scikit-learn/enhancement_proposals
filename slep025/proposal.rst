.. _slep_025:

==============================================
SLEP025: Losing Accuracy in Scikit-Learn Score
==============================================

:Author: Christian Lorentzen
:Status: Draft
:Type: Standards Track
:Created: 2025-12-07
:Resolution: TODO <url> (required for Accepted | Rejected | Withdrawn)

Abstract
--------

This SLEP proposes to rectify the default ``score`` method for scikit-learn
classifiers. Currently, the ease of ``classifier.score(X, y)`` favors the use of
*accuracy*, which has many well known deficiencies. This SLEP changes the default
scoring method.

Motivation
----------

As it stands, *accuracy* is the most used metric for classifiers in scikit-learn. This
is manifest in ``classifier.score(..)`` which applies accuracy. While the original goal
might have been to provide a score method that works for all classifiers, the actual
implication has been the blind usage, without critical thinking, of the accuracy score.
This has mislead many researchers and users because accuracy is well known for its
severe deficiencies: To the point, it is not a *strictly proper scoring rule* and
scikit-learn's implementation hard-coded a probability threshold of 50% into it by
relying on ``predict``.

This situation calls for a correction. Ideally, scikit-learn provides good defaults
or fosters a conscious decision by users, e.g. by forcing engagement with the subject,
see [2]_ subsection "Which scoring function should I use?".

Solution
--------

The solution is a multi-step approach:

1. Introduce the new keyword ``scoring`` to the ``score`` method. The default for
   classifiers is ``scoring="accuracy"``, for regressors ``scoring="r2"``.
2. Deprecate the default ``"accuracy"`` for classifiers.
3. After the release cycle, set a new default for classifiers: ``"d2_brier_score"``.

There are two main questions with this approach:

a. The time frame of the deprecation period. Should it be longer than the usual 2 minor
   releases? Should step 1 and 2 happen in the same minor release?
b. What is the new default scoring parameter in step 3?
   The fact that different scoring metrics focus on different things, i.e. ``predict``
   vs. ``predict_proba``, and not all classifiers provide ``predict_proba`` complicates
   a unified choice.
   Possibilities are
   - D2 Brier score, ``"d2_brier_score"``, which is basically the same as R2 for
     regressors,
   - the objective function of the estimator, i.e. the penalized log loss for
     ``LogisticRegression``.

Proposals:

a. Use a deprecation period of 4 instead of 2 minor releases which amounts to 2 years
   and do step 1 and 2 at the same time (in the same release).
   Reasoning: It is a deprecation that is doable within the current deprecation
   habit of minor releases. It should be longer than the usual 2 minor releases because
   of it's big impact.
   A major release just because of such a deprecation is not very attractive (or
   marketable).
b. Use D2 Brier score.
   Reasoning: Scores will be compared among different models. Therefore, the model
   specific loss is not suitable.
   Note that the Brier score and hence also the D2 Brier score are strictly proper
   scoring rules (or strictly consistent scoring functions) for the probability
   predictions with ``predict_proba``. At the same time, Brier score returns a valid
   score even for ``predict`` (in case a classifier has no ``predict_proba``), in
   constrast to log loss (which returns infinity for false certainty). On top, this
   would result in classifiers and regressors having the same
   score (it's just a different name), returning values in the range [-inf, 1].
   Note that the D2 Brier score as a skill score (a relavitve score to a baseline) is
   invariant under a multiplicative factor, e.g. specified by ``scale_by_half``. It is
   given by ``1 - MSE(model predictions) / MSE(mean of data)``.

Backward compatibility
----------------------

The outlined solution would be feasible within the usual deprecation strategy of
scikit-learn releases.

Alternatives
------------

Removing
^^^^^^^^
An alternative is to remove the ``score`` method altogether. Scoring metrics are well
available in scikit-learn, see ``sklearn.metric`` module and [2]_. The advantages of
removing ``score`` are:

- An active choice by the user is triggered as there is no more default.
- Defaults for ``score`` are tricky anyway. Different estimators estimate different
  things and the output of their ``score`` method most likely is not comparable, e.g.
  consider a hinge loss based SVM vs. log loss based logistic regression.

Disadvantages:

- Disruption of the API.
- Very likely a major release for something not very marketable.
- More imports required and a bit longer code as compared to just
  ``my_estimator.score(X, y)``.

Keep status quo
^^^^^^^^^^^^^^^

Advantages:

- No change or breaking things for users
- No ressources bound

Disadvantages:
- No change for users
- Bad practice is continued
- Bad signal: scikit-learn community is unable to rectify serious grievance

Discussion
----------

The following issues contain discussions on this subject:

- https://github.com/scikit-learn/scikit-learn/issues/28995


References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

.. [2] Scikit-Learn User Guide on "Metrics and Scoring"
    https://scikit-learn.org/stable/modules/model_evaluation.html

Copyright
---------

This document has been placed in the public domain. [1]_
