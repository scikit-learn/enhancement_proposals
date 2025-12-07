.. _slep_025:

=========================================
SLEP025: Losing Accuracy in Scikit-Learn
=========================================

:Author: Christian Lorentzen
:Status: Draft
:Type: Standards Track
:Created: 2025-12-07
:Resolution: TODO <url> (required for Accepted | Rejected | Withdrawn)

Abstract
--------

This SLEP proposes to rectify the default ``score`` method. Currently, the ease of
``classifier.score(X, y)`` favors the use of *accuracy*, which has many well known
deficiencies. This SLEP changes the default scoring method.

Motivation
----------

As it stands, *accuracy* is the most used metric for classifiers in scikit-learn. This
is manifest in `classifier.score(..)` which applies accuracy. While the original goal
might have been to provide a score method that works for all classifiers, the actual
implication was the blind usage, without critical thinking, of the accuracy score.
This has mislead many researchers and users because accuracy is well known for its
severe deficiencies: To the point, it is not a *strictly proper scoring rule* and
scikit-learn's implementation hard-coded a probability threshold of 50% into it.

This situation calls for a correction. Ideally, scikit-learn provides good defaults
or fosters a conscious decision by users, e.g. by forcing engagement with the subject,
see [2]_ subsection "Which scoring function should I use?".

Solution
--------

The solution is a multi-step approach:

1. Introduce the new keyword ``scoring`` to the ``score`` method. The default for
   classifiers is ``scoring="accuracy"``, for regressors ``scoring="r2"``.
2. Deprecate the default ``"accuracy"``.
3. Set a new default.

There are three questions with this approach:

a. The time frame of the deprecation period. Should it be longer than the usual 2 minor
   releases? Should step 1 and 2 happen in the same minor release?
b. What is the new default scoring parameter in step 3? Possibilities are
   - D2 Brier score, which is basically the same as R2 for regressors.
   - The objective function of the estimator, i.e. the penalized log loss for
     ``LogisticRegression``.

   The fact that different scoring metrics focus on different things, i.e. ``predict``
   vs. ``predict_proba``, and not all classifiers provide ``predict_proba`` complicates
   a unified choice.

Backward compatibility
----------------------

The outlined solution would be feasible within the usual deprecation strategy of
scikit-learn releases.

Alternatives
------------

An alternative is to remove the ``score`` method altogether. Scoring metrics are well
available in scikit-learn, see ``sklearn.metric`` module and [2]_. The advantages of
removing ``score`` are:

- An active choice by the user is triggered as there is no more default.
- Defaults for ``score`` are tricky anyway. Different estimators estimate different
  things and the output of their ``score`` method most likely is not comparable, e.g.
  consider a hinge loss based SVM vs. log loss based logistic regression.

Disadvantages:

- Disruption of the API.
- More imports required and a bit longer code as compared to just
  ``my_estimator.score(X, y)``.

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
