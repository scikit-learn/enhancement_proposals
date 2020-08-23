.. _slep_001:

==============================================
SLEP001: Transformers that modify their target
==============================================

.. topic:: **Summary**

    Transformers implement::

        self = estimator.fit(X, y=None)
        X_transform = estimator.transform(X)
        estimator.fit(X, y=None).transform(X) == estimator.fit_transform(X, y)

    Within a chain or processing sequence of estimators, many usecases
    require modifying y. How do we support this?

    Doing many of these things is possible "by hand". The question is:
    how to avoid writing custom connecting logic.

.. contents:: Table of contents
   :depth: 2

Rational
========

Summary of the contract of transformers
----------------------------------------

* .transform(...) returns a data matrix X

* .transform(...) returns one feature vector for each sample of the input

* .fit_transform(...) is the same and .fit(...).transform(...)

Examples of usecases targetted
-------------------------------

#. Over sampling:

   #. Class rembalancing: over sampling the minority class in
      unbalanced dataset
   #. Data enhancement (nudgging images for instance)

#. Under-sampling

   #. Stateless undersampling: Take one sample out of two
   #. Stateful undersampling: apply clustering and transform to cluster
      centers
   #. Coresets: return a smaller number of samples and associated sample
      weights

#. Outlier detection:

   #. Remove outlier from train set
   #. Create a special class 'y' for outliers

#. Completing y:

   #. Missing data imputation on y
   #. Semi-supervised learning (related to above)

#. Data loading / conversion

   #. Pandas in => (X, y) out
   #. Images in => patches out
   #. Filename in => (X, y) with multiple samples (very useful in
      combination with online learning)
   #. Database query => (X, y) out

#. Aggregate statistics over multiple samples

   #. Windowing-like functions on time-series

   In a sense, these are dodgy with scikit-learn's cross-validation API
   that knows nothing about sample structure. But the refactor of the CV
   API is really helping in this regard.

____

These usecases pretty much require breaking the contract of the
Transformer, as detailed above.

The intuition driving this enhancement proposal is that the more the
data-processing pipeline becomes rich, the more the data grow, the more
the usecases above become important.

Enhancements proposed
=====================

Option A: meta-estimators
-------------------------

Proposal
........

This option advocates that any transformer-like usecase that wants to
modify y or the number of samples should not be a transformer-like but a
specific meta-estimator. A core-set object would thus look like:

* From the user perspective::

     from sklearn.sample_shrink import BirchCoreSet
     from sklearn.ensemble import RandomForest
     estimator = BirchCoreSet(RandomForest())

* From the developer perspective::

     class BirchCoreSet(BaseEstimator):

        def fit(self, X, y):
            # The logic here is wrong, as we need to handle y:
            super(BirchCoreSet, self).fit(X)
            X_red = self.subcluster_centers_
            self.estimator_.fit(X_red)

Benefits
.........

#. No change to the existing API

#. The meta-estimator pattern is very powerful, and pretty much anything
   is possible.

Limitations
............

The different limitations listed below are variants of the same
conceptual difficulty

#. It is hard to have mental models and garantees of what a
   meta-estimator does, as it is by definition super versatile

   This is both a problem for the beginner, that needs to learn them on
   an almost case-by-case basis, and for the advanced user, that needs to
   maintain a set of case-specific code

#. The "estimator heap" problem.

   Here the word heap is used to denote the multiple pipelines and
   meta-estimators. It corresponds to what we would naturally call a
   "data processing pipeline", but we use "heap" to avoid confusion with
   the pipeline object.

   Heaps combining many steps of pipelines and meta-estimators become
   very hard to inspect and manipulate, both for the user, and for
   pipeline-management (aka "heap-management") code. Currently, these
   difficulties are mostly in user code, so we don't see them too much in
   scikit-learn. Here are concrete examples

   #. Trying to retrieve coefficients from a model estimated in a
      "heap". Eg:

      * you know there is a lasso in your stack and you want to
        get it's coef (in whatever space that resides?):
        ``pipeline.named_steps['lasso'].coef_`` is possible.

      * you want to retrieve the coef of the last step:
        ``pipeline.steps[-1][1].coef_`` is possible.

      With meta estimators this is tricky.
      Solving this problem requires
      https://github.com/scikit-learn/scikit-learn/issues/2562#issuecomment-27543186
      (this enhancement proposal is not advocating to solve the problem
      above, but pointing it out as an illustration)

   #. DaskLearn has modified the logic of pipeline to expose it as a
      computation graph. The reason that it was relatively easy to do is
      that there was mostly one object to modify to do the dispatching,
      the Pipeline object.

   #. A future, out-of-core "conductor" object to fit a "heap" in out of
      core by connecting it to a data-store would need to have a
      representation of the heap. For instance, when chaining random
      projections with Birch coresets and finally SGD, the user would
      need to specify that random projections are stateless, birch needs
      to do one pass of the data, and SGD a few. Given this information,
      the conductor could orchestrate pull the data from the data source,
      and sending it to the various steps. Such an object is much harder
      to implement if the various steps are to be combined in a heap.
      Note that the scikit-learn pipeline can only implement a linear
      "chain" like set of processing. For instance a One vs All will
      never be able to be implemented in a scikit-learn pipeline.

      This is not a problem in non out-of-core settings, in the sense
      that the BirchCoreSet meta-estimator would take care of doing a
      pass on the data before feeding it to its sub estimator.

In conclusion, meta-estimators are harder to comprehend (problem 1) and
write (problem 2).

That said, we will never get rid of meta estimators. It is a very
powerful pattern. The discussion here is about extending a bit the
estimator API to have a less pressing need for meta-estimators.

Option B: transformer-like that modify y
----------------------------------------

.. topic:: **Two variants**

    1. Changing the semantics of transformers to modify y and return
       something more complex than a data matrix X

    2. Introducing new methods (and a new type of object)

    There is an emerging consensus for option 2.

.. topic:: **``transform`` modifying y**

   Variant 1 above could be implementing by allowing transform to modify
   y. However, the return signature of transform would be unclear.

   Do we modify all transformers to return a y (y=None for unsupervised
   transformers that are not given y?). This sounds like leading to code
   full of surprises and difficult to maintain from the user perspective.

   We would loose the contract that the number of samples is unchanged by
   a transformer. This contract is very useful (eg for model selection:
   measuring error for each sample).

   For these reasons, we feel new methods are necessary.

Proposal
.........

Introduce a ``TransModifier`` type of object with the following API
(names are discussed below):

* ``X_new, y_new = estimator.fit_modify(X, y)``

* ``X_new, y_new = estimator.trans_modify(X, y)``

Or:

* ``X_new, y_new, sample_props = estimator.fit_modify(X, y)``

* ``X_new, y_new, sample_props = estimator.trans_modify(X, y)``

Contracts (these are weaker contracts than the transformer:

* Neither ``fit_modify`` nor ``trans_modify`` are guarantied to keep the
  number of samples unchanged.

* ``fit_modify`` may not exist (questionnable)

Design questions and difficulties
.................................

Should there be a fit method?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In such estimators, it may not be a good idea to call fit rather than
fit_modify (for instance in coreset).


How does a pipeline use such an object?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In particular at test time?

#. Should there be a transform method used at test time?

#. What to do with objects that implement both ``transform`` and
   ``trans_modify``?

**Creating y in a pipeline makes error measurement harder** For some
usecases, test time needs to modify the number of samples (for instance
data loading from a file). However, these will by construction a problem
for eg cross-val-score, as in supervised settings, these expect a y_true.
Indeed, the problem is the following:

- To measure an error, we need y_true at the level of
  `sklearn.model_selection.cross_val_score` or
  `sklearn.model_selection.GridSearchCV`

- y_true is created inside the pipeline by the data-loading object.

It is thus unclear that the data-loading usecases can be fully
integrated in the CV framework (which is not an argument against
enabling them).

|

For our CV framework, we need the number of samples to remain
constant: for each y_pred, we need a corresponding y_true.

|

**Proposal 1**: use transform at ``predict`` time.

#. Objects implementing both ``transform`` and ``trans_modify`` are valid

#. The pipeline's ``predict`` method use ``transform`` on its intermediate
   steps

The different semantics of ``trans_modify`` and ``transform`` can be very useful,
as ``transform`` keeps untouched the notion of sample, and ``y_true``.

|

**Proposal 2** Modify the scoring framework

One option is to modify the scoring framework to be able to handle
these things, the scoring gets the output of the chain of
trans_modify for y. This should rely on clever code in the ``score`` method
of pipeline. Maybe it should be controlled by a keyword argument on the
pipeline, and turned off by default.


How do we deal with sample weights and other sample properties?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This discussion feeds in the ``sample_props`` discussion (that should
be discussed in a different enhancement proposal).

The suggestion is to have the sample properties as a dictionary of
arrays ``sample_props``.

**Example usecase** useful to think about sample properties: coresets:
given (X, y) return (X_new, y_new, weights) with a much smaller number
of samples.

This example is interesting because it shows that TransModifiers can
legitimately create sample properties.

**Proposed solution**:

TransModifiers always return (X_new, y_new, sample_props) where
sample_props can be an empty dictionary.


Naming suggestions
..................

In term of name choice, the rational would be to have method names that
are close to 'fit' and 'transform', to make discoverability and
readability of the code easier.

* Name of the object (referred in the docs):
  - TransModifier
  - TransformPipe
  - PipeTransformer

* Method to fit and apply on training
  - fit_modify
  - fit_pipe
  - pipe_fit
  - fit_filter

* Method to apply on new data
  - trans_modify
  - transform_pipe
  - pipe_transform

Benefits
........

* Many usecases listed above will be implemented scikit-learn without a
  meta-estimator, and thus will be easy to use (eg in a pipeline). Many
  of these are patterns that we should be encouraging.

* The API being more versatile, it will be easier to create
  application-specific code or framework wrappers (ala DaskLearn) that
  are scikit-learn compatible, and thus that can be used with the
  parameter-selection framework. This will be especially true for ETL
  (extract transform and load) pattern.

Limitations
...........

* Introducing new methods, and a new type of estimator object. There are
  probably a total of **3 new methods** that will get introduced by this
  enhancement: fit_modify, trans_modify, and partial_fit_modify.

* Cannot solve all possible cases, and thus we will not get rid of
  meta-estimators.

TODO
====

* Implement an example doing outlier filtering

* Implement an example doing data downsampling
