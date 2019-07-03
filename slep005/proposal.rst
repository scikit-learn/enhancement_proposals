.. _slep_005:

=============
Resampler API
=============

:Author: Oliver Rausch (oliverrausch99@gmail.com),
         Christos Aridas (char@upatras.gr),
         Guillaume Lemaitre (g.lemaitre58@gmail.com)
:Status: Draft
:Created: created on, in 2019-03-01
:Resolution: <url>

Abstract
--------

We propose the inclusion of a new type of estimator: resampler. The
resampler will change the samples in ``X`` and ``y`` and return both
``Xt`` and ``yt``. In short:

* a new verb/method that all resamplers must implement is introduced:
  ``fit_resample``.
* resamplers are able to reduce and/or augment the number of samples in
  ``X`` and ``y`` during ``fit``, but will perform no changes during
  ``predict``.
* to facilitate this behavior a new meta-estimator (``ResampledTrainer``) that
  allows for the composition of resamplers and estimators is proposed.
  Alternatively we propose changes to ``Pipeline`` that also enable similar
  compositions.


Motivation
----------

Sample reduction or augmentation are common parts of machine-learning
pipelines. The current scikit-learn API does not offer support for such
use cases.

Possible Usecases
.................

* Sample rebalancing to correct bias toward class with large cardinality.
* Outlier rejection to fit a clean dataset.
* Sample reduction e.g. representing a dataset by its k-means centroids.
* Currently semi-supervised learning is not supported by scoring-based
  functions like ``cross_val_score``, ``GridSearchCV`` or ``validation_curve``
  since the scorers will regard "unlabeled" as a separate class. A resampler
  could add the unlabeled samples to the dataset during fit time to solve this
  (note that this could also be solved by a new cv splitter).
* NaNRejector (drop all samples that contain nan).
* Dataset augmentation (like is commonly done in DL).

Implementation
--------------

API and Constraints
...................

* Resamplers implement a method ``fit_resample(X, y, **kwargs)``, a pure
  function which returns ``Xt, yt, kwargs`` corresponding to the resampled
  dataset, where samples may have been added and/or removed.
* An estimator may only implement either ``fit_transform`` or ``fit_resample``
  if support for ``Resamplers`` in ``Pipeline`` is enabled.
* Resamplers may not change the order, meaning, dtype or format of features
  (this is left to transformers).
* Resamplers should also handled (e.g. resample, generate anew, etc.) any
  kwargs.

Composition
-----------

A key part of the proposal is the introduction of a way of composing resamplers
with predictors. We present two options: ``ResampledTrainer`` and modifications
to ``Pipeline``.

Alternative 1: ResampledTrainer
...............................

This metaestimator composes a resampler and a predictor. It
behaves as follows:

* ``fit(X, y)``: resample ``X, y`` with the resampler, then fit the predictor
  on the resampled dataset.
* ``predict(X)``: simply predict on ``X`` with the predictor.
* ``score(X)``: simply score on ``X`` with the predictor.

See PR #13269 for an implementation.

One benefit of the ``ResampledTrainer`` is that it does not stop the resampler
having other methods, such as ``transform``, as it is clear that the
``ResampledTrainer`` will only call ``fit_resample``.

There are complications around supporting ``fit_transform``, ``fit_predict``
and ``fit_resample`` methods in ``ResampledTrainer``. ``fit_transform`` support
is only possible by implementing ``fit_transform(X, y)`` as ``fit(X,
y).transform(X)``, rather than calling ``fit_transform`` of the predictor.
``fit_predict`` would have to behave similarly.  Thus ``ResampledTrainer``
would not work with non-inductive estimators (TSNE, AgglomerativeClustering,
etc.) as their final step.  If the predictor of a ``ResampledTrainer`` is
itself a resampler, it's unclear how ``ResampledTrainer.fit_resample`` should
behave.  These caveats also apply to the Pipeline modification below.

Example Usage:
~~~~~~~~~~~~~~

.. code-block:: python

    est = ResampledTrainer(RandomUnderSampler(), SVC())
    est = make_pipeline(
        StandardScaler(),
        ResampledTrainer(Birch(), make_pipeline(SelectKBest(), SVC()))
    )
    est = ResampledTrainer(
        RandomUnderSampler(),
        make_pipeline(StandardScaler(), SelectKBest(), SVC()),
    )
    clf = ResampledTrainer(
        NaNRejector(), # removes samples containing NaN
        ResampledTrainer(RandomUnderSampler(),
            make_pipeline(StandardScaler(), SGDClassifier()))
    )

Alternative 2: Prediction Pipeline
..................................

As an alternative to ``ResampledTrainer``, ``Pipeline`` can be modified to
accomodate resamplers.  The essence of the operation is this: one or more steps
of the pipeline may be a resampler. When fitting the Pipeline, ``fit_resample``
will be called on each resampler instead of ``fit_transform``, and the output
of ``fit_resample`` will be used in place of the original ``X``, ``y``, etc.,
to fit the subsequent step (and so on).  When predicting in the Pipeline,
the resampler will act as a passthrough step.

Limitations
~~~~~~~~~~~

.. rubric:: Prohibiting ``transform`` on resamplers

It may be problematic for a resampler to provide ``transform`` if Pipelines
support resampling:

1. It is unclear what to do at test time if a resampler has a transform
   method.
2. Adding fit_resample to the API of an an existing transformer may
   drastically change its behaviour in a Pipeline.

For this reason, it may be best to reject resamplers supporting ``transform``
from being used in a Pipeline.

.. rubric:: Prohibiting ``transform`` on resampling Pipelines

Providing a ``transform`` method on a Pipeline that contains a resampler
presents several problems:

1. A resampling Pipeline needs to use a special code path for ``fit_transform``
   that would call ``fit(X, y, **kw).transform(X)`` on the Pipeline.
   Ordinarily a Pipeline would pass the transformed data to ``fit_transform``
   of the left step. If the Pipeline contains a resampler, it rather needs to
   fit the Pipeline excluding the last step, then transform the original
   training data until the last step, then fit_transform the last step. This
   means special code paths for pipelines containing resamplers; the effect of
   the resampler is not localised in terms of code maintenance.
2. As a result of issue 1, appending a step to the transformation Pipeline
   means that the transformer which was previously last, and previously trained
   on the full dataset, will now be trained on the resampled dataset.
3. As a result of issue 1, the last step cannot be 'passthrough' as in other
   transformer pipelines.

For this reason, it may be best to disable ``fit_transform`` and ``transform``
on the Pipeline. A resampling Pipeline would therefore not be usable as a
transformation within a ``FeatureUnion`` or ``ColumnTransformer``. Thus the
``ResampledTrainer`` would be strictly more expressive than a resampling
Pipeline.

.. rubric:: Handling ``fit`` parameters

Sample props or weights cannot be routed to steps downstream of a resampler in
a Pipeline, unless they too are resampled. It's very unclear how this would
work with Pipeline's current prefix-based fit parameter routing.

TODO: propose solutions

Example Usage:
~~~~~~~~~~~~~~

.. code-block:: python

    est = make_pipeline(RandomUnderSampler(), SVC())
    est = make_pipeline(StandardScaler(), Birch(), SelectKBest(), SVC())
    est = make_pipeline(
        RandomUnderSampler(), StandardScaler(), SelectKBest(), SVC()
    )
    est = make_pipeline(
        NaNRejector(), RandomUnderSampler(), StandardScaler(), SGDClassifer()
    )


Alternative implementation
..........................

Alternatively ``sample_weight`` could be used as a placeholder to
perform resampling. However, the current limitations are:

* ``sample_weight`` is not available for all estimators;
* ``sample_weight`` will implement only simple resampling (only when resampling
  uses original samples);
* ``sample_weight`` needs to be passed and modified within a
  ``Pipeline``, which isn't possible without something like resamplers.

Current implementation
......................

https://github.com/scikit-learn/scikit-learn/pull/13269

Backward compatibility
----------------------

There is no backward incompatibilities with the current API.

Discussion
----------

* https://github.com/scikit-learn/scikit-learn/pull/13269

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
