.. _slep_005:

=============
Resampler API
=============

:Author: Oliver Rausch (oliverrausch99@gmail.com),
         Christos Aridas (char@upatras.gr),
         Guillaume Lemaitre (g.lemaitre58@gmail.com)
:Status: Draft
:Type: Standards Track
:Created: created on, in 2019-03-01
:Resolution: <url>

Abstract
--------

We propose the inclusion of a new type of estimator: resampler. The
resampler will change the samples in ``X`` and ``y``. In short:

* resamplers will reduce and/or augment the number of samples in ``X`` and
  ``y`` during ``fit``, but will perform no changes during ``predict``.
* a new verb/method that all resamplers must implement is introduced: ``fit_resample``.
* A new meta-estimator, ``ResampledTrainer``, that allows for the composition of
  resamplers and estimators is proposed.


Motivation
----------

Sample reduction or augmentation are common parts of machine-learning
pipelines. The current scikit-learn API does not offer support for such
use cases.

Possible Usecases
.................

* sample rebalancing to correct bias toward class with large cardinality
* outlier rejection to fit a clean dataset
* representing a dataset by generating centroids of clustering methods.
* currently semi-supervised learning is not supported by scoring-based
  functions like ``cross_val_score``, ``GridSearchCV`` or ``validation_curve``
  since the scorers will regard "unlabeled" as a separate class. A resampler
  could add the unlabeled samples to the dataset during fit time to solve this
  (note that this can also be solved by a new cv splitter).
* Dataset augmentation (very common in vision problems)

Implementation
--------------
API and Constraints
...................
Resamplers implement a method ``fit_resample(X, y)``, a pure function which
returns ``Xt, yt`` corresponding to the resampled dataset, where samples may
have been added and/or removed.

Resamplers cannot be transformers, that is, a resampler cannot implement
``fit_transform`` or ``transform``. Similarly, transformers cannot implement ``fit_resample``.

Resamplers may not change the order, meaning, dtype or format of features (this is left
to transformers).

Resamplers should also resample any kwargs that are array-like and have the same `shape[0]` as `X` and `y`.

ResampledTrainer
................
This metaestimator composes a resampler and a predictor. It
behaves as follows:

 ``fit(X, y)``: resample ``X, y`` with the resampler, then fit on the resampled
  dataset.
* ``predict(X)``: simply predict on ``X`` with the predictor.
* ``score(X)``: simply score on ``X`` with the predictor.

See PR #13269 for an implementation.

Modifying Pipeline
..................
As an alternative to ``ResampledTrainer``, ``Pipeline`` could be modified to
accomodate resamplers.
The functionality is described in terms of the head (all stages except the last)
and the tail (the last stage) of the ``Pipeline``. Note that we assume
resamplers and transformers are exclusive so that the pipeline can decide which
method to call. Further note that ``Xt, yt`` are the outputs of the stage, and
``X, y`` are the inputs to the stage.

``fit``:
  head for resamplers: `Xt, yt = est.fit_resample(X, y)`
  head for transformers: `Xt, yt = est.fit_transform(X, y)`
  tail for transformers and predictors: `est.fit(X, y)`
  tail for resamplers: `pass`

``fit_transform``:
  Equivalent to `fit(X, y).transform(X)` overall

``predict``
  head for resamplers: `Xt = X`
  head for transformers: `Xt = est.transform(X)`
  tail for predictors: `return est.predict(X)`
  tail for transformers and resamplers: `error`

``transform``
  head for resamplers: `Xt = X`
  head for transformers: `Xt = est.transform(X)`
  tail for predictors and resamplers: `error`
  tail for transformers: `return est.transform(X)`

``score``
  see predict


Alternative implementation
..........................

Alternatively ``sample_weight`` could be used as a placeholder to
perform resampling. However, the current limitations are:

* ``sample_weight`` is not available for all estimators;
* ``sample_weight`` will implement only sample reductions;
* ``sample_weight`` can be applied at both fit and predict time;
* ``sample_weight`` need to be passed and modified within a
  ``Pipeline``.

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
