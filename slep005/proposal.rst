.. _slep_005:

=============
Resampler API
=============

:Author: Oliver Raush (oliverrausch99@gmail.com),
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

* resamplers will reduce or augment the number of samples in ``X`` and
  ``y``;
* ``Pipeline`` should treat them as a separate type of estimator.

Motivation
----------

Sample reduction or augmentation are part of machine-learning
pipeline. The current scikit-learn API does not offer support for such
use cases.

Two possible use cases are currently reported:

* sample rebalancing to correct bias toward class with large cardinality;
* outlier rejection to fit a clean dataset.
   
Implementation
--------------

To handle outlier rejector in ``Pipeline``, we enforce the following:

* an estimator cannot implement both ``fit_resample(X, y)`` and
  ``fit_transform(X)`` / ``transform(X)``. If both are implemented,
  ``Pipeline`` will not be able to know which of the two methods to
  call.
* resamplers are only applied during ``fit``. Otherwise, scoring will
  be harder. Specifically, the pipeline will act as follows:
  
  ===================== ================================
  Method                Resamplers applied               
  ===================== ================================
  ``fit``               Yes
  ``fit_transform``     Yes
  ``fit_resample``      Yes
  ``transform``         No
  ``predict``           No
  ``score``             No
  ``fit_predict``       not supported 
  ===================== ================================

* ``fit_predict(X)`` (i.e., clustering methods) should not be called
  if an outlier rejector is in the pipeline. The output will be of
  different size than ``X`` breaking metric computation.
* in a supervised scheme, resampler will need to validate which type
  of target is passed. Up to our knowledge, supervised are used for
  binary and multiclass classification.
  
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

* Outlier rejection are implemented in:
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
