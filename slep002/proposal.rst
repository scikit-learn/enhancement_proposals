.. _slep_002:

==========================
SLEP002: Dynamic pipelines
==========================

.. topic:: **Summary**

    Create and manipulate pipelines with ease.

.. contents:: Table of contents
   :depth: 3

Goals
=====

* Being backward-compatible
* Allow interactive pipeline construction (for example in IPython)
* Support adding and replacing parts of pipeline
* Support using steps as label (y's) transformers


Design
======

Imports
-------

In addition to `Pipeline <sklearn.pipeline.Pipeline>` class some additional
wrappers are proposed as part of public API::

    from sklearn.pipeline import (Pipeline, fitted, transformer, predictor
                                  label_transformer, label_predictor,
                                  ignore_transform, ignore_predict)

Pipeline creation
-----------------

Backward-compatible
...................

Of course, old syntax should be supported::

    pipe = Pipeline(steps=[('name1', estimator1), ('name2', 'estimator2)]

Proposed default constructor
............................

It is not backward-compatible, but it shouldn't break most of old code::

    pipe = Pipeline()

It is not yet configured, so trying to use it should fail::

    >>> pipe.predict(...)
    Traceback (most recent call last):
    ...
    NotFittedError: This Pipeline instance is not fitted yet

    >>> pipe.fit(...)
    Traceback (most recent call last):
    ...
    NotConfiguredError: This Pipeline instance is not configured yet

Proposed construction from iterable of dicts
............................................

Dictionaries emphasize structure::

    pipe = Pipeline(
        steps=[
            {'name1': Estimator1()},
            {'name2': Estimator2()},
        ]
    )

Every dict should be of length 1::

    >>> pipe = Pipeline(
    ...     steps=(
    ...         {'name1': Estimator1(),
    ...          'name2': Estimator2()},
    ...         {},
    ...     ),
    ... )
    Traceback (most recent call last):
    ...
    TypeError: Wrong step definition


Proposed construction from ``collections.OrderedDict``
......................................................

It is probably the most natural way to create a pipeline::

    pipe = Pipeline(
        collections.OrderedDict([
            ('name1', Estimator1()),
            ('name2', Estimator2()),
        ]),
    )

Backward-compatibility notice
-----------------------------

As user can provide object of any type as ``steps`` argument to constructor,
there is no way to be 100% compatible, if we are going to maintain our oun
type for ``Pipeline.steps``.
But in most cases people provide ``list`` object as ``steps`` parameter, so
being backward-compatible with ``list`` API should be fine.

Adding estimators
-----------------

Backward-compatible
...................

Although not documented, but popular method of modifying (not fitted) pipelines
should be supported::

    pipe.steps.append(['name', estimator])

The only difference is that special handler is returned instead of ``None``.

Enhanced: by indexing
.....................

Using dict-like syntax if very user-friendly::

    pipe.steps['name'] = estimator

Enhanced: ``add`` function
..........................

Alias to previous two calls::

    pipe.steps.add('name', estimator)

And also::

    pipe.add_estimator('name', estimator)

Adding estimators with type specification
.........................................

Estimator types will be discussed later, but some functions belong to this
section::

    pipe.add_estimator('name0', estimator0).mark_fitted()
    pipe.add_transformer('name1', estimator1)  # never calls .fit (x, y -> x)
    pipe.add_predictor('name2', estimator2)  # never calls .trasform (x -> y)
    pipe.add_label_transformer('name3', estimator3)  # (y -> y)
    pipe.add_label_predictor('name4', estimator4)  # (y -> y)

Steps (subestimators) access
----------------------------

Backward-compatible
...................

Indexing by number should return ``(step, estimator)`` pair::

    >>> pipe.steps[0]
    ('name', SomeEstimator(...))

Enhanced access via indexing
............................

One should be able to retrieve any estimator with indexing by step's name::

    >>> pipe.steps['mame']
    SomeEstimator(param1=value1, param2=value2)

Enhanced access via attributes
..............................

Dotted access should also work if name of step is valid python name literal
and there is no inference with internal methods::

    >>> pipe.steps.name
    SomeEstimator(param1=value1, param2=value2)

    >>> pipe.steps.get
    <bound method index of <StepsOrderedDict object at ...>>

    >>> pipe.add_transformer('my transformer', estimator)
    >>> pipe.steps.my transformer
    File ...
    pipe.steps_.my transformer
                   ^
    SyntaxError: invalid syntax

Replacing estimators
--------------------

Backward-compatible
...................

Replacing should only be supported via access to ``.steps`` attribute. This way
there is no ambiguity with new/old subestimator subtype::

    pipe = Pipeline(steps=[('name', SomeEstimator())])
    pipe.steps[0] = ('name', AnotherEstimator())

Replace via indexing by step name
.................................

Dict-like behavior can be used too::

    pipe = Pipeline(steps=[('name', SomeEstimator())])
    pipe.steps['name'] = AnotherEstimator()

Replace via ``replace()`` function
..................................

This way one can obtain special handler::

    pipe.steps.replace('old_step_name', 'new_step_name', NewEstimator())
    pipe.steps.replace('step_name', 'new_name',
                       SomeEstimator()).mark_transformer()


Rename step via ``rename()`` function
.....................................

Simple way to change step's name (doesn't affect anything except object
representation)::

    pipe.steps.rename('old_name', 'new_name')

Modifying estimators
--------------------

Changing estimator params should only be performed via
``pipeline.set_params()``.  If somebody calls ``subestimator.set_params()``
directly, pipeline object will have no idea about changed state. There is no
easy way to control it, so docs should just warm users about it.

On the other hand, there exist not-so-easy way to at least warm users during
runtime: pipeline will have to keep params of all its children and compare them
with actual params during ``fit`` or ``predict`` routines and raise a warning
if they do not match.  This functionality may be implemented as part of some
kind of debugging mode.

Deleting estimators
-------------------

Backward-compatible
...................

Backward-compatible way to delete a step is to ``del`` it via index number::

    del pipe.steps[2]

Enhanced indexing
.................

A little more user-friendly way to remove a step can be achieved
using enhanced indexing::

    pipe = Pipeline()
    est1 = Estimator1()
    est2 = Estimator2()

    pipe.steps.add('name1', est1)
    pipe.steps.add('name2', est2)

    del pipe.steps['name1']
    del pipe.steps[pipe.steps.index(est2)]

Using dict/list-like ``pop()`` functions
........................................

Last estimator in a chain can be deleted with any of these calls::

    >>> pipe.steps.pop()
    SomeEstimator()

    >>> pipe.steps.popitem()
    ('some_name', SomeEstimator())

Likewise, first estimator in the pipeline can be removed with any of these
calls::

    >>> pipe.steps.popfront()
    BeginEstimator()

    >>> pipe.steps.popitemfront()
    ('begin', BeginEstimator)

Any step can be removed with ``pop(step_name)`` or ``popitem(step_name)``.

Fitted flag reset
-----------------

Internally ``Pipeline`` object should keep track on whatever it is fitted or
not.  It should consider itself fitted if it wasn't modified after:

* successful call to ``.fit``::

    pipe.fit(...)  # Got fitted pipeline if no exception was raised

* construction with list of estimators, all marked as
  fitted via ``fitted`` function::

    pipe = pipeline.Pipeline(steps=[
        ('name1', fitted(estimator1)),
        ('name2', fitted(estimator2)(,
        ...
    ])

* adding fitted estimator to fitted pipeline::

    pipe.steps.append(fitted(estimator1))
    pipe.steps['new_step'] = fitted(estimator2)
    pipe.add_transformer('some_key', estimator3).set_fitted()

* renaming step in fitted pipeline
* removing first or last step from fitted pipeline

Subestimator types
------------------

Subestimator type contains information about the way a pipeline
should process a step with that subestimator.

Subestimator type can be specified:

* By wrapping estimator with subtype constructor call:
    * when creating pipeline::

        Pipeline([
            ('name1', transformer(estimator)),
            ('name2', predictor(estimator)),
            ('name3', label_transformer(estimator)),
            ('name4', label_predictor(estimator)),
        ])
    * when adding or replacing a step::

        pipe.steps.append(['name', label_predictor(estimator])
        pipe.steps.add('name', label_transformer(estimator))
        pipe.add_estimator('name', predictor(estimator))
        pipe.steps.replace('name', transformer(fitted(estimator)))
        pipe.steps['name'] = fitted(predictor(estimator))
* Using ``pipe.add_*`` methods::

    pipe.add_transformer('transformer', Transformer())
    pipe.add_predictor('predictor', Predictor())
    pipe.add_label_transformer('l_transformer', LabelTransformer())
    pipe.add_label_predictor('l_predictor', LabelPredictor())
* Using special handler methods::

    pipe.add_estimator('name1', EstimatorA()).mark_transformer()
    pipe.steps.add('name2', EstimatorB()).mark_predictor()
    pipe.steps.append(['name3', EstimatorC()]).mark_label_transformer()
    pipe.steps.replace('name4', EstimatorD()).mark_label_predictor()
    pipe.steps.replace('name4', EstimatorE()).mark('label_transformer')

Transformer
...........
Is a default type.

It is processed like this::

    y_new = y
    if fiting:
        X_new = step_estimator.fit_transform(X, y)
    else:
        X_new = step.transform(X, y)

Predictor
.........

It is processed like this::

    X_new = X
    if fitting:
        y_new = step_estimator.fit_predict(X, y)
    else:
        y_new = step_estimator.predict(X, y)

Label transformer
.................

Processing pseudocode::

    X_new = X
    if fitting:
        y_new = step_estimator.fit_transform(y)
    else:
        y_new = step_estimator.transform(y)

Label predictor
...............

Processing pseudocode::

    X_new = X
    if fitting:
        y_new = step_estimator.fit_predict(y)
    else:
        y_new = step_estimator.predict(y)

Special handlers and wrapper functions
--------------------------------------

Assuming estimator is already fitted
....................................

to add estimator, that was already fitted to a pipline
one can use fitted function::

    est = SomeEstimator().fit(some_data)
    pipe.steps.add('prefitted', fitted(est))

or special hanlder method::

    pipe.steps.add('prefitted', est).mark_fitted()
    # or
    pipe.steps.add('prefitted', est).mark('fitted')

Ignoring estimator during prediction
....................................

In some cases we only need to apply estimator only during fit-phase::

    pipe.add_estimator('sampler', ignore_transform(Sampler()))
    # or
    pipe.add_estimator('sampler', Sampler()).mark_ignore_transform()
    # or
    pipe.add_estimator('sampler', Sampler()).mark('ignore_transform')

If it is ``predictor`` or ``label_predictor``, then one should use
``ignore_predict``::

    pipe.add_estimator('cluster', ignore_predict(predictor(ClusteringEstimator())))
    # or
    pipe.add_estimator('cluster', predictor(ClusteringEstimator())).mark_ignore_predict()
    # or
    pipe.add_estimator('cluster', predictor(ClusteringEstimator())).mark('ignore_predict')

Setting subestimator type
.........................

As specified above setting subestimator type can be performed with special
handler or special function call.

Combining multiple flags
........................

All sorts of syntax combinations should be supported::

    pipe.steps.add('step', fitted(predictor(Estimator())))
    pipe.steps.add('step', predictor(fitted(Estimator())))
    pipe.steps.add('step', predictor(Estimator())).mark_fitted()
    pipe.steps.add('step', fitted(Estimator())).mark_predictor()
    pipe.steps.add('step', Estimator()).mark_predictor().mark_fitted()
    pipe.steps.add('step', Estimator()).mark_fitted().mark_predictor()
    pipe.steps.add('step', Estimator()).mark('fitted').mark_predictor()
    pipe.steps.add('step', Estimator()).mark('predictor').mark_fitted()
    pipe.steps.add('step', Estimator()).mark('predictor').mark('fitted')
    pipe.steps.add('step', Estimator()).mark('fitted').mark('predictor')
    pipe.steps.add('step', Estimator()).mark('fitted', 'predictor')
    pipe.steps.add('step', Estimator()).mark('predictor', 'fitted')

Type of steps object
--------------------

This is internal type, users shouldn'r usualy mess with that.
But public methods should be considered as part of pipeline API.

Attributes and methods with standard behavior
..............................................

Special methods:

* ``__contains__()``, ``__getitem__()``, ``__setitem__()``, ``__delitem__()``
* ``__len__()``, ``__iter__()``
* ``__add__()``, ``__iadd__()``

Methods:

* ``get()``, ``index()``
* ``extend()``, ``insert()``
* ``keys()``, ``items()``, ``values()``
* ``clear()``, ``pop()``, ``popitem()``, ``popfront()``, ``popitemfront()``

Non-standard methods
....................

* ``replace()``
* ``rename()``

Not supported arguments and methods
...................................

This type provides dict-like and list-like interfaces,
but following methods and attributes are not supported:

* ``fromkeys()``
* ``setdefault()``
* ``sort()``
* ``__mul__()``, ``__rmul__()``, ``__imul__()``

Any attempt to use them should fail with ``AttributeError`` or
``NotImplementedError``

Thease methods may be not supported:

* ``__ge__()``, ``__gt__()``
* ``__le__()``, ``__lt__()``

Serialization
-------------

* Support loading/unpickling pipelines from old scikit-learn versions
* Keep track of API version in ``__getstate__`` / ``picklier``: all future
  versions should support unpickling all previous versions of enhanced pipeline
* Serialization of ``.steps`` attribute (without master pipeline) may be not
  supported.

Examples
========

Example: remove outliers
------------------------

Proposed design allows to do many things, but some of them have to be done in
two steps.  But it shouldn't be a problem, as one can make a pipeline with
those steps::

    def make_outlier_remover(bad_value=-1):
        outlier_remover = Pipeline()
        outlier_remover.steps.add(
            'data',
            DropLinesOfXCorrespondingLabel(remove_if=bad_value),
        )
        outlier_remover.steps.add(
            'labels',
            DropLabelsIf(remove_if=bad_value),
        ).mark_label_transformer()
        return outlier_remover

Example: sample dataset
-----------------------
We can use previous example function for this::

    def make_sampler(percent=75):
        sentinel = object()
        sampler = Pipeline()
        sampler.steps.add(
            'sample',
            LabelSomeRowsAs(percent=percent, label=sentinel),
        ).mark('predictor', 'ignore_predict')
        sampler.steps.add(
            'down',
            make_outlier_remover(bad_value=sentinel),
        )
        return sampler

Benefits
========
* Users can use old code with new pipeline:
  usual ``__init__``, ``set_params``, ``get_params``, ``fit``, ``transform``
  and ``predict`` are the only requirements of subestimators.
* Users can use new pipeline with their old code:
  pipeline is stil usual estimator, that supports usual set of methods.
* We finally can transform ``y`` in a pipeline.

Drawbacks
=========
Well, it's a lot of code to write and support...
