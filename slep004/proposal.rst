.. _slep_004:

=========================
SLEP004: Data information
=========================

This is a specification to introduce data information (as
``sample_weights``) during the computation of an estimator methods
(``fit``, ``score``, ...) based on the different discussion proposes on
issues and PR :

-  `Consistent API for attaching properties to samples
   #4497 <https://github.com/scikit-learn/scikit-learn/issues/4497>`__
-  `Acceptance of sample\_weights in pipeline.score
   #7723 <https://github.com/scikit-learn/scikit-learn/pull/7723>`__
-  `Establish global error state like np.seterr
   #4660 <https://github.com/scikit-learn/scikit-learn/issues/4660>`__
-  `Should cross-validation scoring take sample-weights into account?
   #4632 <https://github.com/scikit-learn/scikit-learn/issues/4632>`__
-  `Sample properties
   #4696 <https://github.com/scikit-learn/scikit-learn/issues/4696>`__

Probably related PR: - `Add feature\_extraction.ColumnTransformer
#3886 <https://github.com/scikit-learn/scikit-learn/pull/3886>`__ -
`Categorical split for decision tree
#3346 <https://github.com/scikit-learn/scikit-learn/pull/3346>`__

Google doc of the sample\_prop discussion done during the sklearn day in
paris the 7th June 2017:
https://docs.google.com/document/d/1k8d4vyw87gWODiyAyQTz91Z1KOnYr6runx-N074qIBY/edit

.. contents:: Table of contents
   :depth: 2

1. Requirement
==============

These requirements are defined from the different issues and PR
discussions:

-  User can attach information to samples.
-  Must be a DataFrame like object.
-  Can be given to ``fit``, ``score``, ``split`` and every time user
   give X.
-  Must work with every meta-estimator
   (``Pipeline, GridSearchCV, cross_val_score``).
-  Can specify what sample property is used by each part of the
   meta-estimator.
-  Must raise an error if not necessary extra information are given to
   an estimator. In the case of meta-estimator these errors are not
   raised.

Requirement proposed but not used by this specification: - User can
attach feature properties to samples.

2. Definition
=============

Some estimator in sklearn can change their behavior when an attribute
``sample_props`` is provided. ``sample_props`` is a dictionary
(``pandas.DataFrame`` compatible) defining sample properties. The
example bellow explain how a ``sample_props`` can be provided to
LogisticRegression to weighted the samples:

.. code:: python

    import numpy as np
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    # Define weights used by sample_props
    weights_fit = np.random.rand(X.shape[0])
    weights_fit /= np.sum(weights_fit)
    weights_score = np.random.rand(X.shape[0])
    weights_score /= np.sum(weights_score)

    logreg = LogisticRegression()

    # Fit and score a LogisticRegression without sample weights
    logreg = logreg.fit(X, y)
    score = logreg.score(X, y)
    print("Score obtained without applying weights: %f" % score)

    # Fit LogisticRegression without sample weights and score with sample weights
    logreg = logreg.fit(X, y)
    score = logreg.score(X, y, sample_props={'weight': weights_score})
    print("Score obtained by applying weights only to score: %f" % score)

    # Fit and score a LogisticRegression with sample weights
    log_reg = logreg.fit(X, y, sample_props={'weight': weights_fit})
    score = logreg.score(X, y, sample_props={'weight': weights_score})
    print("Score obtained by applying weights to both"
          " score and fit: %f" % score)

When an estimator expects a mandatory ``sample_props``, an error is
raised for each property not provided. Moreover if an unintended
properties is given through ``sample_props``, a warning will be
launched to prevent that the result may be different from the one
expected. For example, the following code :

.. code:: python

    import numpy as np
    from sklearn import datasets
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    weights = np.random.rand(X.shape[0])

    logreg = LogisticRegression()

    # This instruction will raise the warning
    logreg = logreg.fit(X, y, sample_props={'bad_property': weights})

will **raise the warning message**: "sample\_props['bad\_property'] is
not used by ``LogisticRegression.fit``. The results obtained may be
different from the one expected."

We provide the function ``sklearn.seterr`` in the case you want to
change the behavior of theses messages. Even if there are considered as
warnings by default, we recommend to change the behavior to raise as
errors. You can do it by adding the following code:

.. code:: python

    sklearn.seterr(sample_props="raise")

Please refer to the documentation of ``np.seterr`` for more information.

3. Behavior of ``sample_props`` for meta-estimator
==================================================

3.1 Common routing scheme
-------------------------

Meta-estimators can also change their behavior when an attribute
``sample_props`` is provided. On that case, ``sample_props`` will be
sent to any internal estimator and function supporting the
``sample_props`` attribute. In other terms all the property defined by
``sample_props`` will be transmitted to each internal functions or
classes supporting ``sample_props``. For example in the following
example, the property ``weights`` is sent through ``sample_props`` to
``pca.fit_transform`` and ``logistic.fit``:

.. code:: python

    import numpy as np
    from sklearn import decomposition, datasets, linear_model
    from sklearn.pipeline import Pipeline

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    logistic = linear_model.LogisticRegression()
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic),])

    # Define weights
    weights = np.random.rand(X.shape[0])
    weights /= np.sum(weights)

    # weights is send to pca.fit_transform and logistic.fit
    pipe.fit(X, sample_props={"weights": weights})

By contrast with the estimator, no warning will be raised by a
meta-estimator if an extra property is sent through ``sample_props``.
Anyway, errors are still raised if a mandatory property is not provided.

3.2 Override common routing scheme
----------------------------------

You can override the common routing scheme of ``sample_props`` of
nested objects by defining sample properties of the form
``<component>__<property>``.

You can override the common routing scheme of ``sample_props`` by
defining your own routes through the ``routing`` attribute of a
meta-estimator.

A route defines a way to override the value of a key of
``sample_props`` by the value of another key in the same
``sample_props``. This modification is done every time a method
compatible with ``sample_prop`` is called.

To illustrate how it works, if you want to send ``weights`` only to
``pca``, you can define a ``sample_prop`` with a property
``pca__weights``:

.. code:: python

    import numpy as np
    from sklearn import decomposition, datasets, linear_model
    from sklearn.pipeline import Pipeline

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    logistic = linear_model.LogisticRegression()
    pca = decomposition.PCA()

    # Create a route using routing
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic),])

    # Define weights
    weights = np.random.rand(X.shape[0])
    weights /= np.sum(pca_weights)
    pca_weights = np.random.rand(X.shape[0])
    pca_weights /= np.sum(pca_weights)

    # Only pca will receive pca_weights as weights
    pipe.fit(X, sample_props={'pca__weights': pca_weights})

    # pca will receive pca_weights and logistic will receive weights as weights
    pipe.fit(X, sample_props={'pca__weights': pca_weights,
                              'weights': weights})

By defining ``pca__weights``, we have overridden the property
``weights`` for ``pca``. On all cases, the property ``pca__weights``
will be send to ``pca`` and ``logistic``.

Overriding the routing scheme can be subtle and you must remember the
priority of application of each route types:

1. Routes applied specifically to a function/estimator:
   ``{'pca__weights': weights}}``
2. Routes defined globally: ``{'weights': weights}``

Let's consider the following code to familiarized yourself with the
different routes definitions :

.. code:: python

    import numpy as np
    from sklearn import datasets
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import cross_val_score, GridSearchCV, LeaveOneLabelOut

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    # Define the groups used by cross_val_score
    cv_groups = np.random.randint(3, size=y.shape)

    # Define the groups used by GridSearchCV
    gs_groups = np.random.randint(3, size=y.shape)

    # Define weights used by cross_val_score
    weights = np.random.rand(X.shape[0])
    weights /= np.sum(weights)

    # We define the GridSearchCV used by cross_val_score
    grid = GridSearchCV(SGDClassifier(), params, cv=LeaveOneLabelOut())

    # When cross_val_score is called, we send all parameters for internal values
    cross_val_score(grid, X, y, cv=LeaveOneLabelOut(),
                    sample_props={'cv__groups': groups,
                                  'split__groups': gs_groups,
                                  'weights': weights})

With this code, the ``sample_props`` sent to each function of
``GridSearchCV`` and ``cross_val_score`` will be:

+-------------+--------------------------------------------------------------+
| function    | ``sample_props``                                             |
+=============+==============================================================+
| grid.fit    | ``{'weights': weights, 'cv__groups': cv_groups, split_groups |
|             | ': gs_groups}``                                              |
+-------------+--------------------------------------------------------------+
| grid.score  | ``{'weights': weights, 'cv__groups': cv_groups, split_groups |
|             | ': gs_groups}``                                              |
+-------------+--------------------------------------------------------------+
| grid.split  | ``{'weights': weights, 'groups': gs_groups, 'cv__groups': cv |
|             | _groups, split_groups': gs_groups}``                         |
+-------------+--------------------------------------------------------------+
| cross\_val\ | ``{'weights': weights, 'groups': groups, 'cv__groups': cv_gr |
| _score      | oups, split_groups': gs_groups}``                            |
+-------------+--------------------------------------------------------------+

Thus, these functions receive as ``weights`` and ``groups`` properties :

+---------------------+---------------+-----------------+
| function            | ``weights``   | ``groups``      |
+=====================+===============+=================+
| grid.fit            | ``weights``   | ``None``        |
+---------------------+---------------+-----------------+
| grid.score          | ``weights``   | ``None``        |
+---------------------+---------------+-----------------+
| grid.split          | ``weights``   | ``gs_groups``   |
+---------------------+---------------+-----------------+
| cross\_val\_score   | ``weights``   | ``cv_groups``   |
+---------------------+---------------+-----------------+

4. Alternative propositions for sample\_props (06.17.17)
========================================================

The meta-estimator says which columns of sample\_props they wanted to
use.

.. code:: python

    p = make_pipeline(
         PCA(n_components=10),
         SVC(C=10).with(<method>_<thing_the_method_knows>=<column_name>)
    )
    p.fit(X, y, sample_props={column_name=value})

For example :

.. code:: python

    p = make_pipeline(
        PCA(n_components=10),
        SVC(C=10).with(fit_weights='weights', score_weights='weights')
    )
    p.fit(X, y, sample_props={"weights": w})

**Other proposals**: - Olivier suggests to modify ``.with(...)`` by
``.sample_props_mapping(...)``. - Gael suggests to change the
``.with(...)`` by a property ``with_props=...`` like :

.. code:: python

    p = make_pipeline(
        PCA(n_components=10),
        SVC(C=10),
        with_props={
            'svc':(<method>_<thing_the_method_knows>=<column_name>)}
    )

4.1 GridSearch + Pipeline case
------------------------------

Let's consider the case of a ``GridSearch`` working with a ``Pipeline``.
How we definer the ``sample_props`` on that case ?

Alternative 1
~~~~~~~~~~~~~

Pass through everything in ``GridSearchCV``:

.. code:: python

    pipe = make_pipeline(
        PCA(), SVC(),
        with_props={pca__fit_weight: 'my_weights'}})
    GridSearchCV(
        pipe, cv=my_cv,
        with_props={'cv__groups': "my_groups", '*':'*')

A more complex example with this solution:

.. code:: python

    pipe = make_pipeline(
        make_union(
            CountVectorizer(analyzer='word').with(fit_weight='my_weight'),
            CountVectorizer(analyzer='char').with(fit_weight='my_weight')),
        SVC())

    GridSearchCV(
        pipe,
        cv=my_cv.with(groups='my_groups'), score_weight='my_weight')

Alternative 2
~~~~~~~~~~~~~

Grid search manage the ``sample_props`` of all internal variable.

.. code:: python

    pipe = make_pipeline(PCA(), SVC())
    GridSearchCV(
        pipe, cv=my_cv,
        with_props={
            'cv__groups': "my_groups",
            'estimator__pca__fit_weight': "my_weights"),
            })

A more complex example with this solution:

.. code:: python

    pipe = make_pipeline(
        make_union(
            CountVectorizer(analyzer='word'),
            CountVectorizer(analyzer='char')),
        SVC())
    GridSearchCV(
        pipe, cv=my_cv,
        with_props={
            'cv__groups': "my_groups",
            'estimator__featureunion__countvectorizer-1__fit_weight': "my_weights",
            'estimator__featureunion__countvectorizer-2__fit_weight': "my_weights",
            'score_weight': "my_weights",
        }
    )
