 .. _slep_007:

=============
Feature Names
=============

``scikit-learn`` has been making it easier to build complex workflows with the
``ColumnTransformer`` and it has been seeing widespread adoption. However,
using it results in pipelines where it's not clear what the input features to
the final predictor are, even more so than before. The following example
illustrates that very well::


    """
    ===================================
    Column Transformer with Mixed Types
    ===================================

    This example illustrates how to apply different preprocessing and
    feature extraction pipelines to different subsets of features,
    using :class:`sklearn.compose.ColumnTransformer`.
    This is particularly handy for the case of datasets that contain
    heterogeneous data types, since we may want to scale the
    numeric features and one-hot encode the categorical ones.

    In this example, the numeric data is standard-scaled after
    mean-imputation, while the categorical data is one-hot
    encoded after imputing missing values with a new category
    (``'missing'``).

    Finally, the preprocessing pipeline is integrated in a
    full prediction pipeline using :class:`sklearn.pipeline.Pipeline`,
    together with a simple classification model.
    """

    # Author: Pedro Morales <part.morales@gmail.com>
    #
    # License: BSD 3 clause

    import numpy as np

    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import fetch_openml
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, GridSearchCV

    np.random.seed(0)

    # Load data from https://www.openml.org/d/40945
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    # Alternatively X and y can be obtained directly from the frame attribute:
    # X = titanic.frame.drop('survived', axis=1)
    # y = titanic.frame['survived']

    # We will train our classifier with the following features:
    # Numeric Features:
    # - age: float.
    # - fare: float.
    # Categorical Features:
    # - embarked: categories encoded as strings {'C', 'S', 'Q'}.
    # - sex: categories encoded as strings {'female', 'male'}.
    # - pclass: ordinal integers {1, 2, 3}.

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_features = ['age', 'fare']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['embarked', 'sex', 'pclass']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression())])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf.fit(X_train, y_train)
    print("model score: %.3f" % clf.score(X_test, y_test))


    ###############################################################################
    # Using the prediction pipeline in a grid search
    ###############################################################################
    # Grid search can also be performed on the different preprocessing steps
    # defined in the ``ColumnTransformer`` object, together with the classifier's
    # hyperparameters as part of the ``Pipeline``.
    # We will search for both the imputer strategy of the numeric preprocessing
    # and the regularization parameter of the logistic regression using
    # :class:`sklearn.model_selection.GridSearchCV`.


    param_grid = {
        'preprocessor__num__imputer__strategy': ['mean', 'median'],
        'classifier__C': [0.1, 1.0, 10, 100],
    }

    grid_search = GridSearchCV(clf, param_grid, cv=10)
    grid_search.fit(X_train, y_train)

    print(("best logistic regression from grid search: %.3f"
           % grid_search.score(X_test, y_test)))


However, it's impossible to interpret or even sanity-check the
``LogisticRegression`` instance that's produced in the example, because the
correspondence of the coefficients to the input features is basically
impossible to figure out.

This proposal suggests adding two attributes to fitted estimators:
``feature_names_in_`` and ``feature_names_out_``, such that in the
abovementioned example ``clf[-1].feature_names_in_`` and
``clf[-2].feature_names_out_`` will be::

    ['num__age',
     'num__fare',
     'cat__embarked_C',
     'cat__embarked_Q',
     'cat__embarked_S',
     'cat__embarked_missing',
     'cat__sex_female',
     'cat__sex_male',
     'cat__pclass_1',
     'cat__pclass_2',
     'cat__pclass_3']

Ideally the generated feature names describe how a feature is generated at each
stage of a pipeline. For instance, ``cat__sex_female`` shows that the feature
has been through a categorical preprocessing pipeline, was originally the
column ``sex``, and has been one hot encoded and is one if it was originally
``female``. However, this is not always possible or desirable especially when a
generated column is based on many columns, since the generated feature names
will be too long, for example in ``PCA``. As a rule of thumb, the following
types of transformers may generate feature names which corresponds to the
original features:

- Leave columns unchanged, *e.g.* ``StandardScaler``
- Select a subset of columns, *e.g.* ``SelectKBest``
- create new columns where each column depends on at most one input column,
  *e.g* ``OneHotEncoder``
- Algorithms that create combinations of a fixed number of features, *e.g.*
  ``PolynomialFeatures``, as opposed to all of
  them where there are many. Note that verbosity considerations and
  ``prefix_feature_names`` as explained later can apply here.

This proposal talks about how feature names are generated and not how they are
propagated.

Scope
-----

All estimators implement a ``feature_names_in_`` API, and any estimator with a
``transform`` method implements a ``feature_names_out_`` API, _i.e._ they
expose the generated feature names via the ``feature_names_out_`` attribute.

Input Feature Names
-------------------

The input feature names are stored in a fitted estimator in a
``feature_names_in_`` attribute, and are taken from the given input data, for
instance a ``pandas`` data frame. This attribute will be ``None`` if the input
provides no feature names.

Output Feature Names
--------------------

A fitted estimator exposes the output feature names through the
``feature_names_out_`` attribute. Here we discuss more in detail how these
feature names are generated. Since for most estimators there are multiple ways
to generate feature names, this SLEP does not intend to define how exactly
feature names are generated for all of them. It is instead a guideline on how
they could generally be generated.

One-to-one Transformers
***********************

From the perspective of feature names, the simplest transformers are the ones
which have the same feature names in the output as in the input, and the
transformation done on the data is semi-trivial. The ``StandardScaler`` can be
one example.

- Input provides feature names: ``feature_names_in_`` and
  ``feature_names_out_`` are the same.
- Input provides no feature names: ``feature_names_out_`` will be ``x0`` to
  ``xn``, where ``n`` is the number of features.

However, a transformer can choose to be more verbose and generate a more
informative feature name, ``scaled(income)`` could be an example, and the
verbosity is controlled by a parameter to the estimator.

Feature Selector Transformers
*****************************

The output feature names are the ones selected from the input, and if no
feature names are provided, ``x0`` to ``xn`` are assumed to be their names. For
example, if a ``SelectKBest`` transformer selects the first and the third
features, and no names are provided, the ``feature_names_out_`` will be ``[x0,
x2]``.

Feature Generating Transformers
*******************************

The simplest category of transformers in this section are the ones which
generate a column based on a single given column. The generated output column
in this case is a sensible transformation of the input feature name. For
instance, a ``LogTransformer`` can do ``'age' -> 'log(age)'``, and a
``OneHotEncoder`` could do ``'gender' -> 'gender_female', 'gender_fluid',
...``.

Transformers where each output feature depends on a fixed number of input
features may generate descriptive names as well. For instance, a
``PolynomialTransformer`` on a small subset of features can generate an output
feature name such as ``x[0] * x[2] ** 3``.

And finally, the transformers where each output feature depends on many or all
input features, generate feature names which has the form of ``name0`` to
``namen``, where ``name`` represents the transformer. For instance, a ``PCA``
transformer will output ``[pca0, ..., pcan]``, ``n`` being the number of PCA
components.

Meta-Estimators
***************

Meta estimators can choose to prefix the output feature names given by the
estimators they are wrapping or not.

By default, ``Pipeline`` adds no prefix, *i.e* its ``feature_names_out_`` is
the same as the ``feature_names_out_`` of the last step, and ``None`` if the
last step is not a transformer.

``ColumnTransformer`` by default adds a prefix to the output feature names,
indicating the name of the transformer applied to them. If a column is in the output
as a part of ``passthrough``, it won't be prefixed since no operation has been
applied on it.

This is the default behavior, and it can be tuned by constructor parameters if
the meta estimator allows it. For instance, a ``prefix_feature_names=False``
may indicate that a ``ColumnTransformer`` should not prefix the generated
feature names with the name of the step.

Examples
--------

Here we include some examples to demonstrate the behavior of output feature
names::

    100 features (no names) -> PCA(n_components=3)
    feature_names_out_: [pca0, pca1, pca2]


    100 features (no names) -> SelectKBest(k=3)
    feature_names_out_: [x2, x17, x42]


    [f1, ..., f100] -> SelectKBest(k=3)
    feature_names_out_: [f2, f17, f42]


    [cat0] -> OneHotEncoder()
    feature_names_out_: [cat0_cat, cat0_dog, ...]


    [f1, ..., f100] -> Pipeline(
                           [SelectKBest(k=30),
                            PCA(n_components=3)]
                       )
    feature_names_out_: [pca0, pca1, pca2]


    [model, make, num0, ..., num100] ->
        ColumnTransformer(
            [('cat', Pipeline(SimpleImputer(), OneHotEncoder()),
              ['model', 'make']),
             ('num', Pipeline(SimpleImputer(), PCA(n_components=3)),
              ['num0', ..., 'num100'])]
        )
    feature_names_out_: ['cat_model_100', 'cat_model_200', ...,
                         'cat_make_ABC', 'cat_make_XYZ', ...,
                         'num_pca0', 'num_pca1', 'num_pca2']

However, the following examples produce a somewhat redundant feature names,
and hence the relevance of ``prefix_feature_names=False``::

    [model, make, num0, ..., num100] ->
        ColumnTransformer([
            ('ohe', OneHotEncoder(), ['model', 'make']),
            ('pca', PCA(n_components=3), ['num0', ..., 'num100'])
        ])
    feature_names_out_: ['ohe_model_100', 'ohe_model_200', ...,
                         'ohe_make_ABC', 'ohe_make_XYZ', ...,
                         'pca_pca0', 'pca_pca1', 'pca_pca2']

If desired, the user can remove the prefixes::

    [model, make, num0, ..., num100] ->
        make_column_transformer(
            (OneHotEncoder(), ['model', 'make']),
            (PCA(n_components=3), ['num0', ..., 'num100']),
            prefix_feature_names=False
        )
    feature_names_out_: ['model_100', 'model_200', ...,
                         'make_ABC', 'make_XYZ', ...,
                         'pca0', 'pca1', 'pca2']

Backward Compatibility
----------------------

All estimators should implement the ``feature_names_in_`` and
``feature_names_out_`` API. This is checked in ``check_estimator``, and the
transition is done with a ``FutureWarning`` to give time to third party
developers to implement the API.

Notes
-----

This SLEP also applies to `resamplers
<https://github.com/scikit-learn/enhancement_proposals/pull/15>`_ the same way
as transformers.
