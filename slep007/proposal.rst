 .. _slep_007:

====================================================
SLEP007: Feature names, their generation and the API
====================================================

:Author: Adrin Jalali
:Status: Under Review
:Type: Standards Track
:Created: 2019-04

Abstract
########

This SLEP proposes the introduction of the ``feature_names_in_`` attribute for
all estimators, and the ``feature_names_out_`` attribute for all transformers.
We here discuss the generation of such attributes and their propagation through
pipelines. Since for most estimators there are multiple ways to generate
feature names, this SLEP does not intend to define how exactly feature names
are generated for all of them.

Motivation
##########

``scikit-learn`` has been making it easier to build complex workflows with the
``ColumnTransformer`` and it has been seeing widespread adoption. However,
using it results in pipelines where it's not clear what the input features to
the final predictor are, even more so than before. For example, after fitting
the following pipeline, users should ideally be able to inspect the features
going into the final predictor::


    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

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
  ``verbose_feature_names`` as explained later can apply here.

This proposal talks about how feature names are generated and not how they are
propagated.

Scope
#####

The API for input and output feature names includes a ``feature_names_in_``
attribute for all estimators, and a ``feature_names_out_`` attribute for any
estimator with a ``transform`` method, *i.e.* they expose the generated feature
names via the ``feature_names_out_`` attribute.

Note that this SLEP also applies to `resamplers
<https://github.com/scikit-learn/enhancement_proposals/pull/15>`_ the same way
as transformers.

Input Feature Names
###################

The input feature names are stored in a fitted estimator in a
``feature_names_in_`` attribute, and are taken from the given input data, for
instance a ``pandas`` data frame. This attribute will be ``None`` if the input
provides no feature names.

Output Feature Names
####################

A fitted estimator exposes the output feature names through the
``feature_names_out_`` attribute. Here we discuss more in detail how these
feature names are generated. Since for most estimators there are multiple ways
to generate feature names, this SLEP does not intend to define how exactly
feature names are generated for all of them. It is instead a guideline on how
they could generally be generated. 

As detailed bellow, some generated output features names are the same or a
derived from the input feature names. In such cases, if no input feature names
are provided, ``x0`` to ``xn`` are assumed to be their names.

Feature Selector Transformers
*****************************

This includes transformers which output a subset of the input features, w/o
changing them. For example, if a ``SelectKBest`` transformer selects the first
and the third features, and no names are provided, the ``feature_names_out_``
will be ``[x0, x2]``.

Feature Generating Transformers
*******************************

The simplest category of transformers in this section are the ones which
generate a column based on a single given column. These would simply
preserve the input feature names if a single new feature is generated,
such as in ``StandardScaler``, which would map ``'age'`` to ``'age'``.
If an input feature maps to multiple new
features, a postfix is added, so that ``OneHotEncoder`` might map
``'gender'`` to ``'gender_female'`` ``'gender_fluid'`` etc.

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

Examples
########

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


    [model, make, numeric0, ..., numeric100] ->
        ColumnTransformer(
            [('cat', Pipeline(SimpleImputer(), OneHotEncoder()),
              ['model', 'make']),
             ('num', Pipeline(SimpleImputer(), PCA(n_components=3)),
              ['numeric0', ..., 'numeric100'])]
        )
    feature_names_out_: ['cat_model_100', 'cat_model_200', ...,
                         'cat_make_ABC', 'cat_make_XYZ', ...,
                         'num_pca0', 'num_pca1', 'num_pca2']

However, the following examples produce a somewhat redundant feature names::

    [model, make, numeric0, ..., numeric100] ->
        ColumnTransformer([
            ('ohe', OneHotEncoder(), ['model', 'make']),
            ('pca', PCA(n_components=3), ['numeric0', ..., 'numeric100'])
        ])
    feature_names_out_: ['ohe_model_100', 'ohe_model_200', ...,
                         'ohe_make_ABC', 'ohe_make_XYZ', ...,
                         'pca_pca0', 'pca_pca1', 'pca_pca2']

Extensions
##########

verbose_feature_names
*********************
To provide more control over feature names, we could add a boolean
``verbose_feature_names`` constructor argument to certain transformers.
The default would reflect the description above, but changes would allow more verbose
names in some transformers, say having ``StandardScaler`` map ``'age'`` to ``'scale(age)'``.

In case of the ``ColumnTransformer`` example above ``verbose_feature_names``
could remove the estimator names, leading to shorter and less redundant names::

    [model, make, numeric0, ..., numeric100] ->
        make_column_transformer(
            (OneHotEncoder(), ['model', 'make']),
            (PCA(n_components=3), ['numeric0', ..., 'numeric100']),
            verbose_feature_names=False
        )
    feature_names_out_: ['model_100', 'model_200', ...,
                         'make_ABC', 'make_XYZ', ...,
                         'pca0', 'pca1', 'pca2']

Alternative solutions to a boolean flag could include:

- an integer: fine tuning the verbosity of the generated feature names.
- a ``callable`` which would give further flexibility to the user to generate
  user defined feature names.

These alternatives may be discussed and implemented in the future if deemed
necessary.

Backward Compatibility
######################

All estimators should implement the ``feature_names_in_`` and
``feature_names_out_`` API. This is checked in ``check_estimator``, and the
transition is done with a ``FutureWarning`` for at least two versions to give
time to third party developers to implement the API.
