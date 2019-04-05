 .. _slep_007:

=============
Feature Names
=============

`scikit-learn/#13307 <https://github.com/scikit-learn/scikit-learn/pull/13307>`_
proposes a solution to support and propagate feature names through a pipeline.

However, the generated feature names can become long. This is to confirm how
we want those generated feature names to behave, and this is the proposal::


    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.compose import make_column_transformer
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectKBest
    import pandas as pd

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    pipe = make_pipeline(StandardScaler(), PCA(), SelectKBest(k=2),
                         LogisticRegression())
    pipe.fit(X, iris.target)
    pipe[-1].input_features_
    > array(['pca0', 'pca1'], dtype='<U4')


    # I have to duplicate StandardScaler if I want to be able to
    # use the pandas column names in the column transformer. Yuk!
    # also there's no easy way to pass through the original columns with
    # ColumnTransformer - FunctionTransformer would remove feature names!
    pipe = make_pipeline(make_column_transformer((make_pipeline(StandardScaler(),
                                                                PCA()), X.columns),
                                                 (StandardScaler(),
                                                 X.columns[:2])),
                         SelectKBest(k=2), LogisticRegression())
    pipe.fit(X, iris.target)
    pipe[-1].input_features_
    ```
    > array(['pipeline__pca0', 'standardscaler__sepal length (cm)'], dtype='<U33')

    pipe = make_pipeline(make_column_transformer((PCA(), X.columns),
                                                 (StandardScaler(),
                                                 X.columns[:2])),
                         SelectKBest(k=2), LogisticRegression())
    pipe.fit(X, iris.target)
    pipe[-1].input_features_

    > array(['pca__pca0', 'standardscaler__sepal length (cm)'], dtype='<U33')

Is that what we want? (apart from changing to object dtype lol)
The first one seems good to me, the others seem a bit long? Not sure how to do
better though.
