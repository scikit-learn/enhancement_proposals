Transformers that modify their target
=====================================

<div class="topic">

****Summary****

Transformers implement:

    self = estimator.fit(X, y=None)
    X_transform = estimator.transform(X)
    estimator.fit(X, y=None).transform(X) == estimator.fit_transform(X, y)

Within a chain or processing sequence of estimators, many usecases
require modifying y. How do we support this?

Doing many of these things is possible "by hand". The question is: how
to avoid writing custom connecting logic.

</div>

Rational
--------

### Summary of the contract of transformers

-   .transform(...) returns a data matrix X
-   .transform(...) returns one feature vector for each sample of the
    input
-   .fit\_transform(...) is the same and .fit(...).transform(...)

### Examples of usecases targetted

1.  Over sampling:
    1.  Class rembalancing: over sampling the minority class in
        unbalanced dataset
    2.  Data enhancement (nudgging images for instance)
2.  Under-sampling
    1.  Stateless undersampling: Take one sample out of two
    2.  Stateful undersampling: apply clustering and transform to
        cluster centers
    3.  Coresets: return a smaller number of samples and associated
        sample weights
3.  Outlier detection:
    1.  Remove outlier from train set
    2.  Create a special class 'y' for outliers
4.  Completing y:
    1.  Missing data imputation on y
    2.  Semi-supervised learning (related to above)
5.  Data loading / conversion
    1.  Pandas in =&gt; (X, y) out
    2.  Images in =&gt; patches out
    3.  Filename in =&gt; (X, y) with multiple samples (very useful in
        combination with online learning)
    4.  Database query =&gt; (X, y) out
6.  Aggregate statistics over multiple samples

    1.  Windowing-like functions on time-series

    In a sense, these are dodgy with scikit-learn's cross-validation API
    that knows nothing about sample structure. But the refactor of the
    CV API is really helping in this regard.

These usecases pretty much require breaking the contract of the
Transformer, as detailed above.

The intuition driving this enhancement proposal is that the more the
data-processing pipeline becomes rich, the more the data grow, the more
the usecases above become important.

Enhancements proposed
---------------------

### Option A: meta-estimators

#### Proposal

This option advocates that any transformer-like usecase that wants to
modify y or the number of samples should not be a transformer-like but a
specific meta-estimator. A core-set object would thus look like:

-   From the user perspective:

        from sklearn.sample_shrink import BirchCoreSet
        from sklearn.ensemble import RandomForest
        estimator = BirchCoreSet(RandomForest())

-   From the developer perspective:

        class BirchCoreSet(BaseEstimator):

           def fit(self, X, y):
               # The logic here is wrong, as we need to handle y:
               super(BirchCoreSet, self).fit(X)
               X_red = self.subcluster_centers_
               self.estimator_.fit(X_red)

#### Benefits

1.  No change to the existing API
2.  The meta-estimator pattern is very powerful, and pretty much
    anything is possible.

#### Limitations

The different limitations listed below are variants of the same
conceptual difficulty

1.  It is hard to have mental models and garantees of what a
    meta-estimator does, as it is by definition super versatile

    This is both a problem for the beginner, that needs to learn them on
    an almost case-by-case basis, and for the advanced user, that needs
    to maintain a set of case-specific code

2.  The "estimator heap" problem.

    Here the word heap is used to denote the multiple pipelines and
    meta-estimators. It corresponds to what we would naturally call a
    "data processing pipeline", but we use "heap" to avoid confusion
    with the pipeline object.

    Heaps combining many steps of pipelines and meta-estimators become
    very hard to inspect and manipulate, both for the user, and for
    pipeline-management (aka "heap-management") code. Currently, these
    difficulties are mostly in user code, so we don't see them too much
    in scikit-learn. Here are concrete examples

    1.  Trying to retrieve coefficients from a model estimated in a
        "heap". Eg:

        -   you know there is a lasso in your stack and you want to get
            it's coef (in whatever space that resides?):
            pipeline.named\_steps\['lasso'\].coef\_ is possible.
        -   you want to retrieve the coef of the last step:
            pipeline.steps\[-1\]\[1\].coef\_ is possible.

        With meta estimators this is tricky. Solving this problem
        requires
        <https://github.com/scikit-learn/scikit-learn/issues/2562#issuecomment-27543186>
        (this enhancement proposal is not advocating to solve the
        problem above, but pointing it out as an illustration)

    2.  DaskLearn has modified the logic of pipeline to expose it as a
        computation graph. The reason that it was relatively easy to do
        is that there was mostly one object to modify to do the
        dispatching, the Pipeline object.
    3.  A future, out-of-core "conductor" object to fit a "heap" in out
        of core by connecting it to a data-store would need to have a
        representation of the heap. For instance, when chaining random
        projections with Birch coresets and finally SGD, the user would
        need to specify that random projections are stateless, birch
        needs to do one pass of the data, and SGD a few. Given this
        information, the conductor could orchestrate pull the data from
        the data source, and sending it to the various steps. Such an
        object is much harder to implement if the various steps are to
        be combined in a heap. Note that the scikit-learn pipeline can
        only implement a linear "chain" like set of processing. For
        instance a One vs All will never be able to be implemented in a
        scikit-learn pipeline.

        This is not a problem in non out-of-core settings, in the sense
        that the BirchCoreSet meta-estimator would take care of doing a
        pass on the data before feeding it to its sub estimator.

In conclusion, meta-estimators are harder to comprehend (problem 1) and
write (problem 2).

That said, we will never get rid of meta estimators. It is a very
powerful pattern. The discussion here is about extending a bit the
estimator API to have a less pressing need for meta-estimators.

### Option B: transformer-like that modify y

<div class="topic">

****Two variants****

1.  Changing the semantics of transformers to modify y and return
    something more complex than a data matrix X
2.  Introducing new methods (and a new type of object)

There is an emerging consensus for option 2.

</div>

<div class="topic">

****transform modifying y****

Variant 1 above could be implementing by allowing transform to modify y.
However, the return signature of transform would be unclear.

Do we modify all transformers to return a y (y=None for unsupervised
transformers that are not given y?). This sounds like leading to code
full of surprises and difficult to maintain from the user perspective.

We would loose the contract that the number of samples is unchanged by a
transformer. This contract is very useful (eg for model selection:
measuring error for each sample).

For these reasons, we feel new methods are necessary.

</div>

#### Proposal

Introduce a TransModifier type of object with the following API (names
are discussed below):

-   X\_new, y\_new = estimator.fit\_modify(X, y)
-   X\_new, y\_new = estimator.trans\_modify(X, y)

Or:

-   X\_new, y\_new, sample\_props = estimator.fit\_modify(X, y)
-   X\_new, y\_new, sample\_props = estimator.trans\_modify(X, y)

Contracts (these are weaker contracts than the transformer:

-   Neither fit\_modify nor trans\_modify are guarantied to keep the
    number of samples unchanged.
-   fit\_modify may not exist (questionnable)

#### Design questions and difficulties

##### Should there be a fit method?

In such estimators, it may not be a good idea to call fit rather than
fit\_modify (for instance in coreset).

##### How does a pipeline use such an object?

In particular at test time?

1.  Should there be a transform method used at test time?
2.  What to do with objects that implement both transform and
    trans\_modify?

**Creating y in a pipeline makes error measurement harder** For some
usecases, test time needs to modify the number of samples (for instance
data loading from a file). However, these will by construction a problem
for eg cross-val-score, as in supervised settings, these expect a
y\_true. Indeed, the problem is the following:

-   To measure an error, we need y\_true at the level of
    cross\_val\_score or GridSearchCV
-   y\_true is created inside the pipeline by the data-loading object.

It is thus unclear that the data-loading usecases can be fully
integrated in the CV framework (which is not an argument against
enabling them).

| 

For our CV framework, we need the number of samples to remain constant:
for each y\_pred, we need a corresponding y\_true.

| 

**Proposal 1**: use transform at predict time.

1.  Objects implementing both transform and trans\_modify are valid
2.  The pipeline's predict method use transform on its intermediate
    steps

The different semantics of trans\_modify and transform can be very
useful, as transform keeps untouched the notion of sample, and y\_true.

| 

**Proposal 2** Modify the scoring framework

One option is to modify the scoring framework to be able to handle these
things, the scoring gets the output of the chain of trans\_modify for y.
This should rely on clever code in the score method of pipeline. Maybe
it should be controlled by a keyword argument on the pipeline, and
turned off by default.

##### How do we deal with sample weights and other sample properties?

This discussion feeds in the sample\_props discussion (that should be
discussed in a different enhancement proposal).

The suggestion is to have the sample properties as a dictionary of
arrays sample\_props.

**Example usecase** useful to think about sample properties: coresets:
given (X, y) return (X\_new, y\_new, weights) with a much smaller number
of samples.

This example is interesting because it shows that TransModifiers can
legitimately create sample properties.

**Proposed solution**:

TransModifiers always return (X\_new, y\_new, sample\_props) where
sample\_props can be an empty dictionary.

#### Naming suggestions

In term of name choice, the rational would be to have method names that
are close to 'fit' and 'transform', to make discoverability and
readability of the code easier.

-   Name of the object (referred in the docs):
    -   TransModifier
    -   TransformPipe
    -   PipeTransformer
-   Method to fit and apply on training
    -   fit\_modify
    -   fit\_pipe
    -   pipe\_fit
    -   fit\_filter
-   Method to apply on new data
    -   trans\_modify
    -   transform\_pipe
    -   pipe\_transform

#### Benefits

-   Many usecases listed above will be implemented scikit-learn without
    a meta-estimator, and thus will be easy to use (eg in a pipeline).
    Many of these are patterns that we should be encouraging.
-   The API being more versatile, it will be easier to create
    application-specific code or framework wrappers (ala DaskLearn) that
    are scikit-learn compatible, and thus that can be used with the
    parameter-selection framework. This will be especially true for ETL
    (extract transform and load) pattern.

#### Limitations

-   Introducing new methods, and a new type of estimator object. There
    are probably a total of **3 new methods** that will get introduced
    by this enhancement: fit\_modify, trans\_modify, and
    partial\_fit\_modify.
-   Cannot solve all possible cases, and thus we will not get rid of
    meta-estimators.

TODO
----

-   Implement an example doing outlier filtering
-   Implement an example doing data downsampling

