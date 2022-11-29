:orphan:

.. _slep_006_other:

Alternative solutions to sample-aligned meta-data
=================================================

This page contains alternative solutions that have been discussed
and finally not considered in the SLEP.

Solution sketches require these definitions:

.. literalinclude:: defs.py

Status quo solution 0a: additional feature
------------------------------------------

Without changing scikit-learn, the following hack can be used:

Additional numeric features representing sample props can be appended to the
data and passed around, being handled specially in each consumer of features
or sample props.

.. literalinclude:: cases_opt0a.py

Status quo solution 0b: Pandas Index and global resources
---------------------------------------------------------

Without changing scikit-learn, the following hack can be used:

If `y` is represented with a Pandas datatype, then its index can be used to
access required elements from props stored in a global namespace (or otherwise
made available to the estimator before fitting). This is possible everywhere
that a ground-truth `y` is passed, including fit, split, score, and metrics.
A similar solution with `X` is also possible (except for metrics), if all
Pipeline components retain the original Pandas Index.

Issues:

* use of global data source
* requires Pandas data types and indices to be maintained

.. literalinclude:: cases_opt0b.py

Solution 1: Pass everything
---------------------------

This proposal passes all props to all consumers (estimators, splitters,
scorers, etc). The consumer would optionally use props it is familiar with by
name and disregard other props.

We may consider providing syntax for the user to control the interpretation of
incoming props:

* to require that some prop is provided (for an estimator where that prop is
  otherwise optional)
* to disregard some provided prop
* to treat a particular prop key as having a certain meaning (e.g. locally
  interpreting 'scoring_sample_weight' as 'sample_weight').

These constraints would be checked by calling a helper at the consumer.

Issues:

* Error handling: if a key is optional in a consumer, no error will be
  raised for misspelling. An introspection API might change this, allowing a
  user or meta-estimator to check if all keys passed are to be used in at least
  one consumer.
* Forwards compatibility: newly supporting a prop key in a consumer will change
  behaviour. Other than a ChangedBehaviorWarning, I don't see any way around
  this.
* Introspection: not inherently supported. Would need an API like
  ``get_prop_support(names: List[str]) -> Dict[str, Literal["supported", "required", "ignored"]]``.

In short, this is a simple solution, but prone to risk.

.. literalinclude:: cases_opt1.py


Solution 2: Specify routes at call
----------------------------------

Similar to the legacy behavior of fit parameters in
:class:`sklearn.pipeline.Pipeline`, this requires the user to specify the
path for each "prop" to follow when calling `fit`.  For example, to pass
a prop named 'weights' to a step named 'spam' in a Pipeline, you might use
`my_pipe.fit(X, y, props={'spam__weights': my_weights})`.

SLEP004's syntax to override the common routing scheme falls under this
solution.

Advantages:

* Very explicit and robust to misspellings.

Issues:

* The user needs to know the nested internal structure, or it is easy to fail
  to pass a prop to a specific estimator.
* A corollary is that prop keys need changing when the developer modifies their
  estimator structure (see case C).
* This gets especially tricky or impossible where the available routes
  change mid-fit, such as where a grid search considers estimators with
  different   structures.
* We would need to find a different solution for :issue:`2630` where a Pipeline
  could not be the base estimator of AdaBoost because AdaBoost expects the base
  estimator to accept a fit param keyed 'sample_weight'.
* This may not work if a meta-estimator were to have the role of changing a
  prop, e.g. a meta-estimator that passes `sample_weight` corresponding to
  balanced classes onto its base estimator.  The meta-estimator would need a
  list of destinations to pass modified props to, or a list of keys to modify.
* We would need to develop naming conventions for different routes, which may
  be more complicated than the current conventions; while a GridSearchCV
  wrapping a Pipeline currently takes parameters with keys like
  `{step_name}__{prop_name}`, this explicit routing, and conflict with
  GridSearchCV routing destinations, implies keys like
  `estimator__{step_name}__{prop_name}`.

.. literalinclude:: cases_opt2.py


Solution 3: Specify routes on metaestimators
--------------------------------------------

Each meta-estimator is given a routing specification which it must follow in
passing only the required parameters to each of its children. In this context,
a GridSearchCV has children including `estimator`, `cv` and (each element of)
`scoring`.

Pull request :pr:`9566` and its extension in :pr:`15425` are partial
implementations of this approach.

A major benefit of this approach is that it may allow only prop routing
meta-estimators to be modified, not prop consumers.

All consumers would be required to check that 

Issues:

* Routing may be hard to get one's head around, especially since the prop
  support belongs to the child estimator but the parent is responsible for the
  routing.
* Need to design an API for specifying routings.
* As in Solution 2, each local destination for routing props needs to be given
  a name.
* Every router along the route will need consistent instructions to pass a
  specific prop to a consumer. If the prop is optional in the consumer, routing
  failures may be hard to identify and debug.
* For estimators to be cloned, this routing information needs to be cloned with
  it. This implies one of: the routing information be stored as a constructor
  parameter; or `clone` is extended to explicitly copy routing information.

Possible public syntax:

Each meta-estimator has a `prop_routing` parameter to encode local routing
rules, and a set of named children which it routes to. In :pr:`9566`, the
`prop_routing` entry for each child may be a white list or black list of
named keys passed to the meta-estimator.

.. literalinclude:: cases_opt3.py
