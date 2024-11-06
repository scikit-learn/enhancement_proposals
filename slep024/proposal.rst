.. _slep_024:

===========================================================================
SLEP024: Guideline for external contributions to the scikit-learn blog post
===========================================================================

:Author: Guillaume Lemaitre, François Goupil
:Status: Draft
:Type: Standards Track
:Created: 2024-08-09

Abstract
--------

This SLEP proposes some guidelines for writing and reviewing external contributions
to the scikit-learn blog post.

Detailed description
--------------------

Scikit-learn has a blog post available at the following URL:
https://blog.scikit-learn.org/. Since its origin, the blog post is used to relay
information related to diverse subject such as sprints, interviews of contributors,
collaborations, and technical content.

When it comes to technical content, up to now, the content is only limited to the
scikit-learn library. However, the scikit-learn community is going beyond the
library itself and had developed compatible tools for years. As an example, the
scikit-learn-contrib repository [2]_ is hosting a collection of tools which are not
part of the main library but are still compatible with scikit-learn.

This SLEP proposes to extend the scope of the technical content of the blog post to
accept contributions in link with the scikit-learn ecosystem but not limited to the
scikit-learn library itself. However, it is necessary to define some guidelines to
manage expectations of contributors and readers.

Here, we define the guidelines for external contributions that should be used to
write and review external contributions to the scikit-learn blog post.

Guidelines
----------

In this section, we provide a set of guidelines to ease the discussions when reviewing
external contributions to the scikit-learn blog post. It should help both the authors
and the reviewers.

Inclusion criteria
^^^^^^^^^^^^^^^^^^

To accept an external contribution, the blog post should be related to the scikit-learn
ecosystem. When it comes to presenting a compatible tool, the criteria are the
following:

- The tool should be compatible with scikit-learn.
- The tool should be under an open-source license.
- The tool should be actively maintained.
- The tool should have a clear documentation.
- The tool should be well tested.
- The tool should not be a commercial product or serve advertisement for a company.

Reproducibility requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the scikit-learn documentation, we ensure that our examples are reproducible and can
be executed by using our continuous integration. When it comes to the scikit-learn blog
post, it is not possible (or rather difficult) to have the same level of integration.

However, we should ensure that the given examples or code snippets are reproducible by
the readers. We therefore recommend the following:

- Provide a link to a repository where the code or notebook is available that is used
  as a baseline for the blog post.
- The repository should contain a system to reproduce the environment (e.g.
  `requirements.txt`, `environment.yml`, or `pixi.toml`).
- If possible, a continuous integration should make sure that the code or notebook can
  be executed. We understand that this step is sometimes impossible due to limit of
  resources.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. [2] `scikit-learn-contrib repository <https://github.com/scikit-learn-contrib>`__

.. _Open Publication License: https://www.opencontent.org/openpub/

Copyright
---------

This document has been placed in the public domain. [1]_
