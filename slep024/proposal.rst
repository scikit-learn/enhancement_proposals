.. _slep_024:

===========================================================================
SLEP024: Guideline for external contributions to the scikit-learn blog post
===========================================================================

:Author: Guillaume Lemaitre, François Goupil, Gaël Varoquaux
:Status: Draft
:Type: Standards Track
:Created: 2024-08-09

Abstract
--------

This SLEP proposes some guidelines for writing and reviewing external contributions
to the scikit-learn blog post.

The motivation is this SLEP is to increasingly open up the scikit-learn blog for
external contribution. Editorial guidelines are difficult to build. The goal of this
SLEP is not to give all the details (they can be refined on the blog website), but the
general guiding lines.

Detailed description
--------------------

Scikit-learn has a blog post available at the following URL:
https://blog.scikit-learn.org/. Since its origin, the blog post is used to relay
information related to diverse subject such as sprints, interviews of contributors,
collaborations, and technical content. Hosting quality content on the scikit-learn blog
increases the visibility of scikit-learn and we should foster it.

Technical content limited to the scikit-learn library is natural on the blog.
However, the scikit-learn community goes beyond the
library itself and had developed compatible tools for years. As an example, the
scikit-learn-contrib repository [2]_ hosts a collection of tools which are not
part of the main library but are still compatible with scikit-learn.

This SLEP proposes to extend the scope of the technical content of the blog post to
accept contributions in link with the scikit-learn ecosystem but not limited to the
scikit-learn library itself. However, it is necessary to define some guidelines to
manage expectations of contributors and readers.

Here, we define the **guidelines for external contributions** that should be used to
write and review external contributions to the scikit-learn blog post.

Guidelines
----------

In this section, we provide a set of guidelines to ease the discussions when reviewing
external contributions to the scikit-learn blog post. It should help both the authors
and the reviewers.

**Philosophy**: the scikit-learn blog lives because people take the effort of writing
posts. These contributions are welcomed to produce quality content on machine learning
and the scikit-learn ecosystem. However, we want the blog to be a place of quality
content which encourages a thriving open-source ecosystem, and which to avoid it being
flooded by advertising or special interests.

General criteria for accepting a contributed blog post
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To accept an external contribution, the blog post should be related to the scikit-learn
ecosystem.

When it comes to presenting **a compatible tool**, the guidelines are the following:

- The tool should be compatible with scikit-learn.
- The tools used in the blog need to be under an OSI approved or similar license.
- The tool should be actively maintained.
- The tool should have a clear documentation.
- The tool should be well tested.

More generally, even for content not presenting a tool, the following guidelines apply:

- The content should adhere to our code of conduct standards.
- The content should not be advertisement for a commercial endeavor (see below)
- Claims should, as much as possible, backed by authoritative sources or reproducible
  code (see below)

Note that there will be an element of human judgment in applying these inclusion
criteria.

Commercial links
^^^^^^^^^^^^^^^^

A blog article  should not be on a commercial product or serve advertisement for a
company. Likewise the content of the post should be limited to the tool at hand, rather
than using the tool to advertise a commercial ecosystem / tool.

However, organizations that financially sponsor the project by hiring (near-)full-time
core contributors can use the scikit-learn blog to advertise open-source resources
related to scikit-learn in their own name as long as the relationship to the
scikit-learn project is made explicit in the blog post.

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
