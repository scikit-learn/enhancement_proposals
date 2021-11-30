.. _slep_000:

==============================
SLEP000: SLEP and its workflow
==============================

:Author: Adrin Jalali
:Status: Draft
:Type: Process
:Created: 2020-02-13

Abstract
########

This SLEP specifies details related to SLEP submission, review, and acceptance
process.

Motivation
##########

Without a predefined workflow, the discussions around a SLEP can be long and
consume a lot of energy for both the author(s) and the reviewers. The lack of a
known workflow also results in the SLEPs to take months (if not years) before
it is merged as ``Under Review``. The purpose of this SLEP is to lubricate and
ease the process of working on a SLEP, and make it a more enjoyable and
productive experience. This SLEP borrows the process used in PEPs and NEPs
which means there will be no ``Under Review`` status.


What is a SLEP?
###############

SLEP stands for Scikit-Learn Enhancement Proposal, inspired from Python PEPs or
Numpy NEPs. A SLEP is a design document providing information to the
scikit-learn community, or describing a new feature for scikit-learn or its
processes or environment. The SLEP should provide a concise technical
specification of the proposed solution, and a rationale for the feature.

We intend SLEPs to be the primary mechanisms for proposing major new features,
for collecting community input on an issue, and for documenting the design
decisions that have gone into scikit-learn. The SLEP author is responsible for
building consensus within the community and documenting dissenting opinions.

Because the SLEPs are maintained as text files in a versioned repository, their
revision history is the historical record of the feature proposal.

SLEP Audience
#############

The typical primary audience for SLEPs are the core developers of
``scikit-learn`` and technical committee, as well as contributors to the
project. However, these documents also serve the purpose of documenting the
changes and decisions to help users understand the changes and why they are
made. The SLEPs are available under `Scikit-learn enhancement proposals
<https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/>`_.

SLEP Types
##########

There are three kinds of SLEPs:

1. A Standards Track SLEP describes a new feature or implementation for
scikit-learn.

2. An Informational SLEP describes a scikit-learn design issue, or provides
general guidelines or information to the scikit-learn community, but does not
propose a new feature. Informational SLEPs do not necessarily represent a
scikit-learn community consensus or recommendation, so users and implementers
are free to ignore Informational SLEPs or follow their advice. For instance, an
informational SLEP could be one explaining how people can write a third party
estimator, one to explain the usual process of adding a package to the contrib
org, or what our inclusion criteria are for scikit-learn and
scikit-learn-extra.

3. A Process SLEP describes a process surrounding scikit-learn, or proposes a
change to (or an event in) a process. Process SLEPs are like Standards Track
SLEPs but apply to areas other than the scikit-learn library itself. They may
propose an implementation, but not to scikit-learn’s codebase; they require
community consensus. Examples include procedures, guidelines, changes to the
decision-making process and the governance document, and changes to the tools
or environment used in scikit-learn development. Any meta-SLEP is also
considered a Process SLEP.


SLEP Workflow
#############

A SLEP starts with an idea, which usually is discussed in an issue or a pull
request on the main repo before submitting a SLEP. It is generally a good idea
for the author of the SLEP to gauge the viability and the interest of the
community before working on a SLEP, mostly to save author's time.

A SLEP must have one or more champions: people who write the SLEP following the
SLEP template, shepherd the discussions around it, and seek consensus in the
community.

The proposal should be submitted as a draft SLEP via a GitHub pull request to a
``slepXXX`` directory with the name ``proposal.rst`` where ``XXX`` is an
appropriately assigned three-digit number (e.g., ``slep000/proposal.rst``). The
draft must use the `SLEP — Template and Instructions
<https://github.com/scikit-learn/enhancement_proposals/blob/master/slep_template.rst>`_
file.

Once the PR for the SLEP is created, a post should be made to the mailing list
containing the sections up to “Backward compatibility”, with the purpose of
limiting discussion there to usage and impact. Discussion on the pull request
will have a broader scope, also including details of implementation.

The first draft of the SLEP needs to be approved by at least one core developer
before being merged. Merging the draft does not mean it is accepted or is ready
for the vote. To this end, the SLEP draft is reviewed for structure,
formatting, and other errors. Approval criteria are:

- The draft is sound and complete. The ideas must make technical sense.
- The initial PR reviewer(s) should not consider whether the SLEP seems likely
  to be accepted.
- The title of the SLEP draft accurately describes its content.

Reviewers are generally quite lenient about this initial review, expecting that
problems will be corrected by the further reviewing process. **Note**: Approval
of the SLEP draft is no guarantee that there are no embarrassing mistakes!
Ideally they're avoided, but they can also be fixed later in separate PRs. Once
approved by at least one core developer, the SLEP draft can be merged.
Additional PRs may be made by the champions to update or expand the SLEP, or by
maintainers to set its status, discussion URL, etc.

Standards Track SLEPs (see bellow) consist of two parts, a design document and
a reference implementation. It is generally recommended that at least a
prototype implementation be co-developed with the SLEP, as ideas that sound
good in principle sometimes turn out to be impractical when subjected to the
test of implementation. Often it makes sense for the prototype implementation
to be made available as PR to the scikit-learn repo (making sure to
appropriately mark the PR as a WIP).

Review and Resolution
---------------------

SLEPs are discussed on the mailing list or the PRs modifying the SLEP. The
possible paths of the status of SLEPs are as follows:

.. image:: pep-0001-process_flow.png
   :alt: SLEP process flow diagram

All SLEPs should be created with the ``Draft`` status.

Eventually, after discussion, there may be a consensus that the SLEP should be
accepted – see the next section for details. At this point the status becomes
``Accepted``.

Once a SLEP has been ``Accepted``, the reference implementation must be
completed. When the reference implementation is complete and incorporated into
the main source code repository, the status will be changed to ``Final``. Since
most SLEPs deal with a part of scikit-learn's API, another way of viewing a
SLEP as ``Final`` is when its corresponding API interface is considered stable.

To allow gathering of additional design and interface feedback before
committing to long term stability for a feature or API, a SLEP may also be
marked as ``Provisional``. This is short for "Provisionally Accepted", and
indicates that the proposal has been accepted for inclusion in the reference
implementation, but additional user feedback is needed before the full design
can be considered ``Final``. Unlike regular accepted SLEPs, provisionally
accepted SLEPs may still be ``Rejected`` or ``Withdrawn`` even after the
related changes have been included in a scikit-learn release.

Wherever possible, it is considered preferable to reduce the scope of a
proposal to avoid the need to rely on the ``Provisional`` status (e.g. by
deferring some features to later SLEPs), as this status can lead to version
compatibility challenges in the wider scikit-learn ecosystem.

A SLEP can also be assigned status ``Deferred``. The SLEP author or a core
developer can assign the SLEP this status when no progress is being made on the
SLEP.

A SLEP can also be ``Rejected``. Perhaps after all is said and done it was not
a good idea. It is still important to have a record of this fact. The
``Withdrawn`` status is similar; it means that the SLEP author themselves has
decided that the SLEP is actually a bad idea, or has accepted that a competing
proposal is a better alternative.

When a SLEP is ``Accepted``, ``Rejected``, or ``Withdrawn``, the SLEP should be
updated accordingly. In addition to updating the status field, at the very
least the ``Resolution`` header should be added with a link to the relevant
thread in the mailing list archives or where the discussion happened.

SLEPs can also be ``Superseded`` by a different SLEP, rendering the original
obsolete. The ``Replaced-By`` and ``Replaces`` headers should be added to the
original and new SLEPs respectively.

``Process`` SLEPs may also have a status of ``Active`` if they are never meant
to be completed, e.g. SLEP 1 (this SLEP).

How a SLEP becomes Accepted
---------------------------

A SLEP is ``Accepted`` by the voting mechanism defined in the `governance model
<https://scikit-learn.org/stable/governance.html?highlight=governance>`_. We
need a concrete way to tell whether consensus has been reached. When you think
a SLEP is ready to accept, create a PR changing the status of the SLEP to
``Accepted``, then send an email to the scikit-learn mailing list with a
subject like:

    [VOTE] Proposal to accept SLEP #<number>: <title>

In the body of your email, you should:

- link to the latest version of the SLEP, and a link to the PR accepting the
  SLEP.

- briefly describe any major points of contention and how they were resolved,

- include a sentence like: “The vote will be closed in a month i.e. on
  <the_date>.”

Generally the SLEP author will be the one to send this email, but anyone can do
it; the important thing is to make sure that everyone knows when a SLEP is on
the verge of acceptance, and give them a final chance to respond.

In general, the goal is to make sure that the community has consensus, not
provide a rigid policy for people to try to game. When in doubt, err on the
side of asking for more feedback and looking for opportunities to compromise.

If the final comment and voting period passes with the required majority, then
the SLEP can officially be marked ``Accepted``. The ``Resolution`` header
should link to the PR accepting the SLEP.

If the vote does not achieve a required majority, then the SLEP remains in
``Draft`` state, discussion continues as normal, and it can be proposed for
acceptance again later once the objections are resolved.

In unusual cases, with the request of the author, the scikit-learn technical
committee may be asked to decide whether a controversial SLEP is ``Accepted``,
put back to ``Draft`` with additional recommendation to try again to reach
consensus or definitely ``Rejected``. Please refer to the governance doc for
more details.

Maintenance
-----------

In general, Standards track SLEPs are no longer modified after they have
reached the ``Final`` state as the code and project documentation are
considered the ultimate reference for the implemented feature. However,
finalized Standards track SLEPs may be updated as needed.

Process SLEPs may be updated over time to reflect changes to development
practices and other details. The precise process followed in these cases will
depend on the nature and purpose of the SLEP being updated.

Format and Template
-------------------

SLEPs are UTF-8 encoded text files using the `reStructuredText
<http://docutils.sourceforge.net/rst.html>`_ format. Please see the `SLEP —
Template and Instructions
<https://github.com/scikit-learn/enhancement_proposals/blob/master/slep_template.rst>`_
file and the `reStructuredTextPrimer
<https://www.sphinx-doc.org/en/stable/rest.html>`_ for more information. We use
`Sphinx <https://www.sphinx-doc.org/en/stable/>`_ to convert SLEPs to HTML for
viewing on the web.

Header Preamble
---------------

Each SLEP must begin with a header preamble. The headers must appear in the
following order. Headers marked with * are optional. All other headers are
required::

      :Author: <list of authors' real names and optionally, email addresses>
      :Status: <Draft | Active | Accepted | Deferred | Rejected |
               Withdrawn | Final | Superseded>
      :Type: <Standards Track | Informational | Process>
      :Created: <date created on, in yyyy-mm-dd format>
    * :Requires: <slep numbers>
    * :scikit-learn-Version: <version number>
    * :Replaces: <slep number>
    * :Replaced-By: <slep number>
    * :Resolution: <url>

The Author header lists the names, and optionally the email addresses of all
the authors of the SLEP. The format of the Author header value must be

    Random J. User <address@dom.ain>

if the email address is included, and just

    Random J. User

if the address is not given. If there are multiple authors, each should be on a
separate line.

Copyright
---------

This document has been placed in the public domain [1]_.

References and Footnotes
------------------------

.. [1] _Open Publication License: https://www.opencontent.org/openpub/
