.. _slep_019:

####################################################################
 SLEP019: Governance Update - Recognizing Contributions Beyond Code
####################################################################

:Author: Julien Jerphanion <git@jjerphan.xyz>,
         Gaël Varoquaux <gael.varoquaux@normalesup.org>,
         Chiara Marmo <marmochiaskl@gmail.com>
:Status: Draft
:Type: Process
:Created: 2022-09-12

**********
 Abstract
**********

This SLEP proposes updating the Governance to broaden the notion of contribution in scikit-learn.
To this end, the Technical Committee is renamed to Steering Committee, opening the
decision process to new, to be defined, core roles.
New core roles are allowed to be defined without requiring new SLEPs, to ease subsequent
related changes to the Governance.

************
 Motivation
************

Current state
=============

The formal decision making process of the scikit-learn project is
limited to a subset of contributors, called Core Developers (also
refered to as Maintainers). Their active and consistent contributions
are recognized by them:

-  being part of scikit-learn organisation on GitHub
-  receiving “commit rights” to the repository
-  having their Pull Request reviews recognised as authoritative
-  having voting rights for the project direction (promoting a
   contributor to be a core-developer, approving a SLEP, etc.)

Core Developers are primarily selected based on their code
contributions. However, there are a lot of other ways to contribute to
the project, and these efforts are currently not recognized [1]_. To
quote Melissa Weber Mendonça [2]_ and Reshama Shaikh [3]_:

.. epigraph::

   "When some people join an open source project, they may be asked to contribute
   with tasks that will never get them on a path to any sort of official input,
   such as voting rights."

Desired Goal: incrementally adapt the Governance
================================================

We need to:

-  value non-coding contributions in the project and acknowledge all
   efforts, including those that are not quantified by GitHub users'
   activity

-  empower more contributors to effectively participate in the project
   without requiring the security responsibilities of tracking code
   changes to the main branches. These considerations should lead to the
   diversification of contribution paths [4]_.

Rather than introducing entirely new structure and Governance, we
propose changes to the existing ones which allow for small incremental
modifications over time.

******************
 Proposed changes
******************

Some of the proposed modification have been discussed in the monthly
meetings, on April 25th 2022 [5]_ and September 5th 2022 [6]_.
The discussion about the proposal is publicly available [7]_ [8]_.  

Define "Contributions" more broadly
===================================

Explicitly define Contributions and emphasize the importance of non-code
contributions in the Governance structure.

New roles, core roles, and subsequent teams will be defined.
Core Team members will become scikit-learn "Core Contributors":
a Contributor will be promoted to a Core Contributor after being proposed by
at least one existing Core Contributor. The proposal must specify which
Core Team the Contributor will be part of. The promotion will be effective
after a vote on the private Core Contributor mailing list which must
last for two weeks and which must reach at least two-thirds positive
majority of the cast votes.

A Core Contributor may belong to multiple core roles.

Evolve the Technical Committee into a Steering Committee
========================================================

Rename "Technical Committee" to "Steering Committee".

Define the Steering Committee as a subset of Core Contributors rather
than a subset of Core Developers.

The current members of the Technical Committee [9]_ will ensure the transition
for a one-years period, starting from the acceptance of this SLEP.
During the transition period, each new created role among the "Core Contributors" is
expected to be represented in the Steering Committee by at least one member having this
role.

Extend voting rights
====================

Give voting rights to all Core Contributors.

Simplify subsequent changes to the Governance
=============================================

Allow changes to the following aspects of the scikit-learn Governance
without requiring a SLEP:

   -  additions and changes to Roles' and Teams' scopes
   -  additions and changes to Roles' and Teams' permissions

Any changes to the scikit-learn Governance (including ones which do not
require being back by a SLEP) will continue to be subject to the
decision making process [10]_, which includes a vote of the Core
Contributors.

If subsequent changes to the Governance are proposed through a GitHub
Pull Request (PR):

   -  a positive vote is cast by approving the PR (i.e. "Approve"
      review)
   -  a negative vote is cast by requesting changes to the PR (i.e.
      "Request changes" review)

In this case, the vote still has to be announced on the Core
Contributors' mailing list, but the system of Pull Request approvals
will replace a vote on the private Core Contributors' mailing list.

***********
 Copyright
***********

This document has been placed in the public domain [11]_.

**************************
 References and Footnotes
**************************

.. [1]

   J. -G. Young, A. Casari, K. McLaughlin, M. Z. Trujillo, L.
   Hébert-Dufresne and J. P. Bagrow, "Which contributions count? Analysis
   of attribution in open source," 2021 IEEE/ACM 18th International
   Conference on Mining Software Repositories (MSR), 2021, pp. 242-253,
   doi: 10.1109/MSR52588.2021.00036: https://arxiv.org/abs/2103.11007

.. [2]

   Contributor experience, diversity and culture in Open Source Projects:
   keynote from Melissa Weber Mendonça:
   https://2022.pycon.de/program/NVBLKH/

.. [3]

   Reshama Shaikh's quote from Melissa Weber Mendonça' keynote:
   https://twitter.com/reshamas/status/1513488342767353857

.. [4]

   NumPy Newcomer's Hour: an Experiment on Community Building, talk from
   Melissa Weber Mendonça: https://www.youtube.com/watch?v=c0XZQbu0xnw

.. [5]

   scikit-learn April 25th 2022 Developer meeting notes:
   https://github.com/scikit-learn/administrative/blob/master/meeting_notes/2022-04-25.md

.. [6]

   scikit-learn September 5th 2022 Developer meeting notes:
   https://github.com/scikit-learn/administrative/blob/master/meeting_notes/2022-09-05.md

.. [7]

   SLEP019: Governance Update - Recognizing Contributions Beyond Code
   https://github.com/scikit-learn/enhancement_proposals/pull/74

.. [8]

   SLEP019: Governance Update - Recognizing Contributions Beyond Code
   https://github.com/scikit-learn/enhancement_proposals/pull/81

.. [9]

   scikit-learn Technical Committee
   https://scikit-learn.org/1.2/governance.html#technical-committee

.. [10]

   Decision Making Process, scikit-learn Governance and Decision-Making:
   https://scikit-learn.org/stable/governance.html#decision-making-process

.. [11]

   Open Publication License: https://www.opencontent.org/openpub/
