.. _slep_019:

####################################################################
 SLEP019: Governance Update - Recognizing Contributions Beyond Code
####################################################################

:Author: Julien Jerphanion <git@jjerphan.xyz>, Gaël Varoquaux <gael.varoquaux@normalesup.org>
:Status: Draft
:Type: Process
:Created: 2022-09-12

**********
 Abstract
**********

This SLEP proposes updating the Governance to broaden the notion of
contribution in scikit-learn and to ease subsequent related changes to
the Governance without requiring SLEPs.

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

Define "Contributions" more broadly
===================================

Explicitly define Contributions and emphasize the importance of non-code
contributions in the Governance structure.

Evolve the Technical Committee into a Steering Committee
========================================================

Rename "Technical Committee" to "Steering Committee".

Define the Steering Committee as a subset of Core Contributors rather
than a subset of Core Developers.

Create a Triage Team
====================

Create a Triage Team which would be given "Write" permissions on GitHub
[7]_ to be able to perform triaging tasks, such as editing issues'
description.

Define "Core Contributors"
==========================

Establish all members of the following teams as "Core Contributors":

   -  Triage Team
   -  Communication Team
   -  Development Team

A Contributor is promoted to a Core Contributor after being proposed by
at least one existing Core Contributor. The proposal must specify which
Core Team the Contributor will be part of. The promotion is effective
after a vote on the private Core Contributor mailing list which must
last for two weeks and which must reach at least two-thirds positive
majority of the cast votes.

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
decision making process [8]_, which includes a vote of the Core
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


**************
Acknowledgment
**************

We thank the following people who have helped with discussions during the
development of this SLEP:

- Lucy Liu: https://github.com/lucyleeow
- Noa Tamir: https://github.com/noatamir
- Reshama Shaikh: https://github.com/reshamas
- Tim Head: https://github.com/betatim

***********
 Copyright
***********

This document has been placed in the public domain [9]_.

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

   Permissions for each role, Repository roles for an organization, GitHub
   Docs:
   https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-roles-for-an-organization#permissions-for-each-role

.. [8]

   Decision Making Process, scikit-learn Governance and Decision-Making:
   https://scikit-learn.org/dev/governance.html#decision-making-process

.. [9]

   Open Publication License: https://www.opencontent.org/openpub/
