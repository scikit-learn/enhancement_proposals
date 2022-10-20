.. _slep_019:

==================================================================
SLEP019: Governance Update - Recognizing Contributions Beyond Code
==================================================================

:Author: Julien Jerphanion, Gaël Varoquaux
:Status: Accepted
:Type: Process
:Created: 2022-09-12

Abstract
--------

This SLEP proposes updating the Governance to broaden the notion of contribution
in scikit-learn and to ease subsequent change to the Governance without requiring
SLEPs.

Motivation
----------

Current state
~~~~~~~~~~~~~

So far the formal decision process in the scikit-learn project is anchored on by
a subset of contributors, called Core Developers (also refered to as
Maintainers). Their active and consistent contributions are recognized by them:

- being part of scikit-learn organisation on GitHub
- receiving “commit rights” to the repository
- having their review recognised as authoritative
- having vote rights for the project direction (promoting a contributor to be a
  core-developer, approving a SLEP, etc.)

However, there are a lot of other ways to contribute to the project, whose
efforts are actually not recognized [1]_. To quote Melissa Weber Mendonça [2]_
and Reshama Shaikh [3]_:

.. epigraph::
  "When some people join an open source project, they may be asked to contribute
  with tasks that will never get them on a path to any sort of official input,
  such as voting rights."

Desired Goal: incrementally adapt the Governance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We need to:

- value non-coding contributions in the project and acknowledge all efforts,
  including those that are not quantified by GitHub users' activity
- empower more contributors in effectively being able to participate in the
  projects without requiring the security responsibilities of tracking code
  changes to the main branches. These considerations should lead to the
  diversification of contribution paths [4]_.

Rather than introducing a fully new fledge structure and Governance, we
propose getting in a state which allows small incremental changes overtime.

Proposed changes
----------------

Some of the proposed modification have been informally discussed in monthly meetings,
namely on April 25th 2022 [5]_ and on September 5th 2022 [6]_ meetings.

Define "Contributions" explicitly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Give more emphasis to the notion of contribution at large (beyond code) in our
wording.

Create a Triagers Team
~~~~~~~~~~~~~~~~~~~~~~

Create a Triagers Team which would be given "Write" permissions on GitHub [7]_ 
to be able to perform triaging tasks, such as editing issues' description.


Define "Core Contributors"
~~~~~~~~~~~~~~~~~~~~~~~~~~

To give decision power to members of our community responsible for the various
aspects of the project, we will list the follow teams as being team
of "Core Contributors", called "Core Teams":

  - Triagers Team
  - Communication Team
  - Core Developers Team

Contributors are promoted as Core Contributor as part of one of the Core Teams,
after the proposal of at least another Core Contributor. This promotions
is effective after a vote on the private mailing list which must last for two
weeks and which must reach at least two-thirds positive majority of the cast votes.

Extend voting rights
~~~~~~~~~~~~~~~~~~~~

Give voting rights beyond to all member of Core Teams.

Simplify subsequent changes to the Governance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With acceptance of this SLEP, changes to the scikit-learn Governance concerning the
following aspects will not require being backed by a SLEP in the future:

  - additions and changes to Roles' and Teams' scopes
  - additions and changes to Roles' and Teams' permissions

Still any changes to scikit-learn Governance (including the ones which will not require
being back by a SLEP) will still be subject to the decision making process [8]_,
which includes a Core Contributors' vote.

If subsequent changes to the Governance are proposed through a GitHub Pull Request (PR):

 - a positive vote is cast by approving the PR (i.e. "Approve" review)
 - a negative vote is cast by requesting changes to the PR (i.e. "Request changes" review)

Copyright
---------

This document has been placed in the public domain [9]_.

References and Footnotes
------------------------

.. [1] J. -G. Young, A. Casari, K. McLaughlin, M. Z. Trujillo, L. Hébert-Dufresne and
    J. P. Bagrow, "Which contributions count? Analysis of attribution in open source,"
    2021 IEEE/ACM 18th International Conference on Mining Software Repositories (MSR),
    2021, pp. 242-253, doi: 10.1109/MSR52588.2021.00036:
    https://arxiv.org/abs/2103.11007

.. [2] Contributor experience, diversity and culture in Open Source Projects:
    keynote from Melissa Weber Mendonça: https://2022.pycon.de/program/NVBLKH/

.. [3] Reshama Shaikh's quote from Melissa Weber Mendonça' keynote:
    https://twitter.com/reshamas/status/1513488342767353857

.. [4] NumPy Newcomer's Hour: an Experiment on Community Building, talk from
    Melissa Weber Mendonça: https://www.youtube.com/watch?v=c0XZQbu0xnw

.. [5] scikit-learn April 25th 2022 Developer meeting notes:
    https://github.com/scikit-learn/administrative/blob/master/meeting_notes/2022-04-25.md

.. [6] scikit-learn September 5th 2022 Developer meeting notes:
    https://github.com/scikit-learn/administrative/blob/master/meeting_notes/2022-09-05.md
.. [7] Permissions for each role, Repository roles for an organization, GitHub Docs:
    https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-roles-for-an-organization#permissions-for-each-role

.. [8] Decision Making Process, scikit-learn Governance and Decision-Making: 
    https://scikit-learn.org/dev/governance.html#decision-making-process

.. [9] Open Publication License: https://www.opencontent.org/openpub/

