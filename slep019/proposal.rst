.. _slep_019:

===========================
SLEP019: Governance Update
===========================

:Author: Julien Jerphanion, Gaël Varoquaux
:Status: Draft
:Type: Process
:Created: 2022-09-12

Abstract
--------

This SLEP proposes updating the Governance to broaden the notion of contribution
in scikit-learn.

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
efforts are actually not recognized [1]_. To quote Melissa Weber Mendonça [2]_ and
Reshama Shaikh [3]_:

.. epigraph::
  "When some people join an open source project, they may be asked to contribute
  with tasks that will never get them on a path to any sort of official input,
  such as voting rights."

Desired goal
~~~~~~~~~~~~

We need to:

- value non-coding contributions in the project and acknowledge all efforts,
  including those that are not quantified by GitHub users' activity
- empower more contributors in effectively being able to participate in the
  projects without requiring the security responsibilities of tracking code
  changes to the main branches. These considerations should lead to the
  diversification of contribution paths [4]_.

Hence, we propose to explicitly define Contributions, Roles and Teams.

Limited changes to the governance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This SLEP proposes an incremental change to the governance model. It
acknowledges that more could be done, defining more roles and teams. In the
spirit of iterations, the idea is to implement a first change to the governance,
evolve our community accordingly, and later propose new changes based on the
understanding of the resulting community dynamics.

Proposed modifications
----------------------

The proposed modification have been informally discussed in monthly meetings,
namely on April 25th 2022 [5]_ and on September 5th 2022 [6]_ meetings.

Create new Teams
~~~~~~~~~~~~~~~~

**Core Triagers**

Create a "Core Triagers" team, given commit rights as a way to circumvent
limitations in Github's permission system's limitation (such as limited rights
for triage such as being able to edit issues' description). People in this team
should not push code on upstream nor merge PRs, but they would be trusted and
this restriction would be socially enforced without a specific GitHub mechanism.

**Recurring "Contributors Team**

Create a Recurring Contributors Team on GitHub, which has no specific GitHub
permissions, and in which we easily add people. The goal of this team is to
acknowledge people's contributions, by displaying them as part of the
scikit-learn organization on Github [7]_ on their GitHub profile, and on the
project's website "About Us" webpage [8]_.

**Experts Team**

Create a Experts Team, also without specific GitHub permissions. Core
developers can delegate PR approval rights to someone in this team so that the
PR can be merged with one approval from a Core Developer and one approval from
the delegated expert.

Beyond Developers: Recognize all Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Define "Contributions" explicitly**

Give more emphasis to the notion of contribution at large (beyond code) in our
wording.

**Define of "Core Contributors"**

To give decision power to members of our community responsible for the various
aspects of the project, we will list the follow teams as "Core" (associated with
the notion of "Core Contributor"):

  - Core Triage
  - Communication Team
  - Core Developers

**Extend voting rights**

Give voting rights beyond to all member of Core Teams.

Copyright
---------

This document has been placed in the public domain [9]_.

References and Footnotes
------------------------

.. [1] Amanda Casari on ACROSS and Measuring Contributions in OSS, Sustain:
    https://podcast.sustainoss.org/111

.. [2] Contributor experience, diversity and culture in Open Source Projects,
    keynote from Melissa Weber Mendonça: https://2022.pycon.de/program/NVBLKH/

.. [3] Reshama Shaikh's quote from Melissa Weber Mendonça' keynote:
    https://twitter.com/reshamas/status/1513488342767353857

.. [4] NumPy Newcomer's Hour: an Experiment on Community Building, talk from
    Melissa Weber Mendonça: https://www.youtube.com/watch?v=c0XZQbu0xnw

.. [5] Scikit-learn April 25th 2022 Developer meeting notes:
    https://github.com/scikit-learn/administrative/blob/master/meeting_notes/2022-04-25.md

.. [6] Scikit-learn September 5th 2022 Developer meeting notes:
    https://github.com/scikit-learn/administrative/blob/master/meeting_notes/2022-09-05.md

.. [7] Scikit-learn organisation on GitHub: https://github.com/scikit-learn

.. [8] Scikit-learn documentation, About Us:
    https://scikit-learn.org/stable/about.html

.. [9] Open Publication License: https://www.opencontent.org/openpub/
