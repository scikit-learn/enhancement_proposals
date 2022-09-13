.. _slep_019:

===========================
SLEP019: Governance Update
===========================

:Author: Julien Jerphanion, Gaël Varoquaux
:Status: Draft
:Type: Process
:Created: 2020-09-12

Abstract
--------

This SLEP proposes updating the Governance to broaden the notion of contribution in scikit-learn.


Motivation
----------


Current state
~~~~~~~~~~~~~

So far the formal decision process in the scikit-learn project is anchored on by a subset of contributors, called Core Developers (also refered to as Maintainers).  Their active and consistent contributions are recognized by them:

- being part of scikit-learn organisation on GitHub
- receiving “commit rights” to the repository
- having their review recognised as authoritative
- having vote rights for the project direction (promoting a contributor to be a core-developer, approving a SLEP, etc.)

However, there are a lot of other ways to contribute to the project, whose efforts are actually not recognized. To quote Melissa Weber Mendonça [1]_ and Reshama Shaikh [2]_:

.. epigraph::
  "When some people join an open source project, they may be asked to contribute with tasks that will never get them on a path to any sort of official input, such as voting rights."

Desired goal
~~~~~~~~~~~~

There is a need to:
- value non-coding contributions in the project and acknowledge all efforts (including those that are not quantified by GitHub users' activity)
- empower more contributors in effectively being able to participate in the projects without giving the security responsibilities of tracking code changes to the main branches. These considerations should lead to "contribution paths".

Hence, we propose to explicit contributions and to define more roles and teams.

Limited changes to the governance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This SLEP proposes an incremental change to the governance model. It acknowledges that more could be done, defining more roles and teams. In the spirit of iterations, the idea is to implement a first change to the governance, evolve our community accordingly, and later propose new changes based on the understanding of the resulting community dynamics.

Proposed modifications
----------------------

The proposed modification have been informally discussed in monthly meetings, namely on April 25th 2022 [3]_ and on September 5th 2022 [4]_ meetings. 


New teams
~~~~~~~~~

**"Create "Core Triagers" Team"**

Create a "Core Triagers" team, given commit rights as a way to circumvent limitations in Github's permission system's limitation (such as limited rights for triage such as being able to edit issues' description). People in this team should not push code on upstream nor merge PRs, but they would be trusted and this restriction would be socially enforced without a specific GitHub mechanism.

**"Recurring "Contributors" Team**

Create a "Recurring Contributors" team (as in PyMC) on GitHub, which has no specific GitHub permissions, and in which we easily add people. The goal of this team is to acknowledge people's contributions, by displaying them as part of the [scikit-learn organization on Github](https://github.com/orgs/scikit-learn/) on their GitHub profile, and on the project's website ["About Us" webpage](https://scikit-learn.org/stable/about.html).

A Contributor is added to the "Recurring Contributor" team after championing by three core contributors (notion of "core contributors" as defined below).

**"Create "Experts" Team**

Create a "Experts" team, also without specific GitHub permissions. Core devs can delegate PR approval rights to someone in this team so that the PR can be merged with one approval from a core dev and one approval from the delegated expert.

Beyond Developers: recognize all Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Define "Contributions" explicitly**

Give more emphasis to the notion of contribution at large (beyond code) in our wording. 

**Define of "Core Contributors"**

To give decision power to members of our community responsible for the various aspects of the project, we will list the follow teams as "core" (associated with the notion of "core contributor"):
- Core Triage
- Communication team
- Core devs

**Extend voting rights**

Give voting rights beyond to all member of Core Teams.


Copyright
---------

This document has been placed in the public domain [5]_.

References and Footnotes
------------------------

.. [1] Contributor experience, diversity and culture in Open Source Projects, keynote from Melissa Weber Mendonça: https://2022.pycon.de/program/NVBLKH/

.. [2] Reshama Shaikh's quote from Melissa Weber Mendonça' keynote: https://twitter.com/reshamas/status/1513488342767353857

.. [3] Scikit-learn April 25th 2022 Developer meeting notes: https://github.com/scikit-learn/administrative/blob/master/meeting_notes/2022-04-25.md

.. [4] Scikit-learn September 5th 2022 Developer meeting notes: https://github.com/scikit-learn/administrative/blob/master/meeting_notes/2022-09-05.md

.. [5] Open Publication License: https://www.opencontent.org/openpub/
