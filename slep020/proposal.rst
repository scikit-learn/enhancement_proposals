.. _slep_020:

======================================
SLEP020: Simplifing Governance Changes
======================================

:Author: Thomas J Fan
:Status: Draft
:Type: Process
:Created: 2023-01-09

Abstract
--------

This SLEP proposes to permit governance changes through GitHub Pull Requests,
where a vote will also occur in the Pull Request.

Detailed description
--------------------

Currently, scikit-learn's governance document [2]_ requires an enhancement
proposal to make any changes to the governance document. In this SLEP, we
propose simplifying the process by allowing governance changes through GitHub
Pull Requests. A Pull Request approval will count as a positive vote, and a
"Request Changes" review will count as a negative vote. Once the authors are
happy with the state of the Pull Request, they can call for a vote on the
mailing list. The voting period starts when it's announced on the mailing list.
The voting period will remain one month as stated in the current Governance and
Decision-Making Document [2]_.

Discussion
----------

Members of the scikit-learn community have discussed changing the governance
in PRs:
`enhancement_proposals#74 <https://github.com/scikit-learn/enhancement_proposals/pull/74>`__
and
`enhancement_proposals#81 <https://github.com/scikit-learn/enhancement_proposals/pull/81>`__.
:ref:`SLEP019 <slep_019>` also includes the voting change proposed in this SLEP.
This SLEP's goal is to simplify the process of making governance changes, thus
enabling the governance structure to evolve more efficiently.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open Publication
   License`_.
.. [2] `scikit-learn Governance and Decision-Making
   <https://scikit-learn.org/stable/governance.html#decision-making-process>`__

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
