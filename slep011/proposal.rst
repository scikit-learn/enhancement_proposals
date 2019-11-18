.. _slep_011:

======================================================
SLEP011: Add Logical Analysis of Data (LAD) classifier
======================================================

:Author: Gregory Morse (gregory.morse@live.com)
:Status: Active
:Type: Standards Track
:Created: 2019-11-17

Abstract
########

This SLEP proposes the introduction of a logical analysis of data (LAD)
classifier which allows for formation of boolean equations in DNF form
from a data set.

Motivation
##########

Decision trees and K-nearest neighbors are the most similar algorithms but
they achieve different purposes, and sometimes LAD outperforms both of them,
depending on the type of data set.  The literature on the issue has grown in
the past 30 or so years immensely.

There is little in the way of public open source implementations of LAD
despite its usefulness in problem solving.  Needless to say it yields human
understandable results and highly usable results given that boolean equations
representing a lot of human processes, and has been used extensively in
the field of medicine.

Solution
########

The proposed solution is to add the classifier to the build which is already
mostly prepared from the latest research papers on the topic.  430 citations
to the original article generally considered as introducing the topic have
been recorded [2]_.  It will use a fast algorithm which can reduce the
factorial nature of the problem as per [3]_.

The classifier will also contain a component to deal with binarization of
data which can be used independently, and as part of the model fitting to
enable hyper-parameterization.

Considerations
##############

Primarily the testing and maintenance that could come along with such an
addition.  It could increase interest in the project given that it would be
unique and somewhat of a niche.

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

.. [2] An implementation of logical analysis of data
   E. Boros ; P.L. Hammer ; T. Ibaraki ; A. Kogan ; E. Mayoraz ; I. Muchnik

.. _Available: https://ieeexplore.ieee.org/document/842268

.. [3] Accelerated algorithm for pattern detection in logical analysis
   of data
   Sorin Alexe and Peter L.Hammer
   
.. _Available: https://www.sciencedirect.com/science/article/pii/S0166218X05003161


Copyright
---------

This document has been placed in the public domain. [1]_
