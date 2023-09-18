.. _slep_020:

==============================
SLEP Template and Instructions
==============================

:Author: Meekail Zain
:Status: Draft
:Type: Standards Track
:Created: 2023-9-16

Abstract
--------

This SLEP proposes the inclusion of C++ based SIMD acceleration.
The initial scope is limited to accelerating ``DistanceMetric`` implementations,
however there may be other areas which could benefit from similar acceleration.


Detailed description
--------------------

For many estimators (see `Implementation`_), a dominant bottleneck for
computation over many samples is distance computation. Currently, any distance
calculated using the ``DistanceMetric`` back-end is completed through repeated
calls to ``DistanceMetric.dist`` which operates on a pair of sample vectors. These
distance calculations can be understood as a series of parallel arithmetic
operations with some reduction which are currently implemented as scalar loops in
Cython/C. These parallel arithmetic operations occur on data contiguous in memory,
and hence are well-suited to SIMD optimization. For modern hardware, leveraging
SIMD instructions can speed up distance computation by **a factor of 4x-16x**,
depending on data type and specific SIMD architecture. For estimators where distance
computation is a significant bottleneck, such as ``KNeighborsRegressor``, this
translates to a **2.5x-4x speed up**.

While SIMD may be leveraged anywhere that there are data-parallel scalar operations
operating on contiguous memory, this SLEP focuses on the use case of accelerating
``DistanceMetric`` implementations.

SIMD optimization relies in CPU-specific features, which may not be present at runtime
if compiled and run on different machines. This increases the complexity of ensuring
portability of code, however modern libraries such as xsimd (CITE) and highway (CITE)
help mitigate this complexity by providing helpful abstractions with allow for writing
architecture-agnostic implementations.

The adoption of SIMD-accelerated computation would pose no change to user-experience
aside from reduced runtimes. It is important to note that the implementations require
C++ which may introduce some friction in terms of maintainence, however the core
computation implementations are simple, with the majority of required code being
boiler-plate. The only notable change to the build system would be the compilation
and use of a runtime library to isolate the C++ necessary for the SIMD routines. 

This SLEP proposes to use Google Highway

Implementation
--------------

In order to leverage SIMD instructions, even with the aforementioned libraries, a
choice must be made regarding how to dispatch to an appropriate implementation.
Generally, one must use either static or dynamic dispatching.

Static Dispatching
^^^^^^^^^^^^^^^^^^

The simplest implementation of SIMD acceleration is to compile for an explicitly
chosen baseline SIMD architecture for each supported platform (e.g. ``AVX`` on the
``x86_64`` platform) and fallback to scalar loops (i.e. standard Cython/C
implementations). This approach is relatively simple to implement with minimal
overhead, however potentially sacrifices coverage since machines without support
for their platform's corresponding static-dispatch target *must* use the scalar
fallback. Therefore, a tradeoff must be made between using *high-throughput*
architectures (e.g. ``AVX3``), or more *common* architectures (e.g. ``SSE3``).


Dynamic Dispatching
^^^^^^^^^^^^^^^^^^^

A more complicated approach is to compile implementations of architecture-agnostic
code multiple times, targeting different SIMD architectures (e.g. ``SSE3, AVX, AVX2``
on ``x86_64``). Then, at runtime, computational calls are dispatched to the
*best available* SIMD architecture. This runtime dispatch strategy incurs some
overhead, however it is negligible for sufficiently-many features (~>12). For
vectors with too-few features, a simple conditional check on the numver of features
results in implementations with no major regressions. The use of dynamic dispatch
ensures the use of the best-possible runtime architecture while preserving greater
general availability of SIMD acceleration for users with older SIMD architectures.

Note that regardless of the choice of static/dynamic dispatch, highway provides
backup universal scalar implementations as fallback (denoted ``EMU128, SCALAR``)
which can simplify scikit-learn implementations by freeing us from the obligation of
providing our own scalar loops (i.e. our current implementations) *however* it is
observed that it is faster to avoid the highway dispatch mechanism entirely for
sufficiently short vectors, and thus maintaining our current scalar implementations
*in addition to* new SIMD optimized implementations is wortwhile.


Affected Estimators
^^^^^^^^^^^^^^^^^^^

Roughly speaking, any estimator which leverages ``DistanceMetric`` objects for
distance computation will stand to benefit from SIMD acceleration. Those that are
more computation-bound will obtain speedups closer to the ideal 4x-16x. A
non-exhaustive list of such estimators can be found here (CITE gh-26329).

Supported Architectures
^^^^^^^^^^^^^^^^^^^^^^^
Highway supports:

- Arm: NEON (Armv7+), SVE, SVE2, SVE_256, SVE2_128;
- POWER: PPC8 (v2.07), PPC9 (v3.0), PPC10 (v3.1B, not yet supported due to compiler bugs);
- RISC-V: RVV (1.0)
- WebAssembly: WASM, WASM_EMU256
- x86: SSE2, SSSE3, SSE4, AVX2, AVX3, AVX3_DL, AVX3_ZEN4, AVX3_SPR


Backward compatibility
----------------------

No concerns.


Alternatives
------------

- `Experimental C++ SIMD library <https://en.cppreference.com/w/cpp/experimental/simd>`_
   - Experimental
   - No WASM (?)
- `xsimd <https://github.com/xtensor-stack/xsimd>`_
   - Complex dynamic dispatch which requires code generation and using compiler flags
   - Limited supported architectures
   - No WASM
- `libsimdpp <https://github.com/p12tic/libsimdpp>`_
   - Inactive, most recent release Dec 14th, 2017
   - No WASM
- `SIMD everywhere <https://github.com/simd-everywhere/simde>`_
   - No WASM
- `SLEEF <https://github.com/shibatch/sleef>`_
   - Inactive, most recent release Sep 14th, 2020
   - No WASM

Discussion
----------

This section may just be a bullet list including links to any discussions
regarding the SLEP:

- This includes links to mailing list threads or relevant GitHub issues.


References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
