# Circuit-Level Purification Quantum Error Suppression: Interleaving Under Noisy Gates

**Jonathan Raghoonanan**<sup>1,2</sup> and **Tim Byrnes**<sup>1,2,3,4</sup>

<sup>1</sup> New York University Shanghai; NYU-ECNU Institute of Physics at
NYU Shanghai, 567 West Yangsi Road, Shanghai 200124, China  
<sup>2</sup> Department of Physics, New York University, New York,
New York 10003, USA  
<sup>3</sup> State Key Laboratory of Precision Spectroscopy, School of
Physical and Material Sciences, East China Normal University,
Shanghai 200062, China  
<sup>4</sup> Center for Quantum and Topological Systems (CQTS), NYUAD
Research Institute, New York University Abu Dhabi, UAE

**Corresponding author:** tim.byrnes@nyu.edu

> **Status.** This is an early scientific draft for a possible follow-up
> paper. It intentionally records implementation details, provisional
> numerical results, caveats, and publication-critical TODOs that would
> normally be shortened or moved to supplementary material. Citations use
> provisional BibTeX keys inherited from the first PQES manuscript.

## Abstract

Purification quantum error suppression (PQES) uses SWAP tests on multiple
noisy copies of an unknown state and combines the complete SWAP-outcome record
with parity signs to estimate observables of a power-purified state. In the
ideal setting, common unitary circuit blocks commute through the SWAP
purification tree, suggesting that purification may be applied either at the
end of a circuit or interleaved between circuit blocks. Here we examine whether
this freedom remains advantageous when the workload circuit and the PQES
gadgets themselves are noisy. We formulate an exact branch-aggregated
density-operator simulation that includes imperfect preparation and reset,
one- and two-qubit gate errors, native controlled-SWAP errors, memory noise,
and ancilla readout errors. The simulation retains both the ordinary physical
branch average and the parity-weighted signed operator, thereby representing
all SWAP outcomes without explicitly enumerating an exponentially large branch
record.

We compare two schedules at matched purification depth and input-copy count:
an interleaved schedule applying \(\ell\) PQES levels after each of \(T\)
circuit blocks, and an end-only schedule applying the same total depth
\(D=T\ell\) after the complete circuit. For a representative five-block
brickwork circuit, register sizes \(M=1,2,3\), and local depolarizing or
dephasing errors, neither schedule is uniformly superior. Among 90 nonzero
noise, matched-depth comparisons, interleaving gives the larger final
target-state overlap in 40 cases and end-only purification in 50. After
optimizing over the tested depths \(\ell=1,2,3\), end-only purification gives
the best fidelity in 29 of 30 parameter combinations. Interleaving,
nevertheless, reduces the number of workload-gate locations and the
normalization-induced signed-estimator overhead in every nonzero-noise
comparison. At large purification depth, both schedules acquire prohibitive
copy and sampling costs. These results show that noisy-circuit PQES placement
is not governed by a universal fidelity ordering; it is a hardware- and
circuit-dependent bias--variance--resource optimization problem.

## 1. Introduction

Quantum error correction (QEC) protects quantum information by encoding it
into a larger Hilbert space, extracting syndromes, and applying recovery
operations. Stabilizer codes, particularly surface-code architectures, provide
the leading framework for scalable fault tolerance, but require repeated
syndrome extraction, high-fidelity measurements and resets, classical
decoding, and substantial qubit and gate overhead
\cite{NielsenChuang2010,Gottesman1997Thesis,Fowler_2012}. Quantum error
mitigation (QEM) instead combines noisy experimental data to reduce bias in
estimated observables without producing a fault-tolerant encoded state.
Examples include zero-noise extrapolation, probabilistic error cancellation,
symmetry verification, and purification-based estimators
\cite{Temme2017Mitigation,Cai2023QEM}.

Purification methods use redundancy in a different way. Several noisy copies
of a state are processed to amplify a desired component, historically in
entanglement distillation and quantum communication
\cite{Bennett1996Twirling,Deutsch1996QPA,Dur2007EPReview}. Virtual
distillation and error suppression by derangements apply the same spectral
idea to quantum-computing outputs: powers of a noisy density matrix suppress
its subdominant eigencomponents, and observables of
\(\rho^N/\operatorname{Tr}(\rho^N)\) can be reconstructed from multicopy
measurements \cite{Huggins2021,Koczor_derangement,Koczor2021_dominantEV}.
These protocols are powerful when the ideal target remains the dominant
eigenvector of the noisy state, although their normalization and copy
requirements can cause a severe sampling cost.

In our preceding work, we introduced a branch-resolved SWAP-test
implementation called *purification quantum error suppression* (PQES)
\cite{RaghoonananByrnesPQES}. The SWAP test produces physical symmetric and
antisymmetric conditional branches. Rather than postselecting the symmetric
branch, PQES retains every branch and multiplies its measurement result by the
total SWAP parity. The ordinary branch average describes the physical state
of the surviving register when the outcome is ignored, while the signed
average selects the power operator \(\rho^N\). Thus the protocol has both a
physical branch dynamics and a QEM-like virtual estimator.

The preceding analysis concentrated on ideal PQES operations. It established
the branch identities, binary-tree and register-recycled implementations, and
spectral recoverability thresholds for representative state-level noise
models. For example, the target product state remains recoverable below
\(p_{\mathrm{th}}=3/4\) under the local depolarizing convention used there,
while local phase-flip noise on an equatorial state has a
\(p_{\mathrm{th}}=1/2\) spectral crossing. Those quantities are
ideal-purification or code-capacity-like thresholds: the SWAP tests, resets,
measurements, and circuit gates were not themselves assigned errors.

A notable identity in the ideal theory is that common unitary evolution
commutes through a SWAP layer. This allows a circuit to be distributed through
the purification tree. Operationally, one may therefore purify only the final
output of a circuit, or place purification layers between circuit blocks. In
the absence of hardware noise these layouts give the same ideal
power-purified observable when they represent the same tree. Once gates,
memory, resets, controlled-SWAPs, and measurements are noisy, however, moving
a PQES layer changes both the state entering subsequent noise channels and the
number of live copies exposed to later circuit operations. The ideal
commutation identity no longer implies equal performance.

The central question of this work is therefore:

> **At fixed input-copy count and fixed number of SWAP-tree levels, does
> interleaving noisy PQES layers within a noisy quantum circuit improve the
> final mitigated state relative to applying all purification at the end?**

Answering this requires a circuit-level model that resolves four effects which
are absent from the ideal power map:

1. purification acts on noisy intermediate states whose eigenvectors may not
   coincide with the final ideal target;
2. later workload noise can reintroduce errors after an early purification
   step;
3. the PQES gadget adds its own gate, memory, reset, and readout errors; and
4. the two schedules have different workload-gate exposure even when their
   copy and SWAP-gadget budgets are matched.

We develop an exact density-operator simulation incorporating these effects.
The branch record is aggregated algebraically into unsigned and signed
operators, avoiding explicit enumeration of the \(2^{N-1}\) possible outcome
strings in an \(N\)-copy tree. We then perform a reproducible parameter sweep
for random brickwork circuits and compare fidelity, trace-distance-like error,
purity, physical branch dynamics, gate resources, and signed-estimator
normalization. The present results should be regarded as a controlled
proof-of-principle circuit study rather than a hardware threshold: one
brickwork instance is used, the error channels are Markovian and Pauli based,
and controlled-SWAPs are treated as native three-qubit gates.

The main conclusion is deliberately nuanced. End-only purification is usually
best when the purification depth is optimized solely for final fidelity.
Interleaving can nevertheless produce a sizeable fidelity advantage at fixed
larger depth and systematically reduces both workload-gate exposure and the
signed-estimator normalization cost. The optimal placement is therefore not
universal and cannot be judged from fidelity alone.

## 2. Review of branch-resolved PQES

### 2.1. Ideal SWAP branches

Let \(A\) and \(B\) be \(M\)-qubit registers, and let
\(\operatorname{SWAP}_{AB}\) exchange their states. The SWAP-test outcomes
\(\sigma=\pm1\) correspond to projectors

\[
\Pi_\sigma=\frac{I+\sigma\,\operatorname{SWAP}_{AB}}{2}.
\]

For input operators \(X\) and \(Y\), projecting onto one outcome, discarding
register \(B\), and retaining register \(A\) gives the unnormalized branch
operator

\[
\mathcal{B}_\sigma(X,Y)
=
\operatorname{Tr}_B\!\left[
\Pi_\sigma(X\otimes Y)\Pi_\sigma
\right]
=
\frac{X+Y+\sigma(XY+YX)}{4}.
\tag{1}
\]

If \(X\) and \(Y\) are normalized states, the branch probability is
\(P_\sigma=\operatorname{Tr}\mathcal{B}_\sigma(X,Y)\), and the corresponding
conditional state is
\(\rho_\sigma=\mathcal{B}_\sigma(X,Y)/P_\sigma\). Summing or subtracting the
unnormalized branches gives

\[
\sum_{\sigma=\pm1}\mathcal{B}_\sigma(X,Y)
=\frac{X+Y}{2},
\qquad
\sum_{\sigma=\pm1}\sigma\mathcal{B}_\sigma(X,Y)
=\frac{XY+YX}{2}.
\tag{2}
\]

For identical inputs \(X=Y=\rho\),

\[
\mathcal{B}_+(\rho,\rho)+\mathcal{B}_-(\rho,\rho)=\rho,
\qquad
\mathcal{B}_+(\rho,\rho)-\mathcal{B}_-(\rho,\rho)=\rho^2.
\tag{3}
\]

The first relation states that ignoring the outcome returns the original
average state. The second states that weighting the two outcomes by their
SWAP eigenvalues selects the squared operator.

### 2.2. Binary-tree identity

After \(d\) binary-tree levels, \(N=2^d\) input copies have been consumed and
the tree contains \(N-1\) SWAP outcomes. For an outcome string
\(\boldsymbol{\sigma}\), define its total parity

\[
\Omega_{\boldsymbol{\sigma}}
=\prod_{j=1}^{N-1}\sigma_j.
\]

If \(P_{\boldsymbol{\sigma}}\) and
\(\rho_{\boldsymbol{\sigma}}\) are the probability and normalized surviving
state of that branch, then

\[
\sum_{\boldsymbol{\sigma}}
P_{\boldsymbol{\sigma}}\rho_{\boldsymbol{\sigma}}
=\rho,
\qquad
\sum_{\boldsymbol{\sigma}}
P_{\boldsymbol{\sigma}}\Omega_{\boldsymbol{\sigma}}
\rho_{\boldsymbol{\sigma}}
=\rho^N.
\tag{4}
\]

The normalized power state is

\[
\mathcal{P}_d(\rho)
=\frac{\rho^N}{\operatorname{Tr}(\rho^N)},
\qquad N=2^d.
\tag{5}
\]

If
\(\rho=\sum_i\lambda_i\lvert\lambda_i\rangle\langle\lambda_i\rvert\),
then Eq. (5) raises every eigenvalue to the \(N\)th power. It amplifies the
dominant eigenspace but does not rotate the eigenbasis. Consequently, PQES
cannot asymptotically recover a pure target to unit fidelity if that target
is not the unique dominant eigenvector of the noisy state.

### 2.3. Observable estimator and normalization cost

For an observable \(O\), the power-state expectation is

\[
\langle O\rangle_N
=
\frac{
\sum_{\boldsymbol{\sigma}}
P_{\boldsymbol{\sigma}}\Omega_{\boldsymbol{\sigma}}
\operatorname{Tr}(O\rho_{\boldsymbol{\sigma}})
}{
\sum_{\boldsymbol{\sigma}}
P_{\boldsymbol{\sigma}}\Omega_{\boldsymbol{\sigma}}
}
=
\frac{\operatorname{Tr}(O\rho^N)}
{\operatorname{Tr}(\rho^N)}.
\tag{6}
\]

The denominator is the signed parity expectation. For bounded observables,
the large-sample variance of the ratio estimator contains a leading factor
\(1/\lvert\operatorname{Tr}(\rho^N)\rvert^2\). In this paper we report

\[
\mathcal{O}_{\mathrm{norm}}
=
\frac{1}{
\left\lvert\operatorname{Tr}S\right\rvert^2
}
\tag{7}
\]

for the corresponding noisy signed operator \(S\). This is a
normalization-induced overhead proxy, not the complete observable-dependent
shot variance. A physical resource estimate must also multiply the number of
shots by the \(N\) input copies and the circuit locations required per shot.

### 2.4. Ideal interleaving covariance

A common unitary \(V\) commutes with the two-register SWAP:

\[
[V\otimes V,\operatorname{SWAP}_{AB}]=0.
\]

It follows that

\[
\mathcal{B}_\sigma(VXV^\dagger,VYV^\dagger)
=
V\mathcal{B}_\sigma(X,Y)V^\dagger,
\tag{8}
\]

and hence

\[
\mathcal{P}_d(V\rho V^\dagger)
=
V\mathcal{P}_d(\rho)V^\dagger.
\tag{9}
\]

Equation (9) permits an ideal circuit block to be moved through an ideal PQES
tree. This is an exact unitary identity, but it does **not** hold after simply
replacing \(V(\cdot)V^\dagger\) with a generic noisy channel. In particular,
for a channel \(\mathcal{E}\),

\[
\mathcal{P}_d[\mathcal{E}(\rho)]
\ne
\mathcal{E}[\mathcal{P}_d(\rho)]
\tag{10}
\]

in general. The inequality in Eq. (10) is the source of the schedule
dependence studied below.

### 2.5. Prior resource scaling and state-level thresholds

The ideal PQES tree has two complementary implementations
\cite{RaghoonananByrnesPQES}. A fully parallel
depth-\(d\) tree stores all \(N=2^d\) input registers and therefore uses
\(O(MN)\) data qubits. Its \(N-1\) register-level SWAP tests are arranged in
\(d\) layers, giving depth \(O(d\,d_{\mathrm{SWAP}})\), where
\(d_{\mathrm{SWAP}}\) is the architecture-dependent depth of one
register-SWAP test. A fully recycled implementation generates the same tree
sequentially with \(d+1\) live registers, or \(O(Md)=O(M\log N)\) data
qubits, but its sequential SWAP-test depth grows as
\(O(N\,d_{\mathrm{SWAP}})\). Intermediate width--depth tradeoffs are also
possible. The present simulator evaluates the tree identities without
choosing a concrete parallel or recycled wall-clock schedule; the resource
tables count the full set of physical locations required to produce one
estimator sample.

The ideal state-level analysis is governed by the spectral ratio
\(\lambda_2/\lambda_1\). If the target is the unique dominant eigenvector,
the unwanted spectral weight is suppressed exponentially in \(N\), with a
representative bound proportional to
\((\lambda_2/\lambda_1)^N\). If the largest eigenvalue is degenerate, the
normalized power map converges to the corresponding dominant subspace; if a
different eigenvector is dominant, increasing \(N\) amplifies the wrong
component.

This criterion gave the following recoverability boundaries in the preceding
work:

- \(p_{\mathrm{th}}=1\) for global depolarization
  \((1-p)\rho+pI/D\);
- \(p_{\mathrm{th}}=3/4\) for independent local depolarizing errors on the
  product-state family considered there;
- \(p_{\mathrm{th}}=1/2\) for local phase-flip noise acting on
  \(\lvert+\rangle^{\otimes M}\); and
- restoration of the local-depolarizing \(3/4\) boundary after full local
  Clifford twirling of the phase-flip channel.

These are ideal-recovery, state-channel thresholds. They assume perfect PQES
gadgets and do not include circuit location counts, parity decay, or finite
sampling. One purpose of the present paper is to prevent those state-level
boundaries from being conflated with circuit-level hardware thresholds.

### 2.6. Motivating matched-budget dephasing example

Before introducing gate noise, it is useful to record the state-only
calculation which motivated the present study. Consider one qubit initialized
in \(\lvert+\rangle\), followed by \(T=5\) identical phase-flip channels and
five ideal power-map levels. The schedules

\[
\mathcal{P}^{\,T}\mathcal{E}^{\,T}(\rho_0)
\quad\text{and}\quad
(\mathcal{P}\mathcal{E})^{T}(\rho_0)
\]

represent end-only and interleaved purification, respectively. Both consume
\(N=2^T=32\) leaf copies and use \(N-1=31\) pairwise merges.

The ordering can be understood analytically. Write the state as

\[
\rho(r)=\frac{I+rX}{2}.
\]

The phase-flip channel contracts the signed \(X\)-component according to

\[
r\longmapsto q r,\qquad q=1-2p,
\]

whereas one ideal purification level gives

\[
r\longmapsto f(r)=\frac{2r}{1+r^2}
=\tanh\!\left[2\operatorname{atanh}(r)\right].
\]

The two final Bloch components are therefore

\[
r_{\mathrm{end}}
=
\tanh\!\left[
2^T\operatorname{atanh}(q^T)
\right],
\qquad
r_t^{\mathrm{int}}=f(qr_{t-1}^{\mathrm{int}}),
\quad r_0^{\mathrm{int}}=1,
\]

and the target overlap is \(F=(1+r)/2\). For \(0<q<1\),

\[
f(qr)-qf(r)
=
\frac{2qr^3(1-q^2)}
{(1+q^2r^2)(1+r^2)}
\ge0
\qquad (0\le r\le1).
\]

Thus, swapping a purification step to occur *after* a later dephasing step
cannot decrease \(r\). Repeatedly applying this adjacent-swap argument moves
all ideal purification levels to the end and explains why end-only
purification is at least as good as interleaving in this special commuting,
below-threshold model. This is a property of the particular phase-flip and
power maps, not a general scheduling theorem.

The resulting trajectories are shown in
Fig. \(\ref{fig:state_only_motivation}\). At \(p=0.3\), the
no-purification, interleaved, and end-only final overlaps are \(0.505120\),
\(0.621080\), and \(0.658222\), respectively. End-only purification is
higher than interleaving by \(0.037143\), despite the equal copy and merge
budgets. At \(p=0.6\), \(q<0\): the target
\(\lvert+\rangle\) is no longer the dominant eigenvector after a single
noise application, purification preserves the sign of \(r\) and amplifies
its magnitude, and the repeated noise produces sign oscillations. The final
overlaps are \(0.499840\) without purification, \(0.495111\) with
interleaving, and \(0.494880\) with end-only purification. Interleaving is
marginally less damaging in this example, but neither purified schedule
recovers the target.

\begin{figure*}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/rho2_interleaving/matched_budget_M1_plus_p030_T5.pdf}
\hfill
\includegraphics[width=0.48\textwidth]{figures/rho2_interleaving/matched_budget_M1_plus_p060_T5.pdf}
\caption{Motivating state-only matched-budget comparison for
\(M=1\), \(\lvert+\rangle\), and \(T=5\). Both purified schedules use five
ideal power-map levels, \(N=32\) leaf copies, and 31 pairwise merges. Left:
\(p=0.3\), below the phase-flip spectral crossing, where end-only
purification gives the higher final overlap. Right: \(p=0.6\), above the
crossing, where neither schedule recovers the target and interleaving is only
marginally less damaging.}
\label{fig:state_only_motivation}
\end{figure*}

\begin{figure*}[t]
\centering
\fbox{\parbox{0.94\textwidth}{
\textbf{TODO schematic.} Draw two matched-depth schedules for \(T\) circuit
blocks and \(\ell\) PQES levels per block. The interleaved circuit alternates
\(\mathcal{C}_t\) and \(\mathcal{P}_{\ell}\), reducing the number of live
registers after every cycle. The end-only circuit applies all \(T\) blocks to
all \(N=2^{T\ell}\) copies and then applies
\(\mathcal{P}_{T\ell}\). Annotate the common leaf-copy count \(N\), common
\(N-1\) SWAP gadgets, and unequal workload-gate counts.}}
\caption{Proposed overview of the matched-copy interleaved and end-only
schedules. This figure is essential because “matched budget” does not mean
that every physical resource is equal.}
\label{fig:schedule_schematic}
\end{figure*}

## 3. Circuit-level noisy PQES

### 3.1. Unsigned and signed branch aggregates

Explicitly storing all SWAP records is unnecessary. Let \(U\) be the ordinary
branch average and \(S\) the parity-weighted branch aggregate,

\[
U=\sum_r P_r\rho_r,
\qquad
S=\sum_r P_r\Omega_r\rho_r.
\tag{11}
\]

Here \(r\) labels the complete record accumulated in a subtree. \(U\) is a
physical density operator when the record is ignored. \(S\) is a Hermitian
signed operator used to evaluate mitigated observables; it is not assumed to
be a deterministically prepared physical state.

Let \(\Phi_\sigma\) denote the trace-nonincreasing operation describing the
entire noisy SWAP gadget with true ancilla outcome \(\sigma\). It includes the
ancilla preparation, the two noisy Hadamards, the controlled-SWAP sequence,
memory noise, measurement projection, discarding the ancilla and register
\(B\), and retention of register \(A\). Because the physical circuit is
linear in its joint input,

\[
U'
=
\Phi_+(U\otimes U)+\Phi_-(U\otimes U).
\tag{12}
\]

Suppose the recorded ancilla bit is flipped independently with probability
\(p_{\mathrm{m}}\). The expected recorded sign is reduced by the visibility

\[
v_{\mathrm{m}}=1-2p_{\mathrm{m}}.
\tag{13}
\]

The signed aggregate therefore obeys

\[
S'
=
v_{\mathrm{m}}\left[
\Phi_+(S\otimes S)-\Phi_-(S\otimes S)
\right].
\tag{14}
\]

Equations (12) and (14) exactly sum the exponentially many records for the
symmetric binary trees considered here. In the ideal limit,
\(\Phi_\sigma=\mathcal{B}_\sigma\), so \(U'=\rho\) and \(S'=\rho^2\) for
identical normalized inputs.

A noisy workload block is represented by a linear channel
\(\mathcal{C}_t\). It acts on both aggregates,

\[
U\longmapsto\mathcal{C}_t(U),
\qquad
S\longmapsto\mathcal{C}_t(S).
\tag{15}
\]

Although \(S\) can be unnormalized or non-positive, its linear evolution
through the physical channel remains well defined.

### 3.2. Workload circuits and ideal target

Each experiment begins with \(M\) imperfectly reset data qubits. Explicit
Hadamard gates prepare the intended initial target

\[
\lvert\psi_0\rangle=\lvert+\rangle^{\otimes M}.
\tag{16}
\]

The main dataset uses a reproducible hardware-efficient brickwork circuit
divided into \(T=5\) blocks. In block \(t\), each qubit \(j\) receives

\[
R_y(\theta_{t,j})\quad\text{followed by}\quad R_z(\phi_{t,j}),
\tag{17}
\]

where \(\theta_{t,j}\) and \(\phi_{t,j}\) are independently drawn from
\([-\pi,\pi]\) using NumPy's default random generator with seed 7. A CNOT
brickwork layer follows the rotations. For zero-indexed blocks, even \(t\)
uses directed pairs

\[
(0,1),(2,3),\ldots,
\]

whereas odd \(t\) uses

\[
(1,2),(3,4),\ldots.
\]

The implementation also supports an `echo` circuit consisting of cancelling
\(R_x\) and CNOT pairs and an `idle` memory-only control, but those circuit
families are not included in the frozen dataset analyzed in Sec. 4.

The reference target is evolved using the corresponding ideal circuit:

\[
\lvert\psi_t\rangle
=V_tV_{t-1}\cdots V_1\lvert+\rangle^{\otimes M}.
\tag{18}
\]

Purification leaves this target unchanged because it remains pure. Every
reported fidelity is evaluated against \(\lvert\psi_t\rangle\), not against
the initial \(\lvert+\rangle^{\otimes M}\) after the circuit has begun.

### 3.3. Noise channels

Six independently configurable location error probabilities are used:

\[
\boldsymbol{p}
=
(p_{\mathrm{1q}},p_{\mathrm{2q}},p_{\mathrm{cswap}},
p_{\mathrm{mem}},p_{\mathrm{m}},p_{\mathrm{r}}).
\tag{19}
\]

The default sweep sets every component equal to a common value \(p\). This
uniform assignment is a controlled baseline, not a claim about realistic
hardware. The command-line interface also permits independent multiplicative
weights for each location type.

Two channel families are considered.

**Depolarizing family.** After an ideal gate acting on an \(n\)-qubit support
\(Q\), the simulator applies the Qiskit Aer depolarizing convention

\[
\mathcal{D}^{(n)}_p(X)
=
(1-p)X
+p\,\frac{I_Q}{2^n}\otimes\operatorname{Tr}_Q(X),
\tag{20}
\]

with the identity understood on the unaffected complement. Thus a CNOT
receives a joint two-qubit depolarizing channel and a native Fredkin gate
receives a joint three-qubit depolarizing channel.

**Dephasing family.** Each qubit in the support of an ideal gate independently
receives a phase-flip channel,

\[
\mathcal{Z}^{(q)}_p(X)
=(1-p)X+pZ_qXZ_q.
\tag{21}
\]

For a multiqubit gate, Eq. (21) is composed once for each qubit in its support.
The depolarizing and dephasing values of \(p\) therefore parameterize
different microscopic error distributions and should not be interpreted as
identical total Pauli-error probabilities.

The imperfect reset state of every data or ancilla qubit is

\[
\tau_{\mathrm{r}}
=(1-p_{\mathrm{r}})\lvert0\rangle\langle0\rvert
+p_{\mathrm{r}}\lvert1\rangle\langle1\rvert.
\tag{22}
\]

Noise is inserted at the following locations:

1. after each data-preparation Hadamard, \(R_y\), or \(R_z\), using
   \(p_{\mathrm{1q}}\);
2. after each workload CNOT, using \(p_{\mathrm{2q}}\);
3. after each native three-qubit Fredkin, using \(p_{\mathrm{cswap}}\);
4. once on every data qubit after every workload block, using
   \(p_{\mathrm{mem}}\);
5. once on all \(1+2M\) live qubits after every Fredkin in a SWAP gadget,
   using \(p_{\mathrm{mem}}\);
6. in every data and ancilla reset through Eq. (22); and
7. as a symmetric classical ancilla-bit flip after the true SWAP outcome,
   using \(p_{\mathrm{m}}\).

All channels are Markovian and gate local. The present model does not include
amplitude damping, leakage, coherent calibration errors, crosstalk,
non-Markovian noise, correlated reset or readout errors, or a hardware
connectivity graph.

### 3.4. Noisy SWAP gadget

The SWAP ancilla, register \(A\), and register \(B\) occupy \(1+2M\) qubits.
One gadget performs:

1. imperfect ancilla reset to \(\tau_{\mathrm{r}}\);
2. an ideal ancilla Hadamard followed by one-qubit noise;
3. \(M\) serial native Fredkin gates, one on
   \((a,A_j,B_j)\);
4. Fredkin-support noise after each Fredkin;
5. one memory interval on all \(1+2M\) live qubits after each Fredkin;
6. a final ideal ancilla Hadamard followed by one-qubit noise;
7. computational-basis ancilla projection;
8. tracing out the ancilla and register \(B\); and
9. a classical symmetric readout flip in the sign record.

The physical branch maps \(\Phi_\pm\) use the *true* pre-readout projections.
Readout noise changes the signed estimator through Eq. (14) but does not
change \(U\). Diagnostic branch probabilities saved in the dataset are
therefore true pre-readout probabilities.

The assumption of a native Fredkin is consequential. A fault-tolerant or
device-specific implementation would decompose the Fredkin into the native
basis, introducing more locations, connectivity constraints, and possibly
coherent error propagation. The present results are consequently a
native-three-qubit-gate benchmark rather than a prediction for a particular
processor.

### 3.5. Purification schedules

Let \(T\) be the number of workload blocks and \(\ell\) the number of binary
PQES levels assigned per block. Both purified schedules use total depth

\[
D=T\ell,
\tag{23}
\]

leaf-copy count

\[
N=2^D=2^{T\ell},
\tag{24}
\]

and number of register-level SWAP gadgets

\[
G=N-1.
\tag{25}
\]

The schedules are:

**Interleaved**

\[
\rho_0
\xrightarrow{\mathcal{C}_1}
\xrightarrow{\mathcal{P}_{\ell}}
\xrightarrow{\mathcal{C}_2}
\xrightarrow{\mathcal{P}_{\ell}}
\cdots
\xrightarrow{\mathcal{C}_T}
\xrightarrow{\mathcal{P}_{\ell}}.
\tag{26}
\]

**End-only**

\[
\rho_0
\xrightarrow{\mathcal{C}_1}
\xrightarrow{\mathcal{C}_2}
\cdots
\xrightarrow{\mathcal{C}_T}
\xrightarrow{\mathcal{P}_{T\ell}}.
\tag{27}
\]

**No-PQES baseline**

\[
\rho_0
\xrightarrow{\mathcal{C}_1}
\xrightarrow{\mathcal{C}_2}
\cdots
\xrightarrow{\mathcal{C}_T}.
\tag{28}
\]

For \(T=5\), the tested copy budgets are

| \(\ell\) | Total depth \(D=T\ell\) | Leaf copies \(N=2^D\) | SWAP gadgets \(N-1\) |
|---:|---:|---:|---:|
| 0 | 0 | 1 | 0 |
| 1 | 5 | 32 | 31 |
| 2 | 10 | 1,024 | 1,023 |
| 3 | 15 | 32,768 | 32,767 |

The comparison is therefore matched in copies, tree depth, controlled-SWAP
count, ancilla measurements, and PQES resets. It is **not** matched in
workload-gate count or workload memory exposure.

### 3.6. Resource accounting and the meaning of a matched budget

Let \(g_1(t)\) and \(g_2(t)\) be the one- and two-qubit workload-gate counts in
block \(t\), with \(t=0,\ldots,T-1\), and let \(D=T\ell\). The number of live
registers entering interleaved block \(t\) is

\[
K_t^{\mathrm{int}}=2^{D-t\ell},
\tag{29}
\]

whereas every end-only block acts on all \(N=2^D\) copies:

\[
K_t^{\mathrm{end}}=N.
\tag{30}
\]

Consequently,

\[
G_{1,\mathrm{work}}^{\mathrm{int}}
=\sum_{t=0}^{T-1}K_t^{\mathrm{int}}g_1(t),
\qquad
G_{1,\mathrm{work}}^{\mathrm{end}}
=N\sum_{t=0}^{T-1}g_1(t),
\tag{31}
\]

and analogously for \(G_{2,\mathrm{work}}\). Including preparation and the two
ancilla Hadamards per SWAP gadget,

\[
G_1
=MN+G_{1,\mathrm{work}}+2(N-1),
\tag{32}
\]

while

\[
G_2=G_{2,\mathrm{work}},
\qquad
G_{\mathrm{cswap}}=M(N-1),
\qquad
G_{\mathrm{meas}}=N-1,
\qquad
G_{\mathrm{reset}}=MN+(N-1).
\tag{33}
\]

The SWAP-gadget memory count is

\[
G_{\mathrm{mem,swap}}
=M(1+2M)(N-1),
\tag{34}
\]

because each of the \(M\) serial Fredkins is followed by a memory interval on
all \(1+2M\) live qubits. Workload-block memory contributes

\[
G_{\mathrm{mem,work}}^{\mathrm{int}}
=M\sum_{t=0}^{T-1}K_t^{\mathrm{int}},
\qquad
G_{\mathrm{mem,work}}^{\mathrm{end}}
=MNT.
\tag{35}
\]

Thus the physically complete memory count is the sum of Eqs. (34) and (35).

> **Publication-critical accounting TODO.** The current
> `physical_memory_locations` column in `finals.csv` contains Eq. (34) only,
> even though workload memory in Eq. (35) is applied during state evolution.
> Correct that exported resource column and regenerate the comparison table
> before using memory counts in a submitted figure or table. This bookkeeping
> omission does not alter any simulated state or fidelity.

The resource asymmetry is not necessarily an unfair artifact: it is one of
the operational motivations for interleaving. Early merges allow later
workload blocks to act on fewer registers. Nevertheless, claims must identify
the matched resource explicitly. A future analysis should separately compare
(i) matched input copies and SWAP gadgets, as here; (ii) matched total
gate locations; and (iii) matched wall-clock depth or spacetime volume.

### 3.7. Performance metrics

At cycle \(t\), let \(\lvert\psi_t\rangle\) be the ideal target, and define
normalized aggregate operators

\[
\bar U_t=\frac{U_t}{\operatorname{Tr}U_t},
\qquad
\bar S_t=\frac{S_t}{\operatorname{Tr}S_t}.
\tag{36}
\]

The primary mitigated target overlap is

\[
F_S(t)
=
\langle\psi_t\rvert\bar S_t\lvert\psi_t\rangle,
\tag{37}
\]

and the physical branch-average fidelity is

\[
F_U(t)
=
\langle\psi_t\rvert\bar U_t\lvert\psi_t\rangle.
\tag{38}
\]

We call Eq. (37) a *signed-estimator fidelity*. If \(\bar S_t\) is not
positive semidefinite, it is not an Uhlmann fidelity between two physical
states; it remains a well-defined target overlap of the mitigated estimator.
The current dataset happens to keep the reported overlaps and purities within
their ordinary physical ranges, but positivity is not assumed by the method.

The code also reports

\[
\epsilon_L(t)
=
\frac12\left\|
\bar S_t-\lvert\psi_t\rangle\langle\psi_t\rvert
\right\|_1,
\qquad
\mathcal{Q}_S(t)=\operatorname{Tr}(\bar S_t^2),
\tag{39}
\]

with corresponding physical quantities obtained from \(\bar U_t\). For
\(M=1\), the Bloch-vector radius of \(\bar S_t\) is saved. The plotted
iteration-one quantity

\[
\gamma_1=1-F_S(t=1)
\tag{40}
\]

is an initial infidelity diagnostic. It should not be described as a
fault-tolerant logical error rate without additional justification.

The signed normalization and its overhead proxy are

\[
\nu_t=\operatorname{Tr}S_t,
\qquad
\log_{10}\mathcal{O}_{\mathrm{norm}}
=-2\log_{10}\lvert\nu_t\rvert.
\tag{41}
\]

`P_success=1` in the legacy-compatible CSV schema means that no branch is
postselected; it is not the probability of a special SWAP outcome.

### 3.8. Exact numerical method and log-space stabilization

All reported states are computed by deterministic dense density-operator
evolution. There is no trajectory sampling and no finite-shot noise in the
present results. One noisy SWAP instrument acts on \(1+2M\) qubits, so its
dense operator dimension grows as \(2^{1+2M}\). Grid runs are restricted to
\(M\le4\), and the frozen sweep stops at \(M=3\).

The magnitude of \(S\) can become smaller than floating-point range at large
tree depth even while \(\bar S\) remains well defined. After each merge the
implementation therefore stores

\[
S=e^a\widehat S,
\qquad
\|\widehat S\|_F=1,
\tag{42}
\]

and updates

\[
a'
=2a+\log\left\|
v_{\mathrm m}
\left[\Phi_+(\widehat S\otimes\widehat S)
-\Phi_-(\widehat S\otimes\widehat S)\right]
\right\|_F.
\tag{43}
\]

The normalized estimator depends only on \(\widehat S/\operatorname{Tr}
\widehat S\), while

\[
\log|\operatorname{Tr}S|
=a+\log|\operatorname{Tr}\widehat S|.
\tag{44}
\]

Equations (42)--(44) permit stable evolution through \(D=15\), where the
ordinary floating-point value of the overhead frequently overflows.

### 3.9. Frozen parameter sweep

The dataset used for the provisional results below is:

| Parameter | Values or convention |
|---|---|
| Workload | random \(R_y\)-\(R_z\)-CNOT brickwork |
| Random seed | 7 |
| Rotation scale | 1 |
| Initial target | \(\lvert+\rangle^{\otimes M}\) |
| Register sizes | \(M=1,2,3\) |
| Circuit blocks | \(T=5\) |
| Noise families | depolarizing, dephasing |
| Common location error \(p\) | \(0,0.001,0.003,0.01,0.03,0.1\) |
| Purification levels per block | \(\ell=0,1,2,3\) |
| Purified schedules | interleaved, end-only |
| Default rate assignment | \(p_{\mathrm{1q}}=p_{\mathrm{2q}}=p_{\mathrm{cswap}}=p_{\mathrm{mem}}=p_{\mathrm m}=p_{\mathrm r}=p\) |
| Numerical method | exact dense density operators and analytic branch aggregation |

For every tuple \((\text{noise},M,p)\), one no-PQES baseline and both purified
schedules for each nonzero \(\ell\) are run. This gives

\[
2\times3\times6\times(1+2\times3)=252
\tag{45}
\]

experiments, 1,260 saved cycle rows, 108 matched schedule pairs including
zero noise, and 90 nonzero-noise matched pairs.

## 4. Numerical results

### 4.1. Validation and internal consistency

The implementation was checked against identities which do not depend on the
particular circuit sweep:

1. for an ideal SWAP gadget and a nontrivial mixed single-qubit state, the
   branch aggregates satisfy \(U'=\rho\) and \(S'=\rho^2\) to an absolute
   tolerance of \(10^{-12}\);
2. a readout-flip probability \(p_{\mathrm m}=0.1\) multiplies \(S'\) by
   \(0.8\) while leaving \(U'\) and the normalized \(\bar S'\) unchanged when
   no other error is present;
3. every schedule gives unit target overlap in the zero-noise limit for
   nontrivial one- and two-qubit circuits;
4. the resource identities \(N=2^D\) and \(G=N-1\) hold for every run;
5. a stress test with \(M=1\), \(T=5\), \(\ell=3\), and \(p=0.1\) retains
   finite normalized metrics using the log-space representation; and
6. the full sweep completed without a failed experiment or a missing
   fidelity value.

Across the 42 zero-noise final rows, the maximum numerical deviation of the
signed-estimator fidelity from unity is
\(7.8\times10^{-16}\). Eight focused model tests pass in the repository
snapshot used for this draft.

### 4.2. First-cycle purification behavior

Figure \(\ref{fig:ell_scan_dephasing}\) and
Fig. \(\ref{fig:ell_scan_depolarizing}\) show the target overlap after the
first circuit block. At this time, the end-only schedule has not yet applied a
PQES layer, so its values are independent of \(\ell\) and coincide with the
no-PQES state. The interleaved curves instead show the immediate effect of
\(\ell\) noisy PQES levels. This distinction is useful for diagnosing whether
a noisy purification gadget is locally beneficial before later circuit blocks
act.

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/noisy_circuit_simulations/fidelity_vs_purification_rounds_iter1_dephasing.pdf}
\caption{Signed-estimator target overlap after the first noisy brickwork
block under dephasing-family location noise. Solid curves denote interleaved
PQES; dashed curves denote the end-only schedule, which has not yet purified
at \(t=1\). The panels use \(p=0,0.003,0.01,0.1\), and include
\(M=1,2,3\).}
\label{fig:ell_scan_dephasing}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/noisy_circuit_simulations/fidelity_vs_purification_rounds_iter1_depolarizing.pdf}
\caption{The same first-cycle diagnostic as
Fig. \(\ref{fig:ell_scan_dephasing}\), for depolarizing-family location
noise. At large \(p\) and \(M\), a small number of interleaved levels can
improve the immediate target overlap, whereas an excessively noisy PQES layer
can also reduce it.}
\label{fig:ell_scan_depolarizing}
\end{figure*}

At low error rates, the immediate improvement from extra purification levels
is small because the state is already close to pure. At larger \(p\), the
interleaved response becomes strongly dependent on \(M\), the channel family,
and \(\ell\). These first-cycle curves alone do not predict the final ordering:
subsequent noisy circuit blocks act after an early purification step, whereas
the end-only schedule applies its full tree directly before final evaluation.

### 4.3. Fidelity trajectories

Figures \(\ref{fig:trajectory_dephasing}\) and
\(\ref{fig:trajectory_depolarizing}\) show complete five-cycle trajectories
for \(M=3\). End-only curves follow the no-PQES trajectory through \(t=4\) and
separate only when the full depth-\(D\) tree is applied after block 5.
Interleaved curves are modified after every block. The lower panels plot the
first-cycle infidelity \(\gamma_1=1-F_S(1)\) for \(M=1\) and \(M=3\).

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/noisy_circuit_simulations/combined_fidelity_gamma_dephasing.pdf}
\caption{Top: signed-estimator target overlap through the five-block
brickwork circuit for \(M=3\) and representative dephasing rates
\(p=0.001\) and \(p=0.1\). Bottom: first-cycle infidelity
\(\gamma_1=1-F_S(1)\) for \(M=1\) and \(M=3\). This is an initial
infidelity diagnostic rather than a fault-tolerant logical-error rate.}
\label{fig:trajectory_dephasing}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/noisy_circuit_simulations/combined_fidelity_gamma_depolarizing.pdf}
\caption{The trajectory and first-cycle diagnostics corresponding to
Fig. \(\ref{fig:trajectory_dephasing}\), for depolarizing-family gate and
memory noise.}
\label{fig:trajectory_depolarizing}
\end{figure*}

The trajectories illustrate why a final overlap cannot be inferred from a
single local purification gain. An interleaved layer may improve \(F_S(t)\)
immediately, but later noisy blocks can undo part of that gain or change the
dominant eigendirection. Conversely, an intermediate state with modest
overlap can sometimes be steered by later ideal circuit dynamics into a state
for which the signed estimator has a larger final target overlap.

### 4.4. Final target overlap

Figure \(\ref{fig:final_fidelity}\) compares final fidelities with \(M=1\) and
\(M=3\). For all 30 nonzero-noise tuples
\((\text{noise},M,p)\), at least one purified configuration in the tested set
has a higher signed-estimator fidelity than the no-PQES baseline. This
statement optimizes over both schedule and \(\ell\) and does not include the
sampling or copy cost.

\begin{figure*}[t]
\centering
\includegraphics[width=0.94\textwidth]{figures/noisy_circuit_simulations/final_fidelity_vs_p.pdf}
\caption{Final signed-estimator target overlap versus the common location
error \(p\). The gray dotted curve is the no-PQES baseline. Solid and dashed
curves show interleaved and end-only schedules at
\(\ell=1,2,3\). Only \(M=1\) and \(M=3\) are displayed for readability;
the matched-comparison heatmap in Fig. \(\ref{fig:advantage}\) includes
\(M=2\).}
\label{fig:final_fidelity}
\end{figure*}

Optimizing only the final fidelity over \(\ell=1,2,3\) gives a pronounced
preference for end-only purification:

- end-only gives the best purified fidelity in 29 of the 30 nonzero-noise
  parameter combinations;
- interleaving gives the best value only for dephasing with
  \(M=2\), \(p=0.1\), and \(\ell=3\), where
  \(F_S=0.274710\) relative to the no-PQES value \(0.265962\);
- the best tested purified configuration improves on the baseline in all
  30 combinations, but the improvement can be very small and can require an
  impractical sampling cost.

For depolarizing noise, the largest optimized gain over the no-PQES baseline
occurs at \(M=3,p=0.03\): end-only \(\ell=1\) gives
\(F_S=0.866479\), compared with \(0.365024\) without PQES. For dephasing,
the largest optimized gain occurs at \(M=2,p=0.03\): end-only \(\ell=1\)
gives \(0.879991\), compared with \(0.491440\).

### 4.5. Direct matched-depth comparison

Define the interleaving advantage

\[
\Delta F
=F_S^{\mathrm{int}}(T)-F_S^{\mathrm{end}}(T).
\tag{46}
\]

Positive values favor interleaving at fixed
\((\text{noise},M,p,\ell,T)\), while negative values favor end-only
purification. Figure \(\ref{fig:advantage}\) displays all 108 matched pairs.
The 18 \(p=0\) entries agree to numerical precision. Of the 90 nonzero-noise
pairs, 40 have \(\Delta F>10^{-12}\) and 50 have
\(\Delta F<-10^{-12}\).

\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{figures/noisy_circuit_simulations/interleaving_advantage.pdf}
\caption{Final matched-depth interleaving advantage
\(\Delta F=F_S^{\mathrm{int}}-F_S^{\mathrm{end}}\).
Blue cells favor interleaving and red cells favor end-only purification.
Every cell is annotated with \(\Delta F\). The comparison fixes \(T=5\),
\(\ell\), \(N=2^{T\ell}\), and the number of SWAP gadgets, but does not
fix the total workload-gate count.}
\label{fig:advantage}
\end{figure*}

The win counts and mean differences for the nonzero-noise data are:

| Noise | \(M\) | Matched pairs | Interleaved wins | End-only wins | Mean \(\Delta F\) | Range of \(\Delta F\) |
|---|---:|---:|---:|---:|---:|---:|
| Dephasing | 1 | 15 | 8 | 7 | \(-0.016148\) | \([-0.189115,\,0.081016]\) |
| Dephasing | 2 | 15 | 9 | 6 | \(-0.011286\) | \([-0.190274,\,0.126782]\) |
| Dephasing | 3 | 15 | 8 | 7 | \(+0.023519\) | \([-0.044787,\,0.198652]\) |
| Depolarizing | 1 | 15 | 0 | 15 | \(-0.004743\) | \([-0.060030,\,0]\) |
| Depolarizing | 2 | 15 | 7 | 8 | \(-0.013736\) | \([-0.095062,\,0.007980]\) |
| Depolarizing | 3 | 15 | 8 | 7 | \(-0.016999\) | \([-0.162536,\,0.056335]\) |

These aggregate counts conceal a strong depth dependence. For dephasing,
interleaving wins only 1 of 15 \(\ell=1\) comparisons, but wins 12 of 15 at
each of \(\ell=2\) and \(\ell=3\). For depolarizing noise, end-only wins all
15 \(\ell=1\) comparisons; the counts become nearly balanced at
\(\ell=2\) and \(\ell=3\). At \(p=0.1\), end-only wins all nine
depolarizing comparisons.

The largest positive difference is

\[
\Delta F=+0.198652
\]

for dephasing with \(M=3,p=0.03,\ell=3\), where
\(F_S^{\mathrm{int}}=0.483771\) and
\(F_S^{\mathrm{end}}=0.285118\). The largest end-only advantage is

\[
\Delta F=-0.190274
\]

for dephasing with \(M=2,p=0.03,\ell=3\), where
\(F_S^{\mathrm{int}}=0.431545\) and
\(F_S^{\mathrm{end}}=0.621820\). The fact that neighboring register sizes can
show opposite signs emphasizes the state- and circuit-dependent nature of
the comparison.

### 4.6. Signed-estimator overhead

The fidelity differences must be interpreted together with the signed
normalization. Figure \(\ref{fig:overhead}\) plots
\(\log_{10}\mathcal{O}_{\mathrm{norm}}\) for \(M=3\). Interleaving has a
smaller normalization-induced overhead than end-only purification in all 90
nonzero-noise matched comparisons.

\begin{figure*}[t]
\centering
\includegraphics[width=0.94\textwidth]{figures/noisy_circuit_simulations/sampling_overhead_vs_p.pdf}
\caption{Normalization-induced signed-estimator overhead for \(M=3\),
separated by purification depth so that the smaller-\(\ell\) curves remain
visible. The vertical coordinate is
\(\log_{10}[1/|\operatorname{Tr}S|^2]\), not the overhead itself. It does
not include the \(N\) copies consumed per shot or the observable-dependent
variance prefactor.}
\label{fig:overhead}
\end{figure*}

Across all nonzero-noise data, the ranges are:

| \(\ell\) | \(N\) | Range of \(\log_{10}\mathcal{O}_{\mathrm{norm}}\) |
|---:|---:|---:|
| 1 | 32 | \(0.351\) to \(108.414\) |
| 2 | 1,024 | \(10.667\) to \(3\,555.917\) |
| 3 | 32,768 | \(333.500\) to \(113\,850.463\) |

Thus many of the largest fixed-depth fidelity advantages occur in regimes
which cannot be resolved with a practical number of shots under the present
estimator. For example, the largest positive \(\Delta F\) above has

\[
\log_{10}\mathcal{O}_{\mathrm{norm}}^{\mathrm{int}}
=26\,455.6,
\qquad
\log_{10}\mathcal{O}_{\mathrm{norm}}^{\mathrm{end}}
=57\,330.2.
\tag{47}
\]

Interleaving is exponentially less unfavorable in that example, but both
numbers are operationally prohibitive. A large formal target-overlap
advantage at deep purification should therefore not be described as a
practical mitigation advantage.

Because interleaving has lower overhead whenever it has higher fidelity, it
Pareto-dominates end-only purification in the 40 positive-\(\Delta F\)
pairs when the comparison is restricted to these two metrics. In the
remaining 50 pairs, end-only has higher fidelity while interleaving has lower
normalization cost, producing a genuine tradeoff rather than a single
ordering.

### 4.7. Gate resources

The gate-resource asymmetry has the same direction as the estimator overhead.
For \(M=3\), the exported one- and two-qubit location counts are:

| \(\ell\) | \(N\) | \(G_1^{\mathrm{int}}\) | \(G_1^{\mathrm{end}}\) | \(G_2^{\mathrm{int}}\) | \(G_2^{\mathrm{end}}\) | Common \(G_{\mathrm{cswap}}\) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 530 | 1,118 | 62 | 160 | 93 |
| 2 | 1,024 | 13,302 | 35,838 | 1,364 | 5,120 | 3,069 |
| 3 | 32,768 | 388,526 | 1,146,878 | 37,448 | 163,840 | 98,301 |

The common controlled-SWAP count grows as \(M(N-1)\) and becomes a substantial
high-arity hardware cost. Nevertheless, interleaving reduces later workload
execution because the number of active registers has already been compressed
by earlier merges. This is a real architectural advantage of the schedule,
but it also means that the present comparison does not isolate placement at
fixed total circuit noise.

### 4.8. Physical branch average versus signed mitigation

The unsigned operator \(U\) provides an important conceptual control. For all
180 nonzero-noise purified configurations in the sweep, the final physical
branch-average fidelity \(F_U\) is lower than the corresponding no-PQES
baseline. The extra noisy SWAP gadgets degrade the ordinary state when their
outcome record is ignored. The fidelity improvements described above are
therefore properties of the parity-weighted estimator \(S\), not evidence
that the non-postselected circuit deterministically outputs a cleaner
physical state.

\begin{figure*}[t]
\centering
\fbox{\parbox{0.94\textwidth}{
\textbf{TODO result figure.} Plot \(F_S\), \(F_U\), and the no-PQES
fidelity together for representative cases. Include at least one case where
end-only is best and one where interleaving wins. This figure should appear
in a submitted version because it prevents the signed-estimator improvement
from being mistaken for deterministic state purification.}}
\caption{Proposed comparison of mitigated signed-estimator fidelity and
ordinary physical branch-average fidelity.}
\label{fig:signed_vs_physical}
\end{figure*}

This result reinforces the hybrid QEC/QEM interpretation of branch-resolved
PQES. Individual branches are physical and continue through the circuit, but
the non-postselected power-state improvement is recovered only after the
classical parity record is included.

## 5. Why end-only purification often gives higher final fidelity

### 5.1. Final filtering versus early filtering

The ideal identity in Eq. (9) can suggest that placement should be irrelevant.
The noisy schedules instead have the schematic forms

\[
S_{\mathrm{end}}
=
\mathcal{Q}_{D}
\circ\mathcal{C}_T
\circ\cdots
\circ\mathcal{C}_1(S_0),
\tag{48}
\]

and

\[
S_{\mathrm{int}}
=
\mathcal{Q}_{\ell}
\circ\mathcal{C}_T
\circ\mathcal{Q}_{\ell}
\circ\cdots
\circ\mathcal{Q}_{\ell}
\circ\mathcal{C}_1(S_0),
\tag{49}
\]

where \(\mathcal{Q}_d\) denotes the noisy, bilinear, parity-weighted merge
operation of depth \(d\). There is no general commutation relation between
\(\mathcal{Q}_d\) and \(\mathcal{C}_t\).

End-only purification has a simple fidelity advantage: its last operation is
the noisy PQES tree, whose intended power-map action filters the accumulated
final noisy state, and no later workload channel can reintroduce error.
Interleaved PQES filters intermediate states, after which subsequent noisy
circuit blocks can contract the state, rotate its dominant eigendirection
relative to the eventual target, or introduce fresh mixture. An early gain
can therefore be partially lost.

### 5.2. State-dependent dominant eigendirections

The power map amplifies whichever eigendirection is dominant at the moment it
is applied. In a nontrivial circuit, the ideal target changes after every
block. A noisy intermediate state's dominant eigenvector need not evolve
under later noisy channels into the final ideal target. Interleaving can then
commit estimator weight to an intermediate eigendirection which is locally
dominant but globally suboptimal. End-only purification instead selects the
dominant component of the final noisy state.

This explanation is especially relevant for dephasing, which is anisotropic
with respect to the instantaneous circuit state. The brickwork rotations
continually change how the target is oriented relative to the phase-flip
axis. The alternating signs in Fig. \(\ref{fig:advantage}\) across \(M\),
\(p\), and \(\ell\) are consistent with such state-dependent eigenbasis
effects.

### 5.3. Noisy purification is itself an error channel

Every register-level PQES merge adds two noisy ancilla Hadamards, \(M\) noisy
Fredkins, \(M(1+2M)\) SWAP-memory locations, one ancilla reset, and one
readout sign. Increasing a complete tree from depth \(D\) to \(D+1\) changes
the merge count from \(2^D-1\) to \(2^{D+1}-1\), thereby adding \(2^D\)
merges. Increasing \(\ell\) therefore strengthens both the desired power
filtering and the hardware error accumulated in the estimator. The result is
not expected to be monotone in \(\ell\). Indeed, the fidelity-optimized
configurations are usually at \(\ell=1\), whereas many fixed-depth
interleaving wins occur at \(\ell=2\) or \(3\), where the estimator overhead
is already extreme.

### 5.4. Why interleaving can still win

Interleaving has two countervailing advantages.

First, purification acts before later errors accumulate. If an intermediate
state is beginning to lose its target-dominant spectral ordering, an early
power step can amplify the desired component while it is still recoverable.
Second, early merging reduces the number of registers that traverse later
workload blocks. This lowers both gate exposure and the decay of
\(\operatorname{Tr}S\). These effects explain why interleaving can outperform
end-only purification at fixed larger depth even though it is rarely the
fidelity-optimal schedule after \(\ell\) is optimized.

The balance among final filtering, intermediate spectral ordering, noisy
PQES operations, and active-register reduction is circuit dependent. No
general theorem ordering Eqs. (48) and (49) should be expected for arbitrary
nonunitary channels.

## 6. Discussion

### 6.1. The central answer

The present data do not support the claim that interleaving universally
improves final fidelity. They support a more specific statement:

> At matched leaf-copy count and SWAP-tree depth, interleaving can improve the
> final signed-estimator fidelity in particular circuit and noise regimes,
> while systematically reducing workload-gate exposure and signed
> normalization cost. End-only purification is usually superior when the
> tested purification depth is optimized solely for final fidelity.

This distinction is scientifically useful. The freedom to interleave is not
itself an error-suppression guarantee; it is an additional compilation
degree of freedom.

### 6.2. Bias, variance, and physical resources

Schedule selection should be formulated as a multiobjective optimization.
For an observable \(O\), a practical cost function might include

\[
\mathcal{C}
\sim
C_{\mathrm{exec}}\,
\frac{
\operatorname{Var}\!\left[
\Omega(o-\langle O\rangle)
\right]
}{
\epsilon^2|\operatorname{Tr}S|^2
},
\tag{50}
\]

where \(C_{\mathrm{exec}}\) is the total gate, time, energy, or spacetime cost
of one execution of the full \(N\)-copy estimator circuit, and \(\epsilon\)
is the desired standard error. In particular, \(C_{\mathrm{exec}}\) contains
at least the cost of \(N\) state preparations together with the workload and
PQES locations counted in Sec. 3.6. Fidelity alone captures the bias of the
normalized signed estimator but not the resources required to resolve it.
Equation (50) also makes clear that the copy contribution within
\(C_{\mathrm{exec}}\) and the normalization factor can overwhelm a modest
fidelity gain.

A useful schedule can therefore be:

- higher fidelity at fixed cost;
- lower cost at fixed target fidelity;
- lower sampling variance at fixed bias; or
- Pareto optimal across several of these quantities.

The present heatmap answers only the first question at fixed \(\ell\), while
the overhead and resource figures begin to address the others.

### 6.3. Relation to the ideal PQES thresholds

The location error \(p\) in this circuit-level study is not directly
comparable to the state-level \(p\) used in the ideal PQES threshold analysis.
Here the same default value is applied repeatedly to preparation, every
one- and two-qubit workload gate, every Fredkin, every memory interval, every
reset, and every readout sign. The total circuit error grows with the number
of locations and with \(N\). Moreover, the SWAP tree is noisy and has finite
depth.

Consequently, the thresholds \(p_{\mathrm{th}}=3/4\) or \(1/2\) derived for a
single idealized state channel should not be described as circuit-level
thresholds. A true circuit-level PQES threshold would require a scalable
family of circuits, a fixed hardware-local error model, an explicit resource
scaling prescription, and an asymptotic demonstration that an operational
error metric decreases as the redundancy grows.

### 6.4. Physical interpretation

The physical branch average is degraded by the noisy PQES circuit in every
nonzero-noise purified case studied here. The improvement resides in the
signed record. This places the present noisy implementation firmly on the QEM
side of the hybrid interpretation: the branch states are real intermediate
states, but the improved final quantity is a ratio estimator reconstructed
from measurement outcomes.

This does not make the branch-resolved formulation redundant. The explicit
branches determine how states can continue through later circuit blocks, how
noise enters the recursive protocol, how registers can be recycled, and how
the estimator's sign and normalization evolve. They are precisely what make
the interleaving question well defined.

## 7. Limitations of the present study

The current conclusions are conditional on several modeling choices.

1. **Single circuit instance.** The frozen dataset uses one brickwork circuit
   with seed 7. The 90 nonzero-noise comparisons are a parameter grid for
   that circuit, not 90 independent circuit samples. There are no ensemble
   confidence intervals.

2. **Small register sizes.** Exact dense SWAP instruments restrict the main
   sweep to \(M\le3\). Behavior at larger \(M\) may differ, particularly
   because entanglement, state spectra, and Fredkin counts scale with register
   size.

3. **Five circuit blocks.** Only \(T=5\) is included. A scheduling advantage
   may depend strongly on circuit depth and on where high-error blocks occur.

4. **Uniform error rates.** The default model sets six physically different
   rates equal. Realistic devices generally have
   \(p_{\mathrm{cswap}}>p_{\mathrm{2q}}>p_{\mathrm{1q}}\), nonuniform
   measurement and reset errors, and memory rates determined by gate
   duration.

5. **Native Fredkin assumption.** The controlled-SWAP is not decomposed into
   a hardware basis and no connectivity, routing, or scheduling constraints
   are imposed.

6. **Restricted channels.** Only depolarizing and phase-flip families are
   included. Amplitude damping, coherent over-rotations, leakage, correlated
   noise, and crosstalk are absent.

7. **Independent symmetric readout error.** Readout changes only the sign
   visibility \(1-2p_{\mathrm m}\). State-dependent, biased, or correlated
   readout is not modeled.

8. **Exact expectation values.** No finite-shot Monte Carlo is performed.
   Equation (7) is a leading normalization proxy rather than an observed
   estimator variance.

9. **Matched-copy comparison.** The schedules have equal \(N\) and equal
   SWAP counts but unequal workload-gate and memory counts. This is an
   intended architectural comparison, not a fixed-total-noise comparison.

10. **No transpilation or wall-clock model.** Parallelism, qubit recycling,
    reset latency, classical feed-forward, and coherence during scheduling
    are not assigned durations.

11. **No state-positivity guarantee for the signed estimator.**
    \(S/\operatorname{Tr}S\) is treated as a mitigated operator. A future
    study should monitor its spectrum and specify how nonphysical estimates
    are handled statistically.

12. **Resource-export bookkeeping.** The current final-table memory-location
    field omits workload-block memory locations, as noted after Eq. (35).

These limitations prevent the current sweep from establishing a universal
interleaving theorem or a hardware threshold. They do not affect the narrower
conclusion that schedule ordering is nonuniversal in the stated model.

## 8. Program for a complete follow-up study

A publication-ready extension should contain the following studies.

### 8.1. Circuit ensemble

Run many independent brickwork seeds for each
\((M,T,p,\ell,\text{schedule})\), and report the mean, median, quantiles, and
fraction of circuits for which \(\Delta F>0\). Include:

- idle controls, isolating memory and PQES-gadget noise;
- echo circuits, adding gates while keeping the ideal state simple;
- random brickwork circuits;
- structured algorithms or ansätze with varying entanglement; and
- circuits with deliberately localized high-noise blocks.

This will distinguish a circuit-specific sign pattern from a robust schedule
trend.

### 8.2. Noise ablations

Vary one location class at a time:

\[
p_{\mathrm{1q}},\quad
p_{\mathrm{2q}},\quad
p_{\mathrm{cswap}},\quad
p_{\mathrm{mem}},\quad
p_{\mathrm m},\quad
p_{\mathrm r}.
\]

The ablations should identify whether any observed interleaving advantage is
driven primarily by memory reduction, fewer workload gates, earlier spectral
filtering, or lower parity decay. At minimum, compare:

1. noisy workload with ideal PQES;
2. ideal workload with noisy PQES;
3. noisy workload and Fredkins but ideal SPAM;
4. SPAM-only errors; and
5. the full model.

### 8.3. Hardware-derived profiles

Replace uniform \(p\) with profiles drawn from one or more target devices,
including native gate durations and connectivity. If no native Fredkin exists,
transpile the SWAP test into the device basis and apply noise to the
decomposed circuit. Report both gate count and scheduled duration.

### 8.4. Alternative fairness criteria

Compare schedules at:

- fixed \(N\) and \(G_{\mathrm{cswap}}\) (the present convention);
- fixed total one- and two-qubit locations;
- fixed total spacetime volume;
- fixed number of state preparations;
- fixed wall-clock time; and
- fixed total shot or copy budget required to reach an error bar.

Padding a lower-cost schedule with idle noise may be useful for separating
placement effects from simple exposure reduction, although such padding
answers a different operational question.

### 8.5. Shot-level estimator analysis

Simulate complete SWAP records and final observable measurements for selected
small cases. Verify the ratio-estimator bias and variance, confidence-interval
coverage, and the scaling predicted by
\(|\operatorname{Tr}S|^{-2}\). Compare direct ratio estimation with
variance-reduction or correlated-estimator strategies.

### 8.6. Placement optimization

The two schedules studied here are endpoints of a larger space. For a circuit
with blocks \(\mathcal{C}_1,\ldots,\mathcal{C}_T\), let
\(\ell_t\ge0\) be the number of PQES levels after block \(t\), subject to

\[
\sum_{t=1}^{T}\ell_t=D.
\tag{51}
\]

The placement vector
\(\boldsymbol{\ell}=(\ell_1,\ldots,\ell_T)\) can be optimized for fidelity,
overhead, or a combined cost such as Eq. (50). Exhaustive search is feasible
for small \(T,D\); larger cases may use dynamic programming, greedy
allocation, Bayesian optimization, or differentiable surrogate models.

### 8.7. Larger-system methods

Dense simulation scales poorly because each noisy SWAP instrument contains
\(1+2M\) qubits. Larger registers will require tensor-network channels,
stabilizer or Pauli-transfer methods for restricted circuits, low-rank
approximations, or stochastic trajectories. Cross-checking those methods
against the exact \(M\le3\) data will be essential.

## 9. Conclusions

We have extended branch-resolved PQES from an ideal state-level purification
map to an explicit circuit-level model in which preparation, workload gates,
controlled-SWAPs, memory, resets, and measurements are noisy. The complete
SWAP record can be propagated without explicit branch enumeration by evolving
an ordinary physical aggregate \(U\) and a parity-weighted signed aggregate
\(S\). This construction preserves the operational distinction between the
physical branch average and the mitigated power-state estimator.

For the representative five-block brickwork circuit studied here,
interleaving is not uniformly superior to applying purification only at the
end. At fixed nonzero noise and matched purification depth, the comparison is
mixed: interleaving wins 40 of 90 cases, while end-only purification wins 50.
When the tested purification depth is optimized solely for fidelity, end-only
purification is best in 29 of 30 parameter combinations. The simplest
explanation is that end-only purification acts as a final spectral filter
after all workload noise has accumulated, while interleaved gains can be
degraded by later channels.

Interleaving nevertheless provides systematic resource advantages. It
reduces the number of live registers exposed to later workload blocks and
has lower signed-normalization overhead in every nonzero-noise matched pair.
At larger fixed depth it can also produce sizeable fidelity advantages.
Unfortunately, many such cases occur where the required copy and sampling
costs are already prohibitive.

The practical question is therefore not whether interleaving is categorically
better, but where purification layers should be placed to optimize bias,
variance, copies, gate count, and time for a specified circuit and hardware
profile. This reframes interleaved PQES as a compilation and resource-
allocation problem. Establishing robust placement principles will require
circuit ensembles, hardware-derived noise, controlled ablations, and
finite-shot analysis.

## Acknowledgments

This work is supported by the SMEC Scientific Research Innovation Project
(2023ZKZD55); the National Natural Science Foundation of China (92576102);
the Science and Technology Commission of Shanghai Municipality
(22ZR1444600); the NYU Shanghai Boost Fund; the China Foreign Experts Program
(G2021013002L); the NYU-ECNU Institute of Physics at NYU Shanghai; the NYU
Shanghai Major-Grants Seed Fund; and Tamkeen under the NYU Abu Dhabi Research
Institute grant CG008.

## Data and code availability

The implementation is contained in
`src/noisy_circuit_simulations/`. The frozen data used in this draft are
`data/noisy_circuit_simulations/steps.csv`,
`data/noisy_circuit_simulations/finals.csv`,
`data/noisy_circuit_simulations/schedule_comparison.csv`, and
`data/noisy_circuit_simulations/manifest.json`. The motivating state-only
study is implemented in
`src/simulation/rho2_sims/interleaving_study.py`, with its two datasets in
`data/rho2_interleaving/`. Figures are stored in
`figures/noisy_circuit_simulations/` and `figures/rho2_interleaving/`.

Before submission, these paths should be replaced by a permanent repository
release and DOI, and the exact committed revision should be recorded.

---

## Appendix A. Data products and column semantics

### A.1. `steps.csv`

Each row corresponds to one circuit cycle of one experiment. Important fields
are:

| Column | Meaning |
|---|---|
| `run_id` | Deterministic identifier containing \(M,T\), circuit, noise, \(p\), schedule, \(\ell\), and seed |
| `iteration` | One-indexed circuit/PQES cycle |
| `M` | Qubits per data register |
| `depth` | PQES levels completed by this cycle |
| `copies_used`, `N_so_far` | \(2^{\text{depth}}\) |
| `noise` | `depolarizing` or `dephasing` |
| `p` | Primary one-qubit location error \(p_{\mathrm{1q}}\) |
| `purification_level` | Assigned \(\ell\) per circuit block |
| `schedule` | `none`, `interleaved`, or `end_only` |
| `fidelity` | Signed-estimator overlap \(F_S(t)\) |
| `physical_fidelity` | Unsigned branch-average overlap \(F_U(t)\) |
| `eps_L`, `purity` | Diagnostics of \(\bar S_t\) |
| `physical_eps_L`, `physical_purity` | Diagnostics of \(\bar U_t\) |
| `signed_trace` | \(\operatorname{Tr}S_t\), possibly underflowed to zero |
| `log_abs_signed_trace` | Stable \(\log|\operatorname{Tr}S_t|\) |
| `sampling_overhead` | \(1/|\operatorname{Tr}S_t|^2\), possibly infinite in floating point |
| `log10_sampling_overhead` | Stable \(-2\log_{10}|\operatorname{Tr}S_t|\) |
| `fidelity_before_noise` | Signed overlap before the current workload block |
| `fidelity_after_noise` | Signed overlap after workload and block-memory noise, before any current PQES |
| `purification_levels_this_cycle` | Actual PQES levels applied at this cycle |
| `branch_probability_plus/minus` | True pre-readout probabilities in the final merge of this cycle |
| `p_1q` ... `p_reset` | Realized location rates |
| gate-count columns | Cumulative counts along one representative surviving branch, not full-tree totals |
| `P_success` | Legacy placeholder equal to 1 because no postselection is used |
| `grover_iters` | Legacy placeholder equal to 0 |
| `twirling_applied` | False in the current circuit-level study |

### A.2. `finals.csv`

Each row corresponds to one completed experiment. It contains the initial
post-first-block diagnostics, final signed and physical metrics, all realized
error rates, and full-tree resource counts. `fidelity_init` is the metric
after the first noisy workload block and its memory interval, before the first
possible PQES layer; it is not the fidelity immediately after preparation.

### A.3. `schedule_comparison.csv`

Each row joins one interleaved/end-only pair with identical
\((\text{noise},M,p,T,\ell,\text{circuit},\text{seed},\boldsymbol p)\).
It contains both schedules' metrics, the no-PQES baseline,
\(\Delta F\), gains over baseline, log-overhead difference, resource
difference, matched \(N\), and the higher-fidelity schedule.

### A.4. Motivating state-only data

`data/rho2_interleaving/matched_budget_M1_plus_p030_T5.csv` and
`matched_budget_M1_plus_p060_T5.csv` contain every noise and purification
checkpoint in the one-qubit calculation of Sec. 2.6. The purified schedules
record a depth budget of five, 32 input copies, and 31 pairwise merges.

## Appendix B. Reproduction record

The motivating state-only comparisons were generated with:

```bash
python -m src.simulation.rho2_sims.interleaving_study
python -m src.simulation.rho2_sims.interleaving_study --p 0.6
```

The frozen full noisy-circuit sweep was generated with:

```bash
MPLCONFIGDIR=/private/tmp/swap_qec_mpl \
python -m src.noisy_circuit_simulations.grid_run

MPLCONFIGDIR=/private/tmp/swap_qec_mpl \
python -m src.noisy_circuit_simulations.plots

python -m src.noisy_circuit_simulations.analysis

MPLCONFIGDIR=/private/tmp/swap_qec_mpl \
python -m pytest src/noisy_circuit_simulations/test_models.py
```

Environment used for the current files:

| Component | Version |
|---|---|
| Python | 3.9.23 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| Matplotlib | 3.9.2 |
| Qiskit | 2.1.2 |
| Qiskit Aer | 0.17.1 |

The repository base revision observed while drafting was
`59668d1425b391123b45a157027985936fc86603`, but the noisy-circuit package,
data, figures, and this draft were uncommitted at that time. A submitted
version must cite a later clean commit which actually contains every artifact.

## Appendix C. Publication-critical TODO checklist

The following items should be resolved before this document is treated as a
paper rather than a preservation draft.

- [ ] Add and verify the citation for the preceding PQES paper
  (`RaghoonananByrnesPQES`).
- [ ] Freeze a clean git revision, environment lockfile, archive, and DOI.
- [ ] Correct the exported full-tree workload-memory resource count and
  regenerate affected tables.
- [ ] Generate the matched-schedule schematic in
  Fig. \(\ref{fig:schedule_schematic}\).
- [ ] Generate the signed-versus-physical comparison in
  Fig. \(\ref{fig:signed_vs_physical}\).
- [ ] Run multiple brickwork seeds and report uncertainty across circuits.
- [ ] Run the idle and echo control families already supported by the code.
- [ ] Add circuit-depth sweeps in \(T\).
- [ ] Perform location-by-location noise ablations.
- [ ] Add at least one hardware-derived, nonuniform error profile.
- [ ] Decompose/transpile the Fredkin for a realistic native gate set.
- [ ] Decide which resource is matched in each headline comparison.
- [ ] Add finite-shot ratio-estimator simulations and confidence intervals.
- [ ] Monitor and report the minimum eigenvalue of \(\bar S\).
- [ ] Decide whether \(\epsilon_L\) and \(\gamma_1\) should remain in the
  main paper or be renamed to avoid QEC-threshold connotations.
- [ ] Re-evaluate whether the uniform-\(p\) depolarizing and dephasing
  channels should share the same horizontal axis in comparison figures.
- [ ] Add a table mapping every paper figure to its generation command and
  source-data filter.
- [ ] Copy finalized figures and references into the Overleaf project.
- [ ] Replace provisional result language after the ensemble and ablation
  studies are complete.

## Appendix D. Provisional citation keys

The following keys are used above and should be synchronized with the
original manuscript bibliography:

`NielsenChuang2010`, `Gottesman1997Thesis`, `Fowler_2012`,
`Temme2017Mitigation`, `Cai2023QEM`, `Bennett1996Twirling`,
`Deutsch1996QPA`, `Dur2007EPReview`, `Huggins2021`,
`Koczor_derangement`, `Koczor2021_dominantEV`, and the new placeholder
`RaghoonananByrnesPQES`.
