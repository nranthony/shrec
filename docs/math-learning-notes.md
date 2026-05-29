# SHREC math — learning dump

> Working scratchpad written *while* building the MM1–MM4 inner-math tests
> and fixing the σ root-solve (branch `fix/simplex-sigma-solve`, 2026-05-28).
> Unstructured on purpose: raw thoughts, "why does this work" moments, and
> **learning topics** flagged for a future Quarto explainer pass. Collate
> later → pick the most human-readable directions for understanding the math.
>
> Convention: a bullet tagged **📚 LEARNING TOPIC** is a candidate Quarto
> page/section. Everything else is context for why it's interesting.

---

## The object under test: the fuzzy simplicial complex (Appendix B, step 3)

For one response channel we have a point cloud (the delay-embedded trajectory).
For each point *i* we build a "fuzzy" membership to every other point *j*:

```
a_ij = exp( -ReLU(d_ij - ρ_i) / σ_i )
```

- `ρ_i` = distance to the *nearest* neighbour of *i*. Subtracting it and
  ReLU-clipping means the nearest neighbour gets membership `exp(0)=1`, and
  anything closer than ρ (only *i* itself) is also clamped to 1. This is the
  "local connectivity = 1" idea: every point is certainly connected to its
  closest neighbour, regardless of absolute scale.
- `σ_i` = a per-point bandwidth chosen so the *row* carries a fixed amount of
  total fuzzy mass.

The bandwidth is fixed by the **defining equation**:

```
Σ_m exp( -ReLU(d_im - ρ_i) / σ_i )  =  log₂(k)
```

summed over the *k* nearest neighbours.

- **📚 LEARNING TOPIC — "Why log₂(k)?"** This is the UMAP `smooth_knn_dist`
  target. It's a *perplexity-like* normalisation: instead of every point
  having k hard neighbours, each point has a fixed *effective* number of
  fuzzy neighbours = log₂(k). It makes the per-point bandwidth adapt to local
  density (dense region → small σ, sparse region → large σ) while keeping the
  total "information" per row constant. Worth a side-by-side with t-SNE's
  perplexity (which uses Shannon entropy / log, not log₂). Good Quarto figure:
  same point cloud, color each point by its solved σ_i, show σ tracking local
  spacing.

- **📚 LEARNING TOPIC — "Local connectivity and the ρ shift."** Why subtract
  the nearest-neighbour distance at all? Show what the affinity graph looks
  like with vs without the ρ shift on a cloud with two density scales. Without
  it, the sparse cluster fragments. This is *the* reason UMAP/SHREC handle
  multi-scale data.

---

## The bug this branch fixes (and what it taught me)

The σ in the defining equation has to be found numerically — no closed form.
The original code used `scipy.optimize.fsolve` with the Newton-style MINPACK
`hybrd` routine, starting from `x0 = ρ_i`.

**What went wrong:** on ~6% of rows `fsolve` returned `ier=1` ("converged")
but the residual of the defining equation was ≈ 2.8, not < 1e-6 — and σ came
back *exactly equal to the initial guess ρ_i*. It never took a step.

- **📚 LEARNING TOPIC — "Convergence criteria lie."** MINPACK's `hybrd`
  declares success on the *step size* falling below `xtol`, NOT on the
  residual being small. If the first proposed step is tiny (bad scaling of
  the analytic Jacobian here), it can quit at iteration 0 and still report
  success. Great cautionary tale: *always check the residual yourself*, never
  trust the solver's status flag. This is a general numerical-methods lesson,
  not SHREC-specific — strong candidate for a standalone explainer.

**Why the fix is clean — monotonicity.** The left side of the defining
equation is *strictly monotone increasing* in σ:

- as σ → 0⁺, every non-nearest term `exp(-positive/σ) → 0`, so the sum → (number
  of neighbours exactly at distance ρ) = 1 in general → and `1 < log₂(k)` for
  k > 2.
- as σ → ∞, every term `exp(-x/σ) → 1`, so the sum → k > log₂(k).

A continuous function going from below-target to above-target has exactly one
root, and a *bracketing* method (`scipy.optimize.brentq` on `[1e-12, 1e6]`)
finds it to machine precision (residual ~1e-12 on every row). No initial guess,
no Jacobian, no silent stalls.

- **📚 LEARNING TOPIC — "Bracketing vs Newton root-finding."** When you can
  prove monotonicity + a sign change, bracketing (bisection/Brent) is
  unconditionally robust; Newton/secant are faster but can stall or diverge.
  The SHREC σ-solve is a perfect worked example because monotonicity is easy
  to see from the physics of the kernel. Quarto: plot `f(σ)` for a stalled row,
  mark the fsolve stall point and the true brentq root.

---

## What the four MM tests actually pin (and the math idea behind each)

- **MM1 — defining equation residual.** The direct check: solve, plug back in,
  assert `|Σ exp(...) − log₂k| < 1e-6`. This is the test that *catches the
  fsolve stall*. Lesson: the cheapest possible test (does the equation we
  claim to solve actually hold?) was the one missing, and it caught a real bug.

- **MM2 — scale invariance of (ρ, σ).** If you scale all distances by α, then
  `ρ → αρ` and `σ → ασ`, leaving every affinity `exp(-ReLU(αd-αρ)/(ασ))`
  unchanged. **📚 LEARNING TOPIC — "Dimensional analysis as a test oracle."**
  σ has units of distance; the equation is scale-covariant by construction.
  Recognising the symmetry *before* coding gives you a free, exact test. This
  is a reusable habit: find the equivariance, turn it into an assertion.

- **MM3 — unit diagonal.** `d_ii = 0`, so `ReLU(0 − ρ_i) = 0`, `exp(0) = 1`;
  the fuzzy union `1 + 1 − 1·1 = 1`. A point is fully self-connected. Trivial,
  but pins that the symmetrisation didn't accidentally hollow the diagonal.

- **MM4 — fuzzy-union stays in [0,1] and is symmetric.** The symmetrisation
  `A + Aᵀ − A∘Aᵀ` is the **probabilistic t-conorm** (fuzzy OR):
  `P(i~j) = P(a) + P(b) − P(a)P(b)`, i.e. "edge exists if *either* direction
  votes for it." **📚 LEARNING TOPIC — "Fuzzy set unions / t-conorms."** Why
  this specific formula and not `max` or averaging? It's the probability that
  at least one of two independent directed edges fires. Connects SHREC's graph
  to probabilistic-graph / fuzzy-logic foundations. Nice Quarto: the three
  candidate symmetrisations (max, mean, t-conorm) and how the consensus graph
  differs.

---

## Threads to pull later (broader than this branch)

- **📚 LEARNING TOPIC — "From recurrence to a graph Laplacian."** The
  continuous-driver path takes the Fiedler eigenvector of `L = D − A`, not the
  2nd singular vector of `A` (MM22 pins this). On *irregular* graphs these
  differ. A Quarto page deriving why the Fiedler vector is the natural
  "smoothest non-trivial coordinate" on the recurrence graph would tie the
  whole continuous side together.
- **📚 LEARNING TOPIC — "Consensus over channels = mean of fuzzy graphs."**
  Step 4 averages the per-channel affinity matrices. Why mean and not a fuzzy
  intersection? What does the mean assume about shared-driver structure?
- **📚 LEARNING TOPIC — "The Sauer limit."** Period-2 recovers ARI≈1 but
  period-4 collapses to 2 Leiden communities (MM20 xfail). The math of *why
  community-detection resolution interacts with the driver's period* is a
  genuinely interesting open thread — ties to modularity-vs-CPM objectives.

---

## Late addition — the bug the *fix* exposed (tied neighbourhoods)

Swapping `fsolve → brentq` immediately broke 5 previously-green pipeline tests,
all with `ValueError: f(a) and f(b) must have different signs`. Tracking it down
was itself the best learning moment of the session:

On the **real** driver-response data (logistic maps with `np.clip(·, 0, 1)`),
many embedded points are *exactly identical* because the clip saturates the
series to 0.0 / 1.0. So a point can have all `k` nearest neighbours at the
**same** distance ρ. Then every kernel term is `exp(-ReLU(0)/σ) = 1` for *any*
σ, the row mass is pinned at `k`, and the target `log₂k < k` is **unreachable** —
the defining equation has no solution. `brentq` correctly refuses (no sign
change); `fsolve` had been silently returning the stalled guess and nobody knew.

- **📚 LEARNING TOPIC — "When the equation has no solution."** A whole genre of
  numerical bug: the solver isn't wrong, the *model* is degenerate on this
  input. The honest fix is to detect the degeneracy (here: `f(σ→0⁺) ≥ 0`) and
  define the limiting behaviour explicitly (σ indeterminate → fall back to ρ),
  not to loosen tolerances until it stops complaining. Great Quarto narrative:
  show the constant `f(σ) = k − log₂k` curve for a tied row next to a healthy
  monotone-crossing curve.
- **📚 LEARNING TOPIC — "Quantisation / clipping creates exact ties."** The
  data-generation detail (`clip`) has a downstream geometric consequence
  (duplicate points → distance ties → degenerate local connectivity). Worth a
  short note on how preprocessing choices silently shape the recurrence graph.
- **Meta-lesson for the test suite:** the *unit* test (MM1) and the *pipeline*
  tests disagreed about what "correct" means until the degenerate case was
  handled. Both were needed — MM1 to find the stall, the pipeline tests to
  reveal that the naive fix changed real behaviour. This is the argument for
  keeping both altitudes of test, not just one.

---

# Round 2 — devising parallel & perpendicular tests (2026-05-28)

After MM1–MM4 landed, surveyed the rest of the code surface + complementary
literature (UMAP `smooth_knn_dist`, von Luxburg spectral-clustering tutorial,
Sauer/Takens embedding, the SHREC paper Appendix B itself). Two axes:

- **Parallel** = more of the same family (closed-form inner-math oracles) on
  operators that currently have *none*.
- **Perpendicular** = different *methodology*: property-based, metamorphic,
  differential-vs-reference, fuzz/stability, statistical/surrogate.

## Parallel candidates (closed-form, cheap)

- **MM6** — `kernel.data_to_connectivity` p-norm limits: `ord=1` ⇒ elementwise
  mean of per-channel kernels; `ord→∞` ⇒ min-over-channels distance (Sauer
  `inf_k d^(k)`). Verified no NaN at ord=500; limit shape is assertable.
- **MM7/MM8** — `cdist` isometry invariance (random orthogonal Q + translation)
  and triangle inequality. Trivially true for euclidean, but pins that no code
  path rescales distances asymmetrically.
- **MM9** — `distance_to_connectivity(sparsity=…)` actually hits the requested
  sparsity, and is monotone in the scale.
- **MM10** — `_leiden` backend agreement (graspologic vs igraph vs leidenalg)
  on the barbell graph: ARI=1 across backends.
- **MM11** — `unionfind.DisjointSet` parity with
  `scipy.cluster.hierarchy.DisjointSet` on random merge lists.
- **NEW — Laplacian algebra**: `L = D−A` has row sums 0 (constant vector is
  eigenvector at λ=0); eigenvalues ≥ 0; λ₂ > 0 **iff** the graph is connected.
  This is the oracle for the disconnected-graph bug below.
- **NEW — embedding reconstruction**: `embed_ts`/`hankel_matrix` output equals a
  hand-rolled delay stack `[x_t, x_{t-τ}, …]`; padding modes preserve length.

## Perpendicular candidates (different methodology)

- **📚 LEARNING TOPIC — property-based testing (Hypothesis).** `hypothesis` is
  already a dev dep and registered in pytest but used *nowhere*. Strong fit
  for the algebraic invariants: for arbitrary small clouds, assert
  `dataset_to_simplex` output is symmetric/in [0,1]/unit-diagonal; assert
  `fit_rho_sigma` residual < tol OR the tied-degenerate fallback fired. Lets
  the machine search for the adversarial inputs we'd never hand-pick (it would
  have found the tied-neighbourhood case on its own).
- **📚 LEARNING TOPIC — metamorphic testing.** When there's no ground-truth
  output, test *relations between* outputs: monotonicity (more coupling ⇒
  higher recovery ARI), graceful degradation (ARI decreases monotonically as
  noise σ rises), adding a pure-noise channel must not *improve* a clean
  recovery. These encode the paper's qualitative claims as ordering assertions.
- **📚 LEARNING TOPIC — differential / reference-oracle testing.** Pin our
  operators against an independent implementation: `dataset_to_simplex` vs
  `umap.umap_.fuzzy_simplicial_set` (this is MM5, currently xfail — worth
  *reconciling conventions* rather than leaving frozen); `RecurrenceManifold`
  Fiedler vs `sklearn.manifold.SpectralEmbedding`; `_leiden` vs raw backends.
- **📚 LEARNING TOPIC — surrogate / null-hypothesis testing.** Borrowed from
  nonlinear-time-series practice (the paper itself uses surrogate ideas):
  feed SHREC *independent* responses with **no** shared driver and assert the
  recovered driver is statistically indistinguishable from noise (low ARI /
  flat eigenvector). Guards against the method hallucinating structure — the
  most important property a causal-discovery tool can have.
- **📚 LEARNING TOPIC — numerical fuzz / stability.** Sweep dtype (float32 vs
  64), tiny/huge distance scales, near-duplicate rows, NaN injection. The σ
  bug was exactly a stability failure; a fuzz harness generalises the guard.

## Potential issues found while surveying (grounded, not speculative)

1. **Unnormalised Laplacian degree bias.** `RecurrenceManifold` uses `L = D−A`
   (RatioCut), not `L_sym`/`L_rw` (NCut). On degree-heterogeneous consensus
   graphs the Fiedler vector localises on high-degree nodes (von Luxburg). The
   paper itself flags this: "correct for response bias … might require
   preconditioning the graph Laplacian." → metamorphic test: skewing one
   channel's recurrence density should not dominate the recovered driver.
   **📚 LEARNING TOPIC — RatioCut vs NCut and why normalisation matters.**
2. **Disconnected-graph degeneracy (demonstrated).** With no bridge between
   blocks, λ₂ = 0 (2-dim null space) and `eigh(subset_by_index=[1,1])` returns
   an arbitrary component *indicator* (step function {0, 0.183}), not a smooth
   driver. No connectivity check exists. → guard: assert/ warn when λ₂ ≈ 0;
   test on a deliberately disconnected affinity.
3. **`distance_to_connectivity` bracket fragility — twin of the σ bug.**
   `root_scalar(optfun, bracket=[1e-16, dscale])` assumes a sign change that
   isn't guaranteed for every distance distribution; same failure class we
   just fixed in `fit_rho_sigma`. Needs the same bracket-guard discipline.
4. **igraph backend silently ignores `resolution`.** `communities._leiden`
   igraph branch hardcodes `resolution_parameter=1.0` instead of forwarding
   `resolution`. A backend-agreement test (MM10) that varies resolution would
   catch this. (graspologic path does forward it correctly.)
5. **`subset_by_index=[1, n_components]` is inclusive-count-off-by-one-prone.**
   For `n_components=1` it returns one vector (good), but the semantics ("index
   1 through n_components") differ from "n_components vectors" for >1. Worth a
   shape test pinning the intended contract.

---

# Round 3 — theory deep-dives (2026-05-29)

Educational write-ups attached to the three fixes (igraph resolution, the
connectivity guard, the Hypothesis suite). These are deliberately fuller than
scratch notes — they're the seed text for Quarto explainer pages. Each is
self-contained; trim/expand when collating.

## A. The graph Laplacian and algebraic connectivity (behind the MM33 guard)

For an undirected weighted graph with affinity `A` and degree `D = diag(Σ_j A_ij)`,
the **combinatorial Laplacian** is `L = D − A`. Three facts make it the natural
operator for the continuous driver:

1. **`L` is symmetric PSD.** For any vector `f`,
   `fᵀ L f = ½ Σ_{i,j} A_ij (f_i − f_j)²  ≥ 0`.
   This quadratic form is the "energy" of `f` on the graph — small when `f`
   varies slowly across strongly-connected nodes. So the *smallest* non-trivial
   eigenvector is the *smoothest non-constant coordinate*: exactly what a slowly
   varying shared driver should look like on the recurrence graph.
2. **The constant vector is always an eigenvector at λ₁ = 0** (`L·1 = 0`,
   because rows sum to zero). That's why `RecurrenceManifold` asks `eigh` for
   `subset_by_index=[1, …]` — it *skips* index 0, the uninformative constant.
3. **Algebraic connectivity.** The second-smallest eigenvalue λ₂ (the *Fiedler
   value*) is `> 0` **iff** the graph is connected. More strongly: *the
   multiplicity of the eigenvalue 0 equals the number of connected components*,
   and the null-space is spanned by the component-indicator vectors.

Fact 3 is the entire content of the MM33 bug. When the consensus graph splits
into (nearly) disconnected pieces, λ₂ → 0 and the "Fiedler" eigenvector we pull
out is an arbitrary mixture of component indicators — a *step function*, not a
driver. `eigh` returns it without complaint. The guard watches λ₂ against the
spectral scale (`trace L = Σ degree`) and warns.

- **📚 LEARNING TOPIC — "The Laplacian quadratic form."** Derive
  `fᵀLf = ½ Σ A_ij (f_i−f_j)²` from scratch; it's three lines and it demystifies
  why spectral methods "find smooth coordinates." Best single figure in the
  whole topic: a path graph with its first few Laplacian eigenvectors drawn as
  standing waves (they're literally `cos(πki/n)` — discrete vibration modes).
- **📚 LEARNING TOPIC — "Eigenvalue 0 counts components."** Worked 2-block
  example with the bridge weight swept from 1 → 0, watching λ₂ slide to 0 and
  the eigenvector morph from a smooth ramp into a hard step. This *is* the MM22
  → MM33 continuum in one animation.

## B. RatioCut vs NCut — why `L=D−A` is not the whole story

Minimising `fᵀLf` subject to `f ⟂ 1`, `‖f‖=1` is a relaxation of **RatioCut**
(balance by *number of nodes* per side). The normalised Laplacians
`L_sym = D^{-1/2} L D^{-1/2}` and `L_rw = D^{-1} L` relax **NCut** (balance by
*volume* = total degree per side). The practical difference: with the
unnormalised `L`, a few very high-degree nodes dominate the Fiedler vector, so
the embedding can fixate on the densest response channel rather than the shared
driver. von Luxburg's tutorial recommends normalised Laplacians by default, and
the SHREC paper itself lists "preconditioning the graph Laplacian" as the fix
for response bias. We did **not** change the operator (that would shift the
paper-faithful behaviour); MM33 just makes the degenerate end visible.

- **📚 LEARNING TOPIC — "RatioCut vs NCut, and degree bias."** Side-by-side
  Fiedler vectors of `L` vs `L_rw` on a graph with one inflated-degree cluster.
  Tie it back to the SHREC "response bias" remark — a concrete reason a user
  might want a normalised variant as an option.

## C. Modularity, CPM, and the resolution parameter (behind the igraph fix)

Leiden optimises a **quality function** over partitions. Two choices:

- **Modularity** compares within-community edge weight to what you'd expect in a
  degree-preserving random graph. It has a famous **resolution limit**: below a
  size set by the *total* graph weight, it cannot resolve small communities and
  *merges* them. This is precisely the MM20 xfail — the period-4 driver's four
  states collapse into two modularity communities.
- **CPM (Constant Potts Model)** compares within-community density to a fixed
  threshold `γ` (the resolution). No resolution limit, and `γ` has a direct
  reading: communities are denser than `γ`, sparser between. Sweeping `γ` traces
  the whole hierarchy from one blob (γ→0) to all singletons (γ→1).

The igraph bug hardcoded the resolution to 1.0, so `γ` was silently ignored — a
user sweeping resolution to escape the modularity resolution limit would see *no
change*. MM20 (period-4) is the standing motivation: the documented path to
closing it is a resolution/CPM sweep, which the bug would have quietly defeated.

- **📚 LEARNING TOPIC — "The modularity resolution limit."** The cleanest demo
  of why a hyperparameter-free method can still miss structure: a ring of
  cliques where modularity provably merges adjacent cliques once there are
  enough of them. Direct line to why SHREC's period-4 case is an xfail and what
  CPM buys you.

## D. Why property-based + metamorphic testing fits this codebase

Most SHREC operators have *no closed-form output* on real inputs — you can't
write down the "right" affinity matrix for a logistic ensemble. Two testing
philosophies handle that without a ground truth:

- **Property-based (Hypothesis).** Don't assert the output value; assert
  *properties that must hold for every input* (symmetry, [0,1] range, unit
  diagonal, residual-or-fallback) and let the engine search the input space —
  including shrinking any failure to a minimal counterexample. It explores the
  adversarial corners (duplicate points, collinear clouds, extreme scales) that
  a human fixture-writer would never enumerate. Our `_point_clouds` strategy is
  the reusable template: bound the values, `assume` away the genuinely
  ill-posed inputs, keep `max_examples` modest because each example runs the
  per-row solver.
- **Metamorphic.** Assert *relations between* outputs of related inputs:
  permutation invariance (already MM13), scale invariance (MM2), and — not yet
  built — monotone degradation (more noise ⇒ lower ARI) and the null-input
  property (no shared driver ⇒ no recovered structure). These encode the
  paper's qualitative claims as inequalities, which is often all the science
  actually asserts.

- **📚 LEARNING TOPIC — "Testing without an oracle."** A short methodology page:
  the ladder from exact oracle (MM1) → invariance/metamorphic (MM2, MM13) →
  reference-implementation differential (MM5) → statistical/surrogate (null
  driver). SHREC has a clean example at every rung — an unusually good teaching
  vehicle for scientific-software testing in general.
