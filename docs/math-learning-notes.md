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
