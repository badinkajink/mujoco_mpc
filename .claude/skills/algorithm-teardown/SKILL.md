---
name: algorithm-teardown
description: Write or audit a didactic teardown of an algorithm implemented in this repo - a document that binds the published math to the actual source lines to measured behaviour, so a reader can follow one equation from paper to code to observed result. Use when documenting a planner/estimator/task implementation, when porting an algorithm from a paper, when asked to explain "how X works" at the level of both math and code, or when an existing algorithm doc has drifted from the code.
---

# Algorithm teardown

A teardown answers one question for a reader who knows neither the paper nor the
code: **which line does this equation, and what does that buy me?**

Three layers, always bound together:

| Layer | Content | Anchored by |
|---|---|---|
| **Math** | the equations as the source states them, with the source's own numbering | a citation (paper section / equation number) |
| **Code** | the lines that evaluate those equations | `file.cc:LINE` for every equation |
| **Result** | what the code measurably does | a number you produced by running something |

A document with math and code but no measured results is a *plan*, not a
teardown. A document with code that was never verified against the file is
*fiction*. Both failure modes are common and both are worse than nothing,
because a reader trusts a teardown by default.

## Procedure

### 1. Read the primary source first

Read the paper (or spec) before the code, and take down its **own** numbering:
equation numbers, algorithm numbers, section numbers. The teardown must use the
source's labels, not invented ones — the reader will have the paper open.

Record the source's stated results too (timings, condition numbers, success
rates). You will check them.

### 2. Read the implementation and build the anchor table

For every equation that the code evaluates, find the line. Produce a table:

```markdown
| Paper | Quantity | Code |
|---|---|---|
| Eq 21 | `K(t) = R⁻¹B(t)ᵀP(t)` | [treevertex.cpp:102](path#L102) |
| Eq 22 | Riccati for `P(t)` | [treevertex.cpp:90](path#L90) |
```

Rules:
- **Verify every line number by reading the file.** Never carry a line number
  over from an older doc, a commit message, or memory.
- Prefer a permalink-style relative path so the anchor is clickable.
- If an equation has *no* corresponding line, say so explicitly — that is
  usually the most interesting fact in the document (the implementation
  silently dropped a term, or the paper describes something never built).
- If a line implements something the paper does **not** say, flag it as a
  deviation and judge whether it is a bug or a deliberate improvement.

### 3. Restate the math in the implementation's own setting

Papers are usually written in continuous time with exact matrix inverses. Code
is discrete, finite-precision, and uses a specific linear algebra API. Give the
reader the *translated* equations next to the originals, because that gap is
where all the real bugs live. For example:

```
Eq 15 (paper, continuous):  Ẇ_K = A_K W_K + W_K A_Kᵀ + B R⁻¹ Bᵀ,  W_K(0) = 0
      (as implemented, discrete):  W_{k+1} = A_{K,k} W_k A_{K,k}ᵀ + B_k R⁻¹ B_kᵀ
```

State the discretization, the integrator, and the numerical method used for any
inverse or factorization.

### 4. Measure something

Every performance, accuracy, or feasibility claim needs a number **you
produced**. Run the code. Minimum bar:

- **Cost of the inner loop** — time the operation that dominates, and say how
  many times per second it must run to be viable in this system.
- **The claim under test** — pick the source's central numerical assertion and
  reproduce it. Report agreement *and* disagreement.
- **End-to-end behaviour** — what the algorithm actually achieves on a task,
  using a task-level metric (did it reach the goal?) rather than only an
  aggregate loss.

Put the exact command next to each number so the reader can re-run it.

### 5. Write the verdict before the details

Lead with the conclusion a reader needs in order to decide whether to keep
reading: does this work here, at what cost, and what breaks. Bury nothing.

## Required structure

```markdown
# <Algorithm> in <system>

## Verdict            <- 5-15 lines. Works / doesn't / works only if X. Numbers.
## What problem it solves   <- the gap in the existing system this closes
## The algorithm       <- math, in the source's numbering, restated for this setting
## Equation-to-code map <- the anchor table from step 2
## Walkthrough         <- one iteration end to end, following real data
## Measured behaviour  <- step 4's numbers, with commands
## Deviations from the paper  <- including suspected bugs, with severity
## Failure modes       <- when it breaks and what the symptom looks like
## Parameters          <- each one: what it does, sane range, how to tell it is wrong
## References
```

Drop a section only if it is genuinely empty, and say why (e.g. "No deviations:
the implementation follows Algorithm 4 exactly").

## The walkthrough section

This is what makes a teardown didactic rather than a reference table. Take
**one** iteration and follow concrete data through it: state the input, name
each transformation with its equation and its line, and give the shape and
meaning of the intermediate at each step. Prefer real numbers from an
instrumented run over symbolic placeholders.

```
x0 = (0,0,0,0)  upright, cart at origin
  ↓ Eq 3, treevertex.cpp:56 -- integrate zero-control dynamics
x_zero(t), 200 samples over [0, 1.0]s   (drift over the horizon: 0.0 -- x0 is an equilibrium)
  ↓ Eq 4, treevertex.cpp:76 -- Jacobians at each sample
A(t) 8x8, B(t) 8x1
  ↓ ...
```

## Anti-patterns

**Aspirational code.** Never paste code that does not exist in the file, in the
form it exists. If you are documenting a design not yet built, the title says
"design" or "plan" and no line numbers appear anywhere.

**Line numbers as decoration.** A wrong line number is worse than none: it
teaches the reader to distrust every anchor. Re-read the file.

**Restating the paper.** If a paragraph would be equally true without this
repository, cut it. The value of a teardown is entirely in the binding.

**Unfalsifiable comparison tables.** "Convergence: Fast / Medium / Slow" with no
measurement is noise. Either measure it or delete the row.

**Burying the failure.** If the algorithm does not work here, that goes in the
Verdict, in the first three lines.

## Auditing an existing doc

When asked to standardize or fix a drifted teardown:

1. Check every line number against the current file. Report the drift rate —
   it calibrates how much of the rest to trust.
2. Check every code block against the real source; mark aspirational ones.
3. Check every quantitative claim for a reproducing command. Claims without one
   are hypotheses — relabel them as such or measure them.
4. Only then restructure. Restructuring an unverified document just makes wrong
   information easier to find.
