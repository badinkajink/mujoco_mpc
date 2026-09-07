# Writing

Everything I write for this user — pages, reports, commit messages, log entries, terminal
prose — is read as if a researcher wrote it. The reference is their own paper,
`paper/hand_iros26-4.pdf` ("A Self-Reconfigurable Hand for Sim-to-Real Verification of
Optimized Morphologies"): plain declarative sentences, the technical noun in the subject
position, numbers with units, and the finding stated before the story about it.

## Titles and filenames

A title is a **descriptive noun phrase naming the subject**, not an image, not a phrase, not a
line of poetry. "Wrist geometry and grasp preload on the UR5e bench", not "A Hundred
Millimetres of Wrist". "Screwdriver reorientation: two finger arrangements compared" is right;
"Never Letting Go", "The Seam and the Seat", "A Spring, a Cone, and a Cliff" are all wrong and
have been called out. No colons carrying a cute half. No definite-article openers ("The
Something and the Something"). If the title would work as a section heading in an IROS paper,
it is fine.

**Date-prefix the filename**, not just the folder: `20260904-real_v1_bench.html`, never
`page.html`. Files must sort chronologically in a flat listing.

## Anti-patterns to delete on sight

These are the tells that make writing read as machine-generated. Each one has been flagged.

* **The rhetorical inversion.** "It is not X, it is Y." "The question is not whether … but how."
  "X is not a detail." One per document is already too many; write the positive claim.
* **The triad.** Three parallel clauses where two would do, especially escalating ones
  ("it centres, it wedges, it constrains").
* **The portentous short sentence** as a paragraph ending. "That is the whole mechanism."
  "This is the finding." "And it is not a tuning problem." Cut it; the reader can tell.
* **Restating the result as a moral.** After a table, do not tell the reader what the table
  means in an elevated register. Say what happens next.
* **"Worth noting", "importantly", "crucially", "notably", "it turns out", "the key insight",
  "fundamentally", "elegant", "powerful", "robust" as praise, "leverage" as a verb.**
* **Em-dash asides stacked two or three to a paragraph.** One per paragraph, at most.
* **Bold applied to a whole sentence for emphasis.** Bold a term or a number, not a claim.
* **A closing recap section** that repeats what the document already said.
* **Hedging stacks**: "may potentially", "could possibly suggest".
* **Praise of the user's idea** before answering it. Answer, then note agreement if relevant.

## Structure

Lead with the number or the outcome. Put the mechanism second and the caveat third. Negative
results are stated flatly in the same voice as positive ones — no softening, no framing them as
"informative". A "what this does not settle" section lists **actionable items with the specific
next measurement named**, not a mood of humility: say which flag, which script, which sweep,
and what would falsify the claim.

# Artifacts are never the only copy

Every artifact I publish must exist as a file in the user's repo first, and I must give the
**local path alongside the URL, every time**, in the same sentence. The user is actively angry
about information siloed on claude.ai. Keep `docs/experiments/INDEX.md` current: one row per
page with its local file, its artifact URL, and its date. When updating a published page, edit
the local file and republish to the same URL — never create a second artifact for the same work.