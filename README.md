# AccessSwitching: an organ that keeps two memories through damage and lets them take turns

A growing active-inference organ (a Python port of Palacios, Razi, Parr, Kirchhoff and Friston 2020, grown
from one cell; see [Embryogenesis](https://github.com/dawsonzhou225/Embryogenesis)) that stores two bits and a
position in the phase waves of its tissues' clocks, keeps them through repeated cuts of most of its tissues,
and gives one bit at a time access to what its tissues express.  The last part was built after Hesse et al.
(2026), who found that the same IT and PFC neurons switch between the seen and the suppressed face about
15 times a second.

![Two cuts of 12 of 16 tissues](figures/cuts.gif)

## The model in a few lines

- Every agent (cell, blanket, organ) runs the same ADEM step on one free energy (`spm.py`, `model.py`'s
  `adem_step`); the organ is 16 blankets of 16 cells, grown from one cell and repaired by the same rules.
- Each blanket believes it is one of two forms, A (1, 6, 9) or B (1, 3, 12); its cells' fingerprint, its
  birth target and its radii follow the belief.
- Each tissue (blanket) has a clock that sets its emission gain, 1 + 0.3 sin(phase).  The tissues near the
  core are the inner ring, the rest the outer; each ring's clocks couple by distance, so each ring carries a
  wave, phase = offset + winding x angle about the core.
- The bits: windings written at 150 (inner -1, outer +1), read back into each tissue's form, voted ring by
  ring; each tissue keeps a backup of the other ring's bit.  The position: a slow 2D walk added to each ring's
  clocks, so each wave's offset integrates the path; the two offsets are a point on a torus that the walk
  moves, as in a grid-cell module (Gardner et al. 2022).
- Repair: a new tissue's clock starts on its ring's wave; the core keeps a copy of each wave (a phase-locked
  loop) that holds a ring while it regrows; the coupling only moves phase within a ring; a tissue that moves
  round the core takes on its wave's phase where it arrives.
- Turn-taking (Hohwy, Roepstorff & Friston 2008): one bit at a time has access and is expressed on the
  tissues' emission as a pattern over the organ; the other builds up error, scaled by how clearly its ring
  shows it, until it takes over.  The threshold can follow the core's clock.

What is built in, not emergent: the forms' targets, the ring split, the write window, the walk, the repair
rules and the turn-taking rule.  What the runs test is whether they hold together in a growing, self-repairing
organ, and what the turn-taking looks like from inside it.

## Files

| file | what |
|---|---|
| `model.py` | the model: layout, process `Gg`, model `Mg`, `Fixed`, the ADEM step, `run(seed, nY, cuts, turns, pace, core_T)` |
| `spm.py` | the seven SPM primitives the step needs |
| `run.py` | run seeds in parallel, one JSON per run, every step traced |
| `fig.py` | the figures: `turns`, `rhythm`, `position`, `levelt` |
| `gif.py` | a GIF of the organ: cells by kind, tissues by form and gain, which memory has access |
| `run_all.sh` | the runs behind the figures, then the figures |
| `check_same.py` | are two run JSONs the same run, to the last bit? |
| `figures/` | the figures below, from the runs the code was checked against |

Requirements: numpy, scipy, matplotlib (Pillow for the GIF).  A 720-step run takes ~1.3 h (three at once).

## Running

    python3 run.py run --name turns_x3 --cut-at 420 480 540 --cut-n 12 --seeds 0-2 --cores 3 --out results
    python3 run.py run --name inner_x3 --cut-at 420 480 540 --cut-n 6 --cut-what inner --seeds 0-2 --cores 3
    python3 run.py run --name free --pace 0                # turn-taking not paced by the core's clock
    python3 run.py run --name noturns --no-turns           # the memories alone
    python3 fig.py turns results/turns_x3_s0.json results/turns_x3_s1.json --out turns.png

The schedule (`AT` in `model.py`): clocks from 150, the bits written until 180, votes and backups from 250,
turn-taking from 300, the walk from 380.  Cuts remove whole blankets with their cells, never the organ's core.

## What it found (seeds 0-2, 720 steps, cuts of 12 of 15 tissues at 420, 480 and 540)

| | result |
|---|---|
| the two bits | held in every run: random cuts, the whole inner ring cut three times, with and without turn-taking |
| the position | error at 719, after three cuts while walking: at most 0.001 cycles |
| which bit has access | read back from what the tissues sense, step by step: 97.6% right; averaged over time, each bit looks half there |
| the rhythm (seed 0, no cuts; run before the last two changes, `run_all.sh` remakes it) | each bit stays on ~8 steps whatever the core's period (10, 17, 25); the switches lock to the core's phase best when one fits a cycle (R 0.80, 0.63, 0.40; not paced 0.03) |
| a weakened bit | cutting only the inner ring: an epoch lasts as (strength of the bit that is off)^-1.00 +- 0.15 and (its own)^-0.02 +- 0.17, Levelt's second law |

![Turn-taking](figures/turns.png)
![Position](figures/position.png)
![Levelt](figures/levelt.png)

In the position figure the spikes during regrowth are a ring too sparse to read; the core's copy carries the
position then.  The step-by-step readout is the analysis of Hesse et al. in miniature: averaged over time, the bit without
access looks weakly present, as if it failed to ignite; read step by step, it is fully there in its own epochs.

## Reproducibility

Runs are deterministic: every random draw comes from the seed.  `check_same.py a.json b.json` compares two
runs trajectory by trajectory with no tolerance.  This folder was cut from a larger research code base and
checked against it, run against run, with the turn-taking on and off, random and inner-ring cuts, and the
core's period changed; across machines the free energy's running sum differs in its 13th digit.

## References

Palacios, E. R., Razi, A., Parr, T., Kirchhoff, M., & Friston, K. (2020). On Markov blankets and hierarchical
self-organisation. *Journal of Theoretical Biology*, 486, 110089.

Hohwy, J., Roepstorff, A., & Friston, K. (2008). Predictive coding explains binocular rivalry: an
epistemological review. *Cognition*, 108(3), 687-701.

Gardner, R. J., Hermansen, E., Pachitariu, M., Burak, Y., Baas, N. A., Dunn, B. A., Moser, M.-B., & Moser, E. I.
(2022). Toroidal topology of population activity in grid cells. *Nature*, 602(7895), 123-128.

Hesse, J. K., Lanfranchi, F. F., McGrory, C., Shi, Y., Engel, T. A., & Tsao, D. Y. (2026). Rapid alternation of
conscious and suppressed stimulus representations in the primate brain. *bioRxiv*, doi:10.64898/2026.10.15.759299.
