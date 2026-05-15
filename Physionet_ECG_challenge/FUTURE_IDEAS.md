# Future Ideas

This file is a backlog for later experiments.

It should stay broader than one notebook or one method, but it can contain focused sections when a topic becomes important.

Current stopping point:
- the current hybrid Hough border-line selector is good enough to pause here
- current qualitative status:
  - about `9 / 10` on the fixed random set

## Hough Boundary

### Current Status

The current Hough boundary path now has three selectors:
- `global`
- `score`
- `hybrid`

Current practical result:
- `hybrid` is the best place to stop for now
- the main remaining failure mode is the monitor case:
  - two close parallel edges from the monitor border / bezel
  - the selector can keep the upper edge when the lower edge is the true target

### Known Failure Modes

#### Monitor double-edge ambiguity

Typical pattern:
- a laptop or monitor border creates two nearby horizontal lines
- the lower border is visually the right page limit
- the current selector sometimes keeps the upper one

Why this is different from the earlier threshold problem:
- theta family is usually correct
- the failure is now local line disambiguation inside one family
- the current `global / score / hybrid` logic is still too coarse for that case

#### Strong inner line beats true outer line

Typical pattern:
- a true border exists
- a stronger inner grid/content line exists nearby
- score-based pair selection prefers the inner line because of accumulator strength

This is mostly improved, but still important as a recurring risk.

### Immediate Ideas To Try Later

#### 1. Local refinement around the chosen boundary line

After `global`, `score`, or `hybrid` selects a boundary line:
- search a small `rho` window around that line
- inspect nearby parallel candidates
- choose the final line with a local rule

Why:
- this directly targets the monitor double-edge case
- it avoids redesigning the whole selector

Possible local features:
- local edge strength
- local continuity along the line
- local support length
- local gap statistics

#### 2. Neighbor-line feature

Add an explicit feature for:
- existence of a nearby parallel line
- distance to the nearest nearby parallel line
- asymmetry of support between the two

Why:
- a monitor border often appears as a close pair
- this feature can help distinguish a true outer border from a bezel double-edge or grid duplication

Possible use:
- penalty if a selected line has a very close stronger neighbor on the ECG side
- bonus if the selected line is the outer member of a close double-edge pair

#### 3. Better energy field for Hough voting

Instead of using only the current Canny-style edge map, test richer energy maps.

Ideas:
- gradient magnitude weighted voting
- signed edge preference by direction
- edge map that suppresses text/logo-like structures
- edge map that favors long straight supports over small sharp clutter

Why:
- the Hough stage is only as good as the energy that feeds it
- monitor logos and bezel details can distort voting

#### 4. Line score beyond raw accumulator

Current score ingredients already explored:
- accumulator
- pair separation
- outerness

Ideas to add:
- line continuity
- support coverage along the segment
- edge density on one side vs the other
- local prominence in rho profile
- local double-edge pattern score

Why:
- raw accumulator is often too favorable to strong inner lines

#### 5. Side-aware rules

Do not use exactly the same local rule for all four borders.

Examples:
- bottom monitor edge may need a different rule than top paper edge
- left/right may be more stable under current Hough logic than top/bottom

Why:
- the failure modes are not fully symmetric across image sides

#### 6. Rectangle-level chooser after line-level chooser

Current hybrid chooses dominant and perpendicular families independently.

Later idea:
- generate a few rectangle candidates
- score the full rectangle, not only each family separately

Possible rectangle criteria:
- corner plausibility
- content occupancy inside rectangle
- expected ECG layout consistency
- penalty for bezel/text strips near one side

Why:
- some bad line choices only become clearly wrong once the full rectangle is seen

### Hough-Specific Experiments

#### Weighted Hough revisited

Try again later with:
- better energy map
- local refinement after selection

Reason:
- weighted voting alone did not solve the selector problem
- but it may become useful with a better edge field

#### Multi-scale Hough

Try:
- coarse Hough for orientation
- finer local rho search for final boundary placement

Reason:
- the current pipeline already separates theta and rho conceptually
- multi-scale refinement is a natural extension

#### Family-specific rho profile refinement

For each chosen family:
- build rho profile
- detect local maxima
- refine the chosen line with a local side-aware rule

Reason:
- this is a direct continuation of the current score method

### CNN Ideas

#### 1. CNN border probability map

Train a small model to predict:
- page/border probability
- or four side-specific border maps

Then use Hough only as a geometric regularizer on top of the CNN map.

Why:
- Hough is strong for geometry
- CNN can learn monitor/logo/background patterns that hand-crafted energy maps miss

#### 2. CNN ranking of candidate rectangles

Keep the current classical candidate generation:
- global
- score
- hybrid
- maybe a few local variants

Then train a small CNN or MLP to rank candidate rectangles.

Why:
- lower engineering risk than replacing the whole detector
- preserves interpretability of the current classical pipeline

#### 3. CNN for local edge disambiguation only

Very targeted idea:
- crop a strip around a selected boundary line
- predict whether the correct border is the current line, a nearby upper line, or a nearby lower line

Why:
- directly targets the remaining monitor double-edge failure
- much smaller learning problem than full page detection

### Useful Evaluation Rules

When returning to this backlog:
- keep the fixed random sample set unchanged first
- keep the known regression cases visible:
  - `19030958 / 0009`
  - `10140238 / 0012`
- evaluate changes on:
  - the fixed random set
  - the known monitor failure cases
- change one idea at a time

### Recommended Next Entry Point

Best next experiment when resuming:
- local refinement around the hybrid-selected line
- start with the bottom monitor edge case
- use nearby parallel-line features before adding a CNN
