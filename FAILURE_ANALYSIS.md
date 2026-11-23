# Failure analysis

Where and why the search methods fail, based on the blind ratings, the failure notes written while rating, and the RRF weight sweep.

Sources: `experiments/failures.csv` (27 notes, train 2026-11-15, test 2026-11-17), per-query scores in `results/train_metrics.json` and `results/test_metrics.json` (commit b4742f6), the weight sweep on branch `RRFWeightSweep` (`results/rrf_sweep/`), and two measurements from 2026-11-23 (commit 87b1228): hive counts per top 10 (`scripts/hive_counts/hive_counts.py`) and the left/right test (`results/flip_test_{train,test}.json`). All scores are AP@10 unless stated; "tuned RRF" is `rrf_tuned` (clip 0.5, SigLIP 2 0.75, motion 1, detector 1.5).

## 1. Summary

- **The detector cannot see the scene.** It counts bees and knows where they are, but knows nothing about hive colour, the pole, plants or sunlight. Scene queries fail or succeed by accident, depending on which hive happens to be busy.
- **SigLIP 2 is the scene expert.** It is the best method on general queries (test 0.86) and gets every pure sunlight query right (1.00), but tuned RRF buries those answers because the detector's vote is twice as strong.
- **No single set of weights serves both kinds of query.** Weighting SigLIP 2 up fixes scene queries and breaks count queries, and the reverse. The weights should depend on the query.
- **No method understands the kind of motion.** Taking off, landing and flying past all look the same; no method scores above 0.56 on the three take-off / landing test queries.
- **CLIP and SigLIP 2 cannot tell left from right** (measured: swapping "left" and "right" in the query leaves 7–9 of their 10 clips in place), so position comes almost entirely from the detector.
- **The hive shortcut is measurable.** The detector takes 99% of each test top 10 from one hive (all 10 from one hive on 43 of 44 queries), against 78% / 22% for the clips overall; the tuned fusions follow it (about 96%).

## 2. How the failures were collected

1. Every method's top 10 per query was pooled and rated blind (0 = not relevant, 1 = partly, 2 = clearly) without knowing which method found a clip.
2. Once a query was fully rated, the app's *Failure analysis* tab showed every method's top 10 side by side with the ratings, sorted worst first, and a note was written for the clear failures.

Notes per category:

| Category | Notes |
|---|---|
| flying or blurry bee | 8 |
| hive shortcut | 7 |
| other (incl. 3 summaries) | 4 |
| wrong side or position | 4 |
| lighting or shadow | 2 |
| wrong count | 1 |
| negation or comparison | 1 |

**Read the numbers with care:** the test split has 8 queries per single type and 2 per pair type, the train split 2 and 1. One query can move a type's score by 0.1–0.5.

## 3. Findings

### 3.1 The detector has no notion of background or scene

> "other than siglip2 all of the other models as well as their combination failed to recognise the backgrounds … Detector is understandable because it was trained only on flying and bees" (train note)

YOLO was trained on two classes, `bee` and `bee_flying`. A query about the hive, the plants, a pole or the light gives it nothing to work with, so it falls back to ranking by bee count, and bee count mostly says *which hive* a clip comes from.

| Query (split) | CLIP | SigLIP 2 | Detector | Tuned RRF |
|---|---|---|---|---|
| Hive with a red stripe and a square mesh grid (train) | 0.00 | **1.00** | 0.00 | 0.00 |
| Bees on the right side of a hive with a red stripe (train) | 0.55 | **0.87** | 0.00 | 0.00 |
| Bees flying above green leafy plants (train) | 0.43 | 0.29 | 0.05 | **0.76** |
| Bees on the left side near the pole (test) | 0.10 | **0.32** | 0.00 | 0.00 |

The detector also scores *well* on some scene queries, for the wrong reason: "A black pole next to the hive entrance" (1.00) and "Landing board in shade" (1.00) work because the hive with the pole and shade is also the busiest one. This is the hive shortcut: right answer, wrong reason, and it breaks as soon as another hive is busier. Section 3.8 puts a number on it.

### 3.2 SigLIP 2 is the scene expert, and fusion hides it

> "siglip however does very well on general queries due to its improved creation and understanding of the embeddings"

SigLIP 2 is the best method on general test queries (0.86 / 0.78 mAP / nDCG, against detector 0.75 / 0.69 and CLIP 0.57 / 0.60).

**Correction to the sunlight notes.** The test notes say "all models seem to be struggling on sunlight queries". That holds for *combined* sunlight queries, but not for pure scene ones, where SigLIP 2 is perfect and the fusion is what fails:

| Query (test) | CLIP | SigLIP 2 | Detector | RRF | Tuned RRF |
|---|---|---|---|---|---|
| Bees in bright sunlight | 0.64 | **1.00** | 0.03 | 0.85 | 0.06 |
| Sun patches on the grass below the hive | 0.22 | **1.00** | 0.03 | 0.21 | 0.04 |
| Bees on the left side of the board in sunlight | 0.02 | **0.61** | 0.00 | 0.02 | 0.00 |
| A few bees on the board in bright sunlight | 0.00 | **0.49** | 0.03 | 0.06 | 0.05 |
| Busy traffic at the entrance in bright sunlight | 0.04 | 0.10 | **0.79** | 0.20 | 0.50 |

- On the first two, plain RRF (equal votes) keeps most of SigLIP's answer (0.85); tuned RRF (detector 1.5 vs SigLIP 0.75) loses it (0.06).
- When sunlight is combined with a count or a side, every method is weak. SigLIP knows the light but not the side or the number; the detector knows the number but not the light.
- When the query is mostly about traffic, the detector carries it (0.79) and the light hardly matters.

### 3.3 The SigLIP 2 ↔ detector balance

> "The main balance to be found is between the weightage given to siglip and the detector to find the sweet spot"

The weight sweep changed one weight at a time:

- More detector weight helps until about 1.2–1.5, then levels off.
- More SigLIP 2 weight lowers the overall score.
- CLIP at weight 0 is slightly better than 0.5, which led to `rrf_tuned_no_clip`.
- The overall gains from the best single setting were small (test +0.025 mAP) because changes cancel out per query. With SigLIP 2 weighted up or the detector down, the two red-stripe train queries go from 0 to +1.00 / +0.88, but "Hive with an orange front and a white toothed entrance" drops by up to 0.95.

The new `rrf_siglip_boost` (no CLIP, SigLIP 2 raised from 0.75 to 1.0) shows the same trade on test (mAP / nDCG):

| Method | Count | General | Motion | Position |
|---|---|---|---|---|
| Tuned RRF, no CLIP | **0.77** / 0.71 | 0.76 / 0.70 | **0.70 / 0.61** | 0.71 / 0.66 |
| SigLIP-boosted RRF | 0.73 / 0.69 | **0.77 / 0.71** | 0.68 / 0.59 | 0.71 / 0.66 |

It gains a little on general and loses on count. A boost of 0.25 is far too small to rescue the four sunlight queries where SigLIP 2 leads in 3.2 (still 0.00–0.07), because the detector's 1.5 still dominates.

**Conclusion:** the "sweet spot" is not one set of weights. Scene queries need SigLIP 2 to lead; count, position and motion queries need the detector to lead. The weights should be chosen per query (see 5.1).

### 3.4 Position: CLIP and SigLIP 2 cannot tell left from right

> "clip and siglip are especially struggling with figuring out the position of the crowded bees. The detector is much better at it but the rrf then comes out to be dominated by the clip and siglip models" (train note)

Test position scores (mAP): CLIP 0.27, SigLIP 2 0.43, detector 0.70. With equal votes (plain RRF, 0.44), two weak methods outvote the one good one. Tuned RRF fixes this (0.67; 0.71 without CLIP).

**Measured with a left/right test** on the 10 queries naming a side (5 train, 5 test), with the 4 test queries about the middle of the board as a control. Each test asks a method the same thing twice and compares the answers: clips shared by the two top 10s, and ρ, how alike the order of all 1,440 clips is (1 = same order, below 0 = turned around).

- **Test A, swap the words** ("left" ↔ "right" in the query, same clips). A method that understands sides must change its list.
- **Test B, mirror the frames** (same query, every frame mirrored left to right; CLIP and SigLIP 2 embeddings recomputed, the detector's boxes mirrored in code). On middle queries a mirror changes nothing, so the list should stay.

| Method | Test A: clips shared of 10 (train / test) | Test A: ρ | Test B, middle queries: clips kept of 10 | Test B, side queries: clips kept of 10 |
|---|---|---|---|---|
| CLIP | 8.2 / 8.6 | 0.996 | 2.0 | 1.6 |
| SigLIP 2 | 7.4 / 9.2 | 0.996 | 2.2 | 1.8 / 2.4 |
| Detector | 0 / 0 | −0.26 | 10 | 0 |

- **CLIP and SigLIP 2 ignore the side word:** with "left" and "right" swapped they return almost the same list. Their position results come from other cues (how busy the board is, which hive).
- **The detector reads sides** from box positions: a completely different list when the side changes, and an unchanged one on middle queries under the mirror. This also shows the test works.
- **Mirroring alone reshuffles CLIP and SigLIP 2**, even on middle queries (about 2 of 10 kept, ρ about 0.73). The scenes are not symmetric (pole, entrance strip, light), and all clips score very close to each other, so a small shift reorders the top 10. For these two models test B measures this mirror sensitivity, not sides; side queries move no more than middle ones, which again says they do not read sides. Test A is the clean answer.
- The mirrored embeddings were checked against a local recompute (within 0.000001), so the reshuffling is not a pipeline bug.

| Query (train) | CLIP | SigLIP 2 | Detector | RRF | Tuned RRF |
|---|---|---|---|---|---|
| Many bees crowded on the left side of the board | 0.14 | 0.30 | **0.90** | 0.46 | 0.80 |
| Left side of the board empty | 0.53 | 0.32 | 0.87 | 0.75 | **1.00** |
| Bees walking on the left side of the board | 0.36 | 0.61 | **1.00** | 0.76 | 0.76 |

### 3.5 Kind of motion: taking off vs landing vs flying past

> "the detector fails to identify the difference between bees flying and landing/taking off of bees … creating a labelled dataset including taking off and landing bees would be helpful"

| Query (test) | CLIP | SigLIP 2 | Motion | Detector | Best |
|---|---|---|---|---|---|
| Bees taking off from the landing board | 0.38 | 0.36 | 0.19 | **0.48** | 0.48 |
| Bees landing on the board | 0.21 | 0.23 | **0.56** | 0.36 | 0.56 |
| A crowded board with many bees taking off | 0.20 | 0.07 | 0.10 | 0.01 | 0.45 (RRF) |

Why:
- **Detector:** its classes are `bee` (sitting) and `bee_flying`. A bee taking off and a bee landing look the same in a single frame; the difference is *direction over time*, which per-frame boxes do not record.
- **Motion method:** frame differences measure *how much* changes, not *what*. It cannot tell many flying bees from a few, and dense crowds of walking bees blur together (test note).
- **CLIP / SigLIP 2:** the 4 frames are averaged into one embedding, so any order in time is lost.

The training data explains the limit: the public dataset and our own frames only label `bee` and `bee_flying`.

### 3.6 Places the detector cannot read

The detector understands left / middle / right, above / below the entrance, and "spread" / "two groups". Any other place is ignored:

- "Bees on the left side near the pole" (test): the detector uses "left", ignores the pole, and scores 0.00. Tuned RRF follows it (0.00).
- "many bees next to the black pole" (typed in the app as count-position): no known place, so the detector does not apply at all.

### 3.7 CLIP sticks to one hive

> "clip is just showing the wrong hive" / "clip is still stuck on the red hive background" (train notes)

CLIP often fills its top 10 from one hive whatever the query asks ("Many bees on an orange hive with a white toothed entrance": 0.00; "Lots of bees flying in front of a busy entrance": 0.08). SigLIP 2 has the same habit on some queries ("siglip just gave hive 2 for 99% of its answers"; "Hive with an orange front…": 0.03), but less often. Together with the sweep result (3.3), this is why `rrf_tuned_no_clip` drops CLIP.

Measured (3.8): on train, 95% of CLIP's top 10 comes from one hive on average, and all 10 on 12 of 14 queries (SigLIP 2: 83%, 4 of 14). On test the two are level (88%, 23 of 44 each).

### 3.8 The hive shortcut, measured

For each query and method, the number of top-10 clips from each hive (`rec2123` = the orange hive, `rec2125` = the red-stripe hive). The clips split 1,125 orange / 315 red-stripe (78% / 22%), so a method picking in that ratio shows no shortcut; one that fills its list from one hive, query after query, does.

| Method | Majority-hive share, train | All 10 from one hive, train (of 14) | Majority-hive share, test | All 10 from one hive, test (of 44) |
|---|---|---|---|---|
| Detector | 99% | 13 | 99% | 43 |
| Tuned RRF, no CLIP | 87% | 10 | 96% | 33 |
| SigLIP-boosted RRF | 84% | 8 | 96% | 34 |
| Tuned RRF | 85% | 9 | 92% | 30 |
| SigLIP 2 | 83% | 4 | 88% | 23 |
| CLIP | 95% | 12 | 88% | 23 |
| RRF | 83% | 4 | 81% | 13 |
| Motion (motion queries only) | 84% | 1 of 5 | 86% | 4 of 14 |

- The detector almost never mixes hives: its counts mostly tell the two hives apart (3.1).
- The tuned fusions inherit this, because the detector has the strongest weight; plain RRF mixes hives the most.
- Every method stays above the 78% that random picking would give, so the hive is a strong cue for all of them.
- A high share is not always wrong: some queries name one hive ("hive with a red stripe"). The share says how often a method *can* only be right through the hive, not that it is wrong.

## 4. Limitations of this analysis

- **Few queries:** 44 test queries over 10 types; pair types have 2 test queries each. Single-query swings dominate the pair scores.
- **Not all methods are blind on test:** `rrf_tuned_no_clip` and `rrf_siglip_boost` were added after the first test scoring, so their test numbers are not a clean hold-out check.
- **One rater:** all ratings and notes come from one person.
- **Left/right test without ratings:** the rated version of test B (judging the mirrored top 10s with ratings, 355 new ratings) was not run; the conclusion rests on test A and the middle-query control.
- **Detector domain gap:** YOLO was trained on landscape images; the clips are portrait. It also mistakes the blurry mesh above some entrances for bees, which inflates "above the entrance" counts.

## 5. Future work

### 5.1 Weights that depend on the query
Instead of one fixed set, pick the weights from the query: scene words (sunlight, shade, colour, pole, plants) → SigLIP 2 leads; count, position and motion words → the detector leads. The query type is already known at search time, and the detector already reads words like "left" or "above", so this is a small step. It should be tuned on train queries only, to keep test honest.

### 5.2 Take-off and landing labels, or tracking
- Label a dataset with `taking_off` and `landing` bees (as suggested in the notes) and retrain YOLO, or
- track bees across the 4 frames (e.g. Ultralytics' built-in tracker on the same YOLO model) and use the direction of movement: away from the board = taking off, towards it = landing. This needs no new labels, but only works if 4 frames are close enough in time to link boxes.

### 5.3 Segmentation as an extra RRF signal
- **SAM 3** takes text prompts ("pole", "sunlit patch", "bee"), so it could ground scene words the detector cannot read and give exact bee areas. It is large and slow on CPU; it would have to be run offline per clip, like the detector, with only text matching at query time.
- **FastSAM** is much faster but class-agnostic: it outlines objects without naming them. It would need CLIP or SigLIP 2 to name each mask, which brings back their weaknesses.
- Either one must be measured against today's latency (tuned RRF without CLIP: 68 ms p50) and added to the RRF like any other method.
