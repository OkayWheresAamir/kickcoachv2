# E7: feedback quality: human, LLM-rater and LLM-judge ratings

Items: every fault item from the 15 accepted clips with a fault, in three variants: LLM with retrieved passages (prompt v3, the app's default), LLM without (v2), and the template. Generator: `gemini-3.5-flash-lite`; judge: `gemini-3.6-flash` (prompt judge-v1). Rubric 1–5 on correctness, specificity, actionability (`results/e7/items.json`). Aamir rates a fixed sample of 20 without seeing the variant.

## Validator, first try

Counted per item; one LLM call writes all of a clip's items, so one rejected call fails every item of that clip.

| variant | items | LLM answered | first-try pass | fell back to template |
|---|---|---|---|---|
| llm_rag | 18 | 18 | 15 of 18 | 0 |
| llm | 18 | 18 | 18 of 18 | 0 |
| template | 18 | 0 | – | – |

Retrieved passages: 3.2 per item on average; 15 of 18 RAG items cite at least one.

## Agreement between Aamir and the judge (0 items)

**No human rating exists.** The sample wasn't rated by a human; an AI persona panel was supplied instead (7 Nov). So the bar above can't be applied, the judge is not validated, and it is not used beyond the sample (DEC-040, DEC-041). The section below compares LLM raters with each other, which shows consistency, not correctness.

## LLM raters on the 20-item sample (DEC-041)

None of these is a human. Claude's ratings follow written rules with a reason per item (`results/e7/raters/claude.json`); the persona panel came from one external AI tool playing five players and a coach (`results/e7/raters/persona_panel.csv`).

Mean rating (correctness / specificity / actionability), by variant:

| rater | llm_rag | llm | template |
|---|---|---|---|
| Claude (rule-based, not blind) | 4.2 / 3.1 / 4.4 (n 8) | 2.6 / 4.0 / 3.8 (n 8) | 4.2 / 4.2 / 5.0 (n 4) |
| AI persona panel (median of 5) | 3.2 / 3.6 / 4.2 (n 8) | 2.5 / 3.4 / 3.5 (n 8) | 3.2 / 3.5 / 4.5 (n 4) |
| gemini-3.6-flash | 5.0 / 4.0 / 4.6 (n 8) | 5.0 / 4.0 / 3.5 (n 8) | 5.0 / 5.0 / 5.0 (n 4) |
| gemma-4-31b-it | 5.0 / 5.0 / 4.5 (n 4) | 5.0 / 5.0 / 4.2 (n 4) | 5.0 / 5.0 / 5.0 (n 2) |

With passages minus without, on the same clip and fault (positive = passages rated higher):

| rater | pairs | correctness | specificity | actionability |
|---|---|---|---|---|
| Claude (rule-based, not blind) | 8 | +1.62 (better 7, worse 1) | -0.88 (better 0, worse 7) | +0.62 (better 5, worse 0) |
| AI persona panel (median of 5) | 8 | +0.75 (better 6, worse 0) | +0.25 (better 3, worse 1) | +0.75 (better 7, worse 1) |
| gemini-3.6-flash | 8 | +0.00 (better 0, worse 0) | +0.00 (better 2, worse 2) | +1.12 (better 7, worse 0) |
| gemma-4-31b-it | 4 | +0.00 (better 0, worse 0) | +0.00 (better 0, worse 0) | +0.25 (better 1, worse 0) |

Agreement between raters (weighted κ):

| raters | items | correctness | specificity | actionability |
|---|---|---|---|---|
| Claude (rule-based, not blind) vs AI persona panel (median of 5) | 20 | 0.48 | 0.00 | 0.67 |
| Claude (rule-based, not blind) vs gemini-3.6-flash | 20 | 0.00 | 0.35 | 0.74 |
| Claude (rule-based, not blind) vs gemma-4-31b-it | 10 | 0.00 | 0.00 | 0.56 |
| AI persona panel (median of 5) vs gemini-3.6-flash | 20 | 0.00 | 0.09 | 0.72 |
| AI persona panel (median of 5) vs gemma-4-31b-it | 10 | 0.00 | 0.00 | 0.45 |
| gemini-3.6-flash vs gemma-4-31b-it | 10 | – | 0.00 | 0.55 |

## The two judges compared (22 items both rated)

`gemini-3.6-flash` is the judge; `gemma-4-31b-it` stood in while Flash's free quota (20 requests a day) was used up (DEC-040). Where both rated:

| criterion | weighted κ | exact | within 1 | mean gemini-3.6-flash | mean gemma-4-31b-it |
|---|---|---|---|---|---|
| correctness | 0.65 | 95% | 100% | 4.91 | 4.95 |
| specificity | 0.00 | 41% | 100% | 4.41 | 5.00 |
| actionability | 0.60 | 59% | 100% | 4.23 | 4.64 |

## Caveats

- **No human rated:** every score here comes from an LLM, including Claude's, and Claude also built the pipeline being rated. Agreement between LLM raters is consistency, not validity.
- **Small n:** 20 items, 8 pairs.
- **Blinding is partial:** the variant is hidden, but each fault's template has fixed wording that repeats across clips.
- **No item for the shank-before-thigh rule:** it never fires on these clips.
