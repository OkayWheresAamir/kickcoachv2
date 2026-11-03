# E7: feedback quality, rated by Aamir and an LLM judge

Items: every fault item from the 15 accepted clips with a fault, in three variants: LLM with retrieved passages (prompt v3, the app's default), LLM without (v2), and the template. Generator: `gemini-3.5-flash-lite`; judge: `gemma-4-31b-it` (prompt judge-v1). Rubric 1–5 on correctness, specificity, actionability (`results/e7/items.json`). Aamir rates a fixed sample of 20 without seeing the variant.

## Validator, first try

Counted per item; one LLM call writes all of a clip's items, so one rejected call fails every item of that clip.

| variant | items | LLM answered | first-try pass | fell back to template |
|---|---|---|---|---|
| llm_rag | 18 | 18 | 15 of 18 | 0 |
| llm | 18 | 18 | 18 of 18 | 0 |
| template | 18 | 0 | – | – |

Retrieved passages: 3.2 per item on average; 15 of 18 RAG items cite at least one.

## Agreement between Aamir and the judge (0 items)

Not computed yet: needs Aamir's ratings and the judge's.

## The two judges compared (21 items both rated)

`gemini-3.6-flash` was the planned judge; its free tier stopped at 20 requests a day, so `gemma-4-31b-it` judged everything (DEC-040). Where both rated:

| criterion | weighted κ | exact | within 1 | mean gemma-4-31b-it | mean gemini-3.6-flash |
|---|---|---|---|---|---|
| correctness | 0.64 | 95% | 100% | 4.95 | 4.90 |
| specificity | 0.00 | 38% | 100% | 5.00 | 4.38 |
| actionability | 0.59 | 57% | 100% | 4.62 | 4.19 |

## Caveats

- **One rater, small n:** Aamir isn't a coach, and the sample is 20 items; the agreement is with him, not with coaching truth.
- **Blinding is partial:** the variant is hidden, but each fault's template has fixed wording that repeats across clips.
- **No item for the shank-before-thigh rule:** it never fires on these clips.
