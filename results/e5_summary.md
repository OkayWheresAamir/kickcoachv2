# E5: LLM-assisted norm extraction vs human verification

Extractor: `tools/extract_norms.py` (prompt norms-v1, `gemini-3.6-flash`, one call per paper, structured output against the D12 schema). Verification: Aamir, in-session, against the papers (`knowledge/norms.csv`, column `review`). Fields scored per row: 16 (variable, kick_type, condition, population_level, population_age_group, population_sex, stat_mean, stat_sd, stat_min, stat_max, unit, n, method, angle_convention, page_or_table, primary_source).

| metric | value |
|---|---|
| rows extracted | 32 (from 4 papers; 2 control papers gave 0) |
| hallucinated rows (rejected, or from a control paper) | 0 of 32 |
| rows with every field right | 32 of 32 |
| field-level accuracy | 512 of 512 fields (100%) |
| recall (supported / supported + missed) | 32 of 34 (94%) |
| QA-flagged rows that were wrong | 0 of 6 |

## By paper

| paper | role | rows | all fields right | hallucinated |
|---|---|---|---|---|
| kellis_katis_2007 | source | 18 | 18 | 0 |
| scurr_hall_2009 | source | 12 | 12 | 0 |
| kapidzic_2014 | source | 1 | 1 | 0 |
| xu_2025 | source | 1 | 1 | 0 |
| harrison_mannering_2006 | control | 0 | – | 0 |
| nunome_ikegami_2006 | control | 0 | – | 0 |

## Missed rows

| paper | where | variable | quote | decision |
|---|---|---|---|---|
| kellis_katis_2007 | p. 162 | peak_knee_extension_velocity | Maximum knee angular velocities during the kick range from 1014 deg·s -1 for 4.6 year-old children (Capranica et al., 1992) | missed; not added to norms.csv (the text contradicts the paper's own Table 4: 1014 at 4.4 yr from Elliott et al. 1980) (Claude, on Aamir's delegation (2026-11-02)) |
| kellis_katis_2007 | p. 162 | peak_knee_extension_velocity | to 1204 deg·s -1 for 14 year-old players (Table 4) | missed; not added to norms.csv (cites Table 4 but the value is not in Table 4) (Claude, on Aamir's delegation (2026-11-02)) |

## Caveats

- **Small n:** 4 source papers, 2 controls, one extraction run with one model. The result says this extractor didn't invent rows here, not that it never does.
- **Verification was answered per table group** (7 groups for 32 rows), each with every field shown. A group answer of "all correct" covers every field of every row in it.
- **The QA flags caught layout, not errors:** no flagged row was wrong; the flags came from quotes rebuilt out of table columns that pypdf extracted on separate lines. A flag means "check the table".
- **Correct ≠ usable:** some verified rows can't be compared with the pipeline (cm vs leg lengths, a suggestion rather than a measurement). That is the comparison engine's job (`kickcoach/compare/engine.py`), not E5's.
