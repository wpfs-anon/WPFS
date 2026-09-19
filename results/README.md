# Evaluation records

Each JSON is one seed of one configuration. `results.baseline` and
`results.corrector` both carry `per_episode`, a list of `{task, trial, ok}`, so
the two sides can be paired episode by episode rather than compared as rates.

| directory | configuration | seeds |
|---|---|---|
| `spnet/` | network student, lambda = 1 | 7 17 27 37 47 57 67 |
| `net120x7/` | network student, lambda = 0 (ablation) | 7 17 27 37 47 57 67 |
| `lora5/` | LoRA student `final30` | 7 17 27 37 47 |
| `ad_t1.0_s*.json` | teacher, full depth | 7 17 27 |
| `cf_base_s*.json` | the baseline runs those teacher rows pair against | 7 17 27 |

`main_results.html` is a self-contained page with the results table and eleven
side-by-side clips of episodes where exactly one side succeeded.
