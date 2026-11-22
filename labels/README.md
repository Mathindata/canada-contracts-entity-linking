# Labeling guide

`labeling_sheet.csv` holds pairs of vendor spellings. For each pair, decide:
**is this the same organization?**

Run `python -m cel label` to answer pair by pair in the terminal (it saves after
every answer, so stop and resume any time), or fill the `label` column in a
spreadsheet.

| Answer | Write | When |
|---|---|---|
| Same vendor | `1` (or `y`) | Same company despite spelling, suffix, language, accents, typos, abbreviations or an acronym. Different offices or branches of one company count as the same vendor. |
| Different | `0` (or `n`) | Different companies, even when the names look alike (`SUMMIT DATA SOLUTIONS` vs `SUMMIT DATA SYSTEMS`). A parent company and its separately named subsidiary are different. |
| Unsure | `?` (or `u`) | You cannot tell from the names, postal prefixes, departments and work. These are excluded from the metrics and counted separately. |

Rules of thumb:
- The postal prefix (first 3 characters) points to a city area. Same name, far-apart
  regions, different kinds of work: lean "different" unless the name is distinctive.
- A very common name (`ABC CONSULTING`) needs more evidence than a distinctive one.
- Do not look up the model's score. The sheet hides it on purpose, so the labels
  stay independent of the model.
- Aim for about 10 seconds per pair. When in doubt, answer `?`.

`label_key.csv` maps each `pair_id` to its stratum and weight. Do not edit it.
The labels are part of the evidence for this project, so commit them.
