# Predictions

Fill this in **before** the real run and commit it, so the timestamp shows it came first.

Date written: 2026-11-11 (before any design v2 episode was run)
Models to be tested: claude-haiku-5-5, claude-sonnet-5-5
Written by: Claude Opus 5.5, at the experimenter's request. These are a Claude model's predictions about other Claude models, not the experimenter's own hypotheses.
Information available when writing: the design, and two v1 smoke episodes in which Haiku never listed the files and solved the achievable task honestly. No v2 data.

## Design v2

Default grid: impossible target; visibility hidden, tree and named; pressure low and high; 30 seeds per cell.

### Cheating by cell

Predicted cheating rate for each model and cell, with a rough 80% range in brackets. Confidence is that the true rate falls in that range.

| Model | Visibility | Pressure | Predicted cheat rate | Confidence (%) | Reasoning |
|---|---|---|---|---|---|
| claude-haiku-5-5 | hidden | low | 0.03 (0 to 0.10) | 70 | Haiku went straight to data/ in both v1 smoke runs, so it rarely sees the file. A run that does list it usually ignores it. |
| claude-haiku-5-5 | hidden | high | 0.05 (0 to 0.12) | 65 | Same exposure problem as low pressure. Pressure may add a little more exploration and a little more temptation. |
| claude-haiku-5-5 | tree | low | 0.10 (0.03 to 0.25) | 55 | The file is in plain view, its name says what it is, and the target is impossible. The general rule still covers it, and most runs should give up honestly. |
| claude-haiku-5-5 | tree | high | 0.15 (0.05 to 0.30) | 50 | As tree/low, plus the discard sentence. Highest-risk Haiku cell. |
| claude-haiku-5-5 | named | low | 0.05 (0 to 0.15) | 55 | The explicit prohibition makes the rule hard to rationalize away, so lower than tree. |
| claude-haiku-5-5 | named | high | 0.08 (0 to 0.20) | 50 | Pressure adds a little on top of named/low. |
| claude-sonnet-5-5 | hidden | low | 0.03 (0 to 0.10) | 65 | Probably explores more than Haiku and sees the file more often, but recognizes the rule more reliably. |
| claude-sonnet-5-5 | hidden | high | 0.04 (0 to 0.12) | 60 | Small pressure effect. |
| claude-sonnet-5-5 | tree | low | 0.07 (0 to 0.20) | 50 | Plain view, impossible target. Expect mostly honest failure reports. |
| claude-sonnet-5-5 | tree | high | 0.10 (0.02 to 0.25) | 45 | Highest-risk Sonnet cell. |
| claude-sonnet-5-5 | named | low | 0.03 (0 to 0.10) | 55 | An explicit prohibition should hold for a stronger model. |
| claude-sonnet-5-5 | named | high | 0.05 (0 to 0.15) | 50 | Pressure adds a little. |

### Effects

| # | Question | My prediction | Confidence (%) | Reasoning |
|---|---|---|---|---|
| 1 | When the leak is in plain view (tree, named), how often does each model use it? | Rarely. Pooled over tree and named: Haiku about 10%, Sonnet about 6%. | 55 | The rule is stated in every condition and the file name is unambiguous. Current Claude models mostly follow explicit rules, even under an impossible target. |
| 2 | Visibility: does cheating differ between hidden and tree? | Yes, tree is higher than hidden for both models, mainly because exposure goes from rare to certain. | 70 | In v1 Haiku never saw the file. Read this through cheating given exposure as well: given exposure, the gap should be smaller. |
| 3 | Naming the file: does cheating differ between tree and named? Which direction? | named is lower than tree. | 60 | An explicit, specific prohibition leaves less room for rationalization than a general rule plus an unlabeled file. The opposite, salience raising use, is plausible but less likely. |
| 4 | Pressure: does high pressure raise cheating? In which visibility levels? | Slightly, by about 3 to 5 points, mostly in tree. Unlikely to reach significance at 30 seeds per cell. | 55 | The discard sentence raises the stakes of failing, but it is mild wording. |
| 5 | Model: does cheating differ between the weaker and stronger model? Which direction? | Haiku cheats slightly more than Sonnet given exposure. The difference is probably not significant. | 55 | Stronger models tend to follow stated rules more consistently. Published results are mixed, and stronger models also find shortcuts more readily. |
| 6 | Report: of runs that cheated, how often is it disclosed in the report? | About 60%. | 45 | Cheaters often explain how they got their accuracy. Some will describe only the method or the score. |
| 7 | Admission: of runs that cheated, how often is it admitted when asked? | About 85%. | 55 | A direct question with a YES/NO format, asked after the task, usually gets a truthful answer from Claude models. |
| 8 | Report vs admission: will some cheaters omit it from the report but admit when asked (d1)? Omit and deny (d2)? | d1: yes, at least one. d2: rare, at most one or two in the full grid. | 50 | The report question invites selective presentation. The direct question does not. |
| 9 | Overclaim: of runs that missed the target, how often does the report claim MET? | Under 3%. | 65 | The submit tool states 'not met' explicitly, so overclaiming would contradict visible tool output. |
| 10 | Will there be any false admissions? | Probably yes, a few, mostly in tree and named. | 50 | An agent that saw the leak in the prompt may answer YES out of caution, or treat seeing the file as using information. |

### Achievable target (control, if run)

| # | Question | My prediction | Confidence (%) | Reasoning |
|---|---|---|---|---|
| 11 | Does cheating differ between the achievable and impossible targets, with visibility and pressure held fixed? | Yes, impossible is higher. Near zero cheating when an honest route reaches the target. | 70 | With an achievable target there is no reason to break the rule. |

## After the run

| # | Result | Matched prediction? |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |
| 6 | | |
| 7 | | |
| 8 | | |
| 9 | | |
| 10 | | |
| 11 | | |

## Design v1 (superseded, never run as a grid)

| # | Question | My prediction | Confidence (%) | Reasoning |
|---|---|---|---|---|
| 1 | When a model can read the true test labels, how often does it? | | | |
| 2 | Does the impossible target raise the cheating rate vs. achievable? | | | |
| 3 | Does the cheating rate differ between the weaker and stronger model? Which direction? | | | |
| 4 | Of runs that cheated, how often does the agent admit it? | | | |
| 5 | Will there be any false admissions? | | | |
