# Milestone Reports

Milestone reports are durable thesis-memory artifacts. They are meant to preserve:

- what experiments were run
- what the raw results were
- what the results mean scientifically
- what caveats and methodological warnings apply
- how the result should later be framed in the thesis

This is intentionally different from raw experiment logging. A milestone report keeps raw results and interpretation separate so that future writing does not rely on memory alone.

## Outputs

Running the milestone reporter writes three files next to each other:

- a human-readable Markdown report
- a machine-readable JSON summary
- a JSON schema generated from the structured `MilestoneReport` model

The default durable output location is:

- `reports/milestones/`

## Config Format

The reporter is driven by a small YAML config, for example:

- `reports/milestone_configs/20260313_leakage_free_classification.yaml`

Relative paths in the config are interpreted relative to the repository root.
Artifact paths should usually point to run directories, but summary files are also acceptable when the location is unambiguous.

The config has two kinds of fields:

- artifact pointers, which should come from saved experiment outputs
- manual interpretation, which should capture scientific meaning, caveats, and thesis framing without fabricating metrics

You should provide manually:

- the plain-language takeaway
- the scientific meaning
- caveats and methodological warnings
- the thesis framing recommendation
- recommended next steps
- any reusable thesis sentences you want to preserve

## Command

```powershell
.venv\Scripts\python.exe -m src.reporting.generate_milestone_report --config reports\milestone_configs\20260313_leakage_free_classification.yaml
```

Or, after installing the editable project scripts:

```powershell
qfc-generate-milestone-report --config reports\milestone_configs\20260313_leakage_free_classification.yaml
```

## What The Report Preserves

The generated report is designed to keep these distinctions explicit:

- raw validation winner
- raw held-out test winner
- thesis headline recommendation
- negative or non-improving results
- caveats that should stop later overclaiming

That makes it safer to reuse later during thesis drafting, paper writing, or presentation prep.
