# Assembling the human-facing explanation report

The diagnostic (`report.json` + images) is a fixed **self-check for you, the agent**. It is not the
report the human asked for. Once a run PASSes (or you've decided to ship a WARN with caveats),
synthesize a human-facing report from the run's artifacts, tailored to what the user actually
requested. There is no fixed template — the shape depends on the prompt — but the following are
the reusable ingredients and good defaults.

## Where the material comes from
Everything is in the archive folder
`graph_attention_student/experiments/results/train_model__megan__<dataset>/debug/`:
- **`report_test.json`** — the **held-out test** metrics + verdicts. **Quote these as the headline
  numbers** (predictive performance + explanation quality). This is the honest, unbiased estimate.
- `report.json` — the **validation** metrics used for tuning. If you swept knobs, these are
  *selection* estimates (optimistically biased) — do not present them as the model's true
  performance; use `report_test.json` for any number you report to the user.
- `scorecard.png` — the at-a-glance quality panel.
- `examples.png` — per-channel explanation overlays on example molecules.
- `example_explanations.pdf` — the fuller set of example explanations (more molecules).
- `fidelity.png`, `leave_one_out.pdf` — fidelity evidence.
- `regression.png` / confusion matrix — predictive fit.
- If clustering was enabled: `cluster__ch*.pdf` — recurring explanation motifs (concepts).

## What to include (defaults)
1. **Headline result** — what was predicted, on how many molecules, and the predictive quality in
   plain terms (e.g. "predicts logS with R² 0.87 / MAE 0.54 on a held-out test set of N molecules").
2. **What the explanations mean** — state the channel semantics for THIS task (from
   `CHANNEL_INFOS`), e.g. "the model attributes each prediction to two channels: substructures that
   *decrease* solubility and substructures that *increase* it."
3. **Are the explanations trustworthy** — summarize the self-check honestly: which axes passed,
   the explanation-accuracy AUC, and the fidelity sign result. If anything WARNed/FAILed, say so
   and what it means, don't bury it.
4. **Illustrative examples** — embed a few from `examples.png` / `example_explanations.pdf` and
   describe what the highlighted substructures are, connecting them to domain intuition where you
   can. This is usually the part the user cares about most.
5. **Caveats & how to use** — dataset size/coverage, domains of applicability, and that
   explanations are model attributions (correlational), not ground-truth causal mechanisms.

## Presentation
- Match the requested format. If none specified, a concise markdown summary with a couple of
  embedded images is a good default; a shareable HTML/PDF artifact is appropriate when the user
  wants something to circulate.
- Lead with the conclusion (verdict + headline metric), then the explanation examples, then the
  supporting detail. Keep raw metric dumps in an appendix, not the lede.
- **Never overstate.** If the diagnostic didn't PASS cleanly, the human report must reflect that.
