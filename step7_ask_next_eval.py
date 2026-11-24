"""Out-of-time test of the "ask next" engine.

Question: if a call-taker can ask only one (or two) of the four optional questions, does asking the one the engine
ranks first improve the score more than asking a random one?

Protocol: train on 2021-2024, test once on 2025. Calls start with only what is known on arrival (the five core fields,
the time, and the police force from routing). The four optional context fields start hidden. Each strategy reveals
the true value of one question, then a second. Calls where any of the four true values is itself recorded as unknown
are left out, because the answer could not be obtained in practice.

Outputs: models/ask_priors.json (how common each answer is, used to weight the ranking), reports/ask_next_eval.csv,
reports/ask_next_eval.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from common import MULTI_CLEAN, OPTIONAL_LABELS, ROOT, SEED, TIERS
from insights import ASKABLE, PRIORS_PATH
from step4_train_multiyear import fit_lgbm, load_clean, p_at_rate, score_lgbm

TIER = "T4_plus_force"
SAMPLE = 8000
RANDOM_REPEATS = 5
BOOTSTRAPS = 300


def metrics(y, p):
    return dict(roc_auc=float(roc_auc_score(y, p)), log_loss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
                p_at_2pct=p_at_rate(y, p, 0.02))


def main():
    df, categories = load_clean()
    features = TIERS[TIER]
    train, test = df[df.year <= 2024], df[df.year == 2025]
    model = fit_lgbm(train, features, categories)

    priors = {}
    for field in ASKABLE:
        counts = train[field].astype(str).value_counts()
        known = {code: int(counts.get(code, 0)) for code in OPTIONAL_LABELS[field]}
        total = sum(known.values())
        priors[field] = {code: n / total for code, n in known.items()}
    PRIORS_PATH.write_text(json.dumps(priors, indent=2))

    answered = np.logical_and.reduce([test[f].astype(str).isin(OPTIONAL_LABELS[f]) for f in ASKABLE])
    sample = test[answered].sample(min(SAMPLE, int(answered.sum())), random_state=SEED).reset_index(drop=True)
    y = sample.target.to_numpy()
    print(f"2025 test calls with all four answers known: {int(answered.sum()):,}; sample used: {len(sample):,}")

    def score(frame):
        return score_lgbm(model, frame, features, categories)

    def blank_state():
        state = sample.copy()
        for f in ASKABLE:
            state[f] = -1
        return state

    def value_of_asking(state, hidden):
        """Expected absolute change in the score from asking each field, weighted by how common each answer is."""
        base = score(state)
        voi = np.full((len(state), len(ASKABLE)), -np.inf)
        for j, f in enumerate(ASKABLE):
            gain = np.zeros(len(state))
            for code, weight in priors[f].items():
                trial = state.copy()
                trial[f] = int(code)
                gain += weight * np.abs(score(trial) - base)
            voi[:, j] = np.where(hidden[:, j], gain, -np.inf)
        return voi

    def run(strategy, steps, rng=None, fixed_order=None):
        state, hidden = blank_state(), np.ones((len(sample), len(ASKABLE)), bool)
        rows = np.arange(len(sample))
        history = [score(state)]
        for step in range(steps):
            if strategy == "ask_next":
                pick = value_of_asking(state, hidden).argmax(axis=1)
            elif strategy == "random":
                weights = hidden * rng.random(hidden.shape)
                pick = weights.argmax(axis=1)
            else:
                pick = np.full(len(sample), fixed_order[step])
            for j, f in enumerate(ASKABLE):
                chosen = pick == j
                state.loc[chosen, f] = sample.loc[chosen, f].to_numpy()
            hidden[rows, pick] = False
            history.append(score(state))
        return history

    results, curves = [], {}
    nothing = run("ask_next", 0)[0]
    results.append(dict(strategy="no optional answers", questions=0, **metrics(y, nothing)))
    ask = run("ask_next", 2)
    curves["ask_next"] = ask
    random_runs = [run("random", 2, rng=np.random.default_rng(SEED + r)) for r in range(RANDOM_REPEATS)]
    curves["random"] = random_runs[0]
    for q in (1, 2):
        results.append(dict(strategy="ask next (engine order)", questions=q, **metrics(y, ask[q])))
        per_seed = [metrics(y, run_[q]) for run_ in random_runs]
        results.append(dict(strategy="random order (mean of 5)", questions=q,
                            **{k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}))
    for j, f in enumerate(ASKABLE):
        order = [j] + [k for k in range(len(ASKABLE)) if k != j]
        fixed = run("fixed", 1, fixed_order=order)
        results.append(dict(strategy=f"always ask {f} first", questions=1, **metrics(y, fixed[1])))
    state = sample.copy()
    results.append(dict(strategy="all four answers (upper bound)", questions=4, **metrics(y, score(state))))

    rng = np.random.default_rng(SEED)
    diffs = {"roc_auc": [], "log_loss": []}
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(y), len(y))
        for name, fn in [("roc_auc", roc_auc_score), ("log_loss", lambda a, b: log_loss(a, np.clip(b, 1e-6, 1 - 1e-6)))]:
            diffs[name].append(fn(y[idx], ask[1][idx]) - fn(y[idx], curves["random"][1][idx]))
    interval = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in diffs.items()}

    table = pd.DataFrame(results)
    table.to_csv(ROOT / "reports/ask_next_eval.csv", index=False)
    summary = dict(test_year=2025, calls_in_sample=len(sample), base_rate=float(y.mean()), train_years="2021-2024",
                   ask_next_minus_random_one_question=dict(
                       roc_auc=float(table[(table.strategy == "ask next (engine order)") & (table.questions == 1)].roc_auc.iloc[0]
                                     - table[(table.strategy == "random order (mean of 5)") & (table.questions == 1)].roc_auc.iloc[0]),
                       ci95=interval),
                   note="Calls where any optional answer is recorded as unknown are excluded. CI is a paired bootstrap against one random run.")
    (ROOT / "reports/ask_next_eval.json").write_text(json.dumps(summary, indent=2))
    print(table.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
