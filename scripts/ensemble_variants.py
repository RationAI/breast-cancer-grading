"""Squeeze more out of finished runs: ensemble + uniform-prior decision rule.

Both tricks need only the per-slide prediction dumps that training already writes
(``<ckpt-dir>/predictions/{valid,test}_epoch<N>.csv``), so they cost CPU minutes and
no GPU. Neither is a new model; both target the metric we judge on.

* **Ensemble** — average the softmax probabilities of several runs. Seed noise here is
  a 6-point range on identical configs, so averaging members is the cheapest gain.
* **Uniform-prior decision rule** — ``balanced accuracy`` scores every class equally,
  but the training prior is 593/68/254/293, so an argmax trained on it is biased
  toward class 0 (recall 0.98-0.99 in every run) at the cost of class 1 (0.29-0.79).
  Re-deciding with ``argmax_k p_k / prior_k ** lambda`` corrects exactly that;
  ``lambda`` is one scalar fitted on the *validation* dumps, and ``lambda = 1`` is the
  fully corrected uniform prior. The row-wise renormalisation cancels in the argmax, so
  the correction moves the decision boundary and nothing else - AUROC, which ranks
  rather than decides, is reported from the uncorrected probabilities.

Usage::

    python scripts/ensemble_variants.py --variants full_ord_s456,full_ord_s789
    python scripts/ensemble_variants.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from report_variants import NUM_CLASSES, metrics_from_logits
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


# Training-split slide counts (bc20-24 ``labels.csv``), the prior the argmax learned.
TRAIN_COUNTS = np.array([593.0, 68.0, 254.0, 293.0])
LAMBDA_GRID = np.linspace(0.0, 1.0, 11)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def prior_corrected(probs: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
    """Rescale each class's probability by ``1 / prior ** lam`` and renormalise.

    ``lam = 0`` is the identity; ``lam = 1`` decides as if the classes were equiprobable.
    The renormalisation is per slide, so it cancels in ``argmax``: this is a decision
    rule, not a new score (AUROC does move, because it ranks across slides).
    """
    scaled = probs / prior**lam
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_lambda(probs: np.ndarray, labels: np.ndarray, prior: np.ndarray) -> float:
    """Best ``lambda`` on *validation* balanced accuracy; ties keep the smallest."""
    grid = [
        balanced_accuracy_score(labels, prior_corrected(probs, prior, lam).argmax(1))
        for lam in LAMBDA_GRID
    ]
    return float(LAMBDA_GRID[int(np.argmax(grid))])


def dump_dir(checkpoint_root: str, variant: str) -> str:
    """MLflow stores ``full_ord_s456``; the checkpoint directory is ``ml_attention-full-ord-s456``."""
    return os.path.join(
        checkpoint_root, f"ml_attention-{variant.replace('_', '-')}", "predictions"
    )


def tested_epoch(predictions_dir: str) -> int | None:
    """Epoch of the checkpoint the test dump came from, read out of ``best.ckpt``.

    ``filename: best`` in ``configs/trainer/default.yaml`` means the filename carries no epoch,
    and the test dump's own number is one past the last validation dump because Lightning
    increments ``current_epoch`` before testing - so the epoch has to come from the checkpoint.
    The head is a few thousand parameters, so loading it is a millisecond-scale read.
    """
    try:
        import torch

        checkpoint = torch.load(
            os.path.join(os.path.dirname(predictions_dir), "best.ckpt"),
            map_location="cpu",
            weights_only=False,
        )
        return int(checkpoint["epoch"])
    except Exception:
        return None


def load_variant(
    checkpoint_root: str, variant: str
) -> tuple[str, pd.DataFrame, pd.DataFrame] | None:
    """Return (epoch, valid frame, test frame) for the dumped predictions, or None.

    The valid and test dumps are **not** numbered the same: Lightning bumps
    ``current_epoch`` before the final test, so a 19-epoch run leaves ``test_epoch18.csv``
    next to ``valid_epoch0..17.csv``, and pairing by equal number never matches (the first
    ensemble job printed "no dumps" for variants the reporter had just listed). The test frame
    is the single ``test_epoch<N>.csv``; the valid frame the decision rule is fitted on is the
    epoch the tested checkpoint was saved at, falling back to the last dumped epoch when the
    checkpoint cannot be read.
    """
    directory = dump_dir(checkpoint_root, variant)
    tests = [
        int(p.rsplit("epoch", 1)[1].split(".")[0])
        for p in glob.glob(os.path.join(directory, "test_epoch*.csv"))
    ]
    valids = [
        int(p.rsplit("epoch", 1)[1].split(".")[0])
        for p in glob.glob(os.path.join(directory, "valid_epoch*.csv"))
    ]
    if not tests or not valids:
        return None
    # Prefer the epoch the tested checkpoint was saved at: fitting the decision rule on a
    # different epoch's model than the one that produced the test frame would mispair them.
    epoch = tested_epoch(directory)
    if epoch not in valids:
        epoch = max(valids)
    frame = pd.read_csv(os.path.join(directory, f"valid_epoch{epoch}.csv"))
    test = pd.read_csv(os.path.join(directory, f"test_epoch{max(tests)}.csv"))
    return (f"test{max(tests)}/valid{epoch}", frame, test)


def align(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Keep only slides present in every frame, in one shared order."""
    common = set(frames[0]["slide_id"])
    for frame in frames[1:]:
        common &= set(frame["slide_id"])
    order = sorted(common)
    return [f.set_index("slide_id").loc[order].reset_index() for f in frames]


def probs_of(frame: pd.DataFrame) -> np.ndarray:
    return softmax(frame[[f"logit_{c}" for c in range(NUM_CLASSES)]].to_numpy())


def blend(frames: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(frames), axis=0)


def best_lambda(
    probs: np.ndarray, labels: np.ndarray, prior: np.ndarray
) -> tuple[float, float]:
    """``(lambda, balanced accuracy)`` of the best point on ``LAMBDA_GRID``."""
    scored = [
        (
            balanced_accuracy_score(
                labels, prior_corrected(probs, prior, lam).argmax(1)
            ),
            float(lam),
        )
        for lam in LAMBDA_GRID
    ]
    score, lam = max(scored)
    return lam, score


def greedy_select(
    val_probs: list[np.ndarray],
    labels: np.ndarray,
    prior: np.ndarray,
    rounds: int = 12,
    min_gain: float = 1e-3,
) -> tuple[list[int], list[float]]:
    """Caruana ensemble selection with replacement, scored **on validation only**.

    Replaces hand-picking member sets, which is what made the 2026-11-11 numbers partly
    selection luck: each round adds the member whose inclusion raises validation balanced
    accuracy most, at that blend's own best lambda, and stops when the gain falls under
    ``min_gain``. Selecting on 261 slides (16 of them class 1) overfits a little by
    construction - hence the printed validation curve and the single test evaluation.
    """
    chosen: list[int] = []
    curve: list[float] = []
    best_score = -1.0
    for _ in range(rounds):
        best: tuple[float, int] | None = None
        for i in range(len(val_probs)):
            _, score = best_lambda(
                blend([val_probs[j] for j in [*chosen, i]]), labels, prior
            )
            if best is None or score > best[0]:
                best = (score, i)
        if best is None or best[0] <= best_score + min_gain:
            break
        best_score, index = best
        chosen.append(index)
        curve.append(best_score)
    return chosen, curve


def rescored(probs: np.ndarray, temperature: float, bias: np.ndarray) -> np.ndarray:
    """Temperature-scaled log-probabilities plus a per-class offset (argmax scores)."""
    return np.log(np.clip(probs, 1e-12, None)) * temperature + bias


def fit_rescoring(
    probs: np.ndarray,
    labels: np.ndarray,
    sweeps: int = 3,
    reg: float = 0.02,
    objective: str = "balanced",
) -> tuple[float, np.ndarray, float]:
    """Fit one temperature and a four-value class offset on ``labels``.

    Balanced accuracy is maximised by per-class thresholds, not by the single scalar the prior
    rule applies, and that rule is the special case ``bias = -lambda * log(prior)``. Five free
    parameters on 261 slides is a real overfitting risk, so the objective carries a ridge
    penalty on the offset and the caller prints the val -> test drop next to the number.

    ``objective="accuracy"`` fits the same five numbers to the metric the confusion matrix is
    counted in: a boundary can only gain accuracy if the model is mis-scaled per class, and this
    is the version that would find it.
    """
    temperature, bias = 1.0, np.zeros(NUM_CLASSES)
    classes = np.arange(NUM_CLASSES)
    metric = (
        balanced_accuracy_score
        if objective == "balanced"
        else lambda y, p: accuracy_score(y, p)
    )

    def score(temperature: float, bias: np.ndarray) -> float:
        predicted = metric(labels, rescored(probs, temperature, bias).argmax(1))
        return predicted - reg * float(np.dot(bias, bias))

    for _ in range(sweeps):
        for k in range(NUM_CLASSES):
            candidates = [
                np.where(classes == k, bias[k] + delta, bias)
                for delta in np.linspace(-0.6, 0.6, 13)
            ]
            bias = max(candidates, key=lambda candidate: score(temperature, candidate))
        temperatures = [float(t) for t in np.linspace(0.5, 2.5, 11)]
        temperature = max(temperatures, key=lambda t: score(t, bias))
    fitted = metric(labels, rescored(probs, temperature, bias).argmax(1))
    return temperature, bias, float(fitted)


def show(
    name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    auroc_from: np.ndarray | None = None,
) -> dict[str, float]:
    m = metrics_from_logits(
        labels, probs
    )  # argmax only: scores or probabilities both fine
    ranking = probs if auroc_from is None else auroc_from
    m["auroc"] = float(
        roc_auc_score(labels, ranking, average="macro", multi_class="ovr")
    )
    print(
        f"{name:34s} n={len(labels):3d} bal={m['balanced_accuracy']:.4f} "
        f"auroc={m['auroc']:.4f} f1={m['macro_f1']:.4f} kappa={m['quadratic_kappa']:.4f} "
        f"recals {m['recall_c0']:.2f}/{m['recall_c1']:.2f}/{m['recall_c2']:.2f}/{m['recall_c3']:.2f}"
    )
    return m


def confusion(name: str, pred: np.ndarray, labels: np.ndarray) -> None:
    """Where the mistakes sit, as counts: rows are the pathologist, columns are the model."""
    print(f"  confusion {name} (rows = label, cols = predicted)")
    print("        " + " ".join(f"p{c:<4d}" for c in range(NUM_CLASSES)))
    for truth in range(NUM_CLASSES):
        row = [int(((pred == c) & (labels == truth)).sum()) for c in range(NUM_CLASSES)]
        print(f"    t{truth}  " + " ".join(f"{v:5d}" for v in row))


def direction(pred: np.ndarray, labels: np.ndarray) -> str:
    """Which way the mistakes point, because balanced accuracy cannot tell.

    ``pred < label`` means the model read a slide as *less* aggressive than the pathologist did -
    an undercall, the error a clinician asks about first, and the one a screening score must not
    trade away for a prettier average. ``pred > label`` is the harmless opposite.
    """
    d = np.asarray(pred) - np.asarray(labels)
    steps = [
        ("exact", d == 0),
        ("under1", d == -1),
        ("under2+", d <= -2),
        ("over1", d == 1),
        ("over2+", d >= 2),
    ]
    parts = [f"{name} {int(m.sum())} ({m.mean():.1%})" for name, m in steps]
    per_class = " ".join(
        f"c{c}:{int(((d < 0) & (labels == c)).sum())}/{int((labels == c).sum())}"
        for c in range(NUM_CLASSES)
    )
    return (
        ", ".join(parts) + f" of {len(labels)}; undercalled per class {per_class}"
    )


def lambda_ladder(
    scores: np.ndarray,
    labels: np.ndarray,
    prior: np.ndarray,
    lambdas: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
) -> list[dict]:
    """Show what each rule strength buys and costs, instead of picking one.

    Printed on out-of-fold rows only, and the shipped lambda stays **fitted** by ``best_lambda``.
    Choosing the best row by eye on the slides it is scored on is the mistake behind the
    retracted 0.8002. What is genuinely missing is the cost of an undercall versus an overcall,
    and that is the pathologist's to state - so this prints counts, not a verdict.
    """
    print("  lambda ladder (shown, not chosen; a cost matrix decides the row):")
    rows = []
    for lam in lambdas:
        corrected = prior_corrected(scores, prior, lam)
        pred = corrected.argmax(1)
        m = metrics_from_logits(labels, corrected)
        row = {
            "lambda": float(lam),
            "balanced_accuracy": m["balanced_accuracy"],
            "quadratic_kappa": m["quadratic_kappa"],
            "undercalled": int((pred < labels).sum()),
            "overcalled": int((pred > labels).sum()),
        }
        print(
            f"    lambda {lam:>3.1f} bal={row['balanced_accuracy']:.4f} "
            f"kappa={row['quadratic_kappa']:.4f} recals "
            f"{m['recall_c0']:.2f}/{m['recall_c1']:.2f}/{m['recall_c2']:.2f}/{m['recall_c3']:.2f} "
            f"under={row['undercalled']} over={row['overcalled']}"
        )
        rows.append(row)
    return rows


def rule_pairing(name: str, raw: np.ndarray, corrected: np.ndarray, labels: np.ndarray) -> None:
    """Pair the two operating points slide by slide: right one way, wrong the other.

    Two balanced accuracies 2 points apart are inside the ruler, so the number that decides is the
    discordant pair count on the same slides. Counted alongside is *which way a flip goes*, because
    McNemar on accuracy cannot see a slide moving from "correct" to "read one grade too low", which
    is the whole content of the lambda decision.
    """
    first, second = raw.argmax(1), corrected.argmax(1)
    right_first, right_second = first == labels, second == labels
    only_first = int((right_first & ~right_second).sum())
    only_second = int((right_second & ~right_first).sum())
    split = only_first + only_second
    p = binomtest(max(only_first, only_second), split, 0.5).pvalue if split else 1.0
    into_under = int(((first >= labels) & (second < labels)).sum())
    out_of_under = int(((first < labels) & (second >= labels)).sum())
    print(
        f"  {name}: accuracy {right_first.mean():.4f} vs {right_second.mean():.4f} on "
        f"{len(labels)} slides, discordant {only_first}/{only_second}, McNemar p={p:.4f}; "
        f"flips into undercall {into_under}, out of undercall {out_of_under} "
        f"(under calls {int((first < labels).sum())} vs {int((second < labels).sum())})"
    )


def escalate(scores: np.ndarray, beta: float) -> np.ndarray:
    """Bias the argmax *upward*: ``log p_c + beta * c``, so ties break to the higher grade.

    This is the knob for "never report a slide lower than it is". The prior correction is not that
    knob - it shifts mass toward whichever class the training split was short of, and class 1 sits
    *below* class 2, so a positive lambda pushes grade-4 slides down into grade 3.
    """
    ramp = np.arange(NUM_CLASSES, dtype=float)
    return np.log(np.clip(scores, 1e-12, None)) + beta * ramp


def sensitivity(pred: np.ndarray, labels: np.ndarray) -> tuple[float, int, int]:
    """Detection of anything above the bottom class, plus the false alarms it costs.

    ``label >= 1`` is "the pathologist saw something", so this is the miss rate a screening use
    cares about; the third number is the price - bottom-class slides escalated.
    """
    tumour, called = labels >= 1, pred >= 1
    caught = int((tumour & called).sum())
    return (
        caught / int(tumour.sum()) if tumour.any() else float("nan"),
        int((tumour & ~called).sum()),
        int((~tumour & called).sum()),
    )


def escalation_ladder(
    scores: np.ndarray,
    labels: np.ndarray,
    betas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    target: float = 0.99,
) -> float:
    """Price list for escalating: what each bias strength buys in misses and costs in false alarms.

    Chosen on these rows by the protocol below - the smallest beta that reaches ``target`` detection
    of the non-bottom classes - so the pick does not depend on reading the test numbers. Returns the
    chosen beta.
    """
    print(f"  escalation ladder: smallest beta reaching {target:.0%} detection of grades above c0")
    chosen, rows = None, []
    for beta in betas:
        pred = escalate(scores, beta).argmax(1)
        sens, missed, false = sensitivity(pred, labels)
        m = metrics_from_logits(labels, np.eye(NUM_CLASSES)[pred])
        rows.append((beta, sens, missed, false, m))
        print(
            f"    beta {beta:>4.2f} detect={sens:.4f} missed={missed:3d} false-alarms={false:3d} "
            f"under={int((pred < labels).sum()):3d} over={int((pred > labels).sum()):3d} "
            f"bal={m['balanced_accuracy']:.4f} kappa={m['quadratic_kappa']:.4f} recals "
            f"{m['recall_c0']:.2f}/{m['recall_c1']:.2f}/{m['recall_c2']:.2f}/{m['recall_c3']:.2f}"
        )
        if chosen is None and sens >= target:
            chosen = beta
    chosen = betas[-1] if chosen is None else chosen
    if rows[-1][1] < target:
        print(
            f"    target NOT reachable with this bias: best detection {rows[-1][1]:.4f} at beta "
            f"{betas[-1]} (the grid maximum), so the beta below is the most aggressive tested, "
            "not the cheapest that hits the target"
        )
    print(f"    chosen beta = {chosen}")
    return float(chosen)


def _fold_recipe(variant: str) -> tuple[str, str]:
    """``kf3-ctl`` is fold ``kf3`` of recipe ``ctl``.

    A name with no recipe suffix (``kf3``) is a fold of one shared recipe: several bare fold names
    in one call are seeds or repeats of the same model, and pooling them is the point.
    """
    fold, sep, recipe = variant.rpartition("-")
    return (variant, "all") if not sep else (fold, recipe)


def pooled_oof(
    loaded: list[tuple[str, str, pd.DataFrame, pd.DataFrame]], prior: np.ndarray
) -> None:
    """Score every fold's held-out fifth together: ~1469 slides instead of 261.

    Each fold's validation split is the fifth its model never trained on, so concatenating the
    five validation frames is out-of-fold for the whole train+val pool. The decision rule is
    fitted the honest way - leave one fold out, fit on the rest, predict that fold - so the
    number includes fitting the rule and still has a ~1-point ruler instead of \u00b12.6.

    Several recipes may be pooled in one call (``kf0-ctl,kf0-cand,...``). Each is scored on its
    own pooled folds, then the recipes are compared slide by slide: only a per-slide difference
    cancels the ~4-point fold-to-fold spread that swamps a 1-point recipe effect.
    """
    logits = [f"logit_{c}" for c in range(NUM_CLASSES)]
    columns = [f"prob_{c}" for c in range(NUM_CLASSES)]
    rows = pd.concat(
        [
            valid[["slide_id", "label"]]
            .assign(fold=_fold_recipe(variant)[0], recipe=_fold_recipe(variant)[1])
            .join(pd.DataFrame(softmax(valid[logits].to_numpy()), columns=columns))
            for variant, _, valid, _ in loaded
        ],
        ignore_index=True,
    )
    # Several models of one fold average into one prediction per slide; a slide must not appear
    # in two folds, which the fold generator guarantees and this would silently corrupt.
    if rows.groupby("slide_id")["fold"].nunique().gt(1).any():
        raise ValueError(
            "a slide was held out by two folds - the label CSVs are not the same split"
        )
    # `recipe` is a group key so two recipes never average into each other; seeds of one recipe
    # and one fold do, into a single prediction per slide.
    pooled = rows.groupby(["slide_id", "label", "fold", "recipe"], as_index=False)[
        columns
    ].mean()
    recipes = sorted(pooled["recipe"].unique())
    print(
        f"pooled out-of-fold over {pooled['fold'].nunique()} folds"
        + (f", recipes {', '.join(recipes)}" if len(recipes) > 1 else "")
    )
    right: dict[str, pd.Series] = {}
    for recipe in recipes:
        part = pooled[pooled["recipe"] == recipe]
        # With one recipe the label would be noise, so it only tags lines that compare recipes.
        tag = f"[{recipe}] " if len(recipes) > 1 else ""
        scores = part[columns].to_numpy()
        labels = part["label"].to_numpy()
        folds = part["fold"].to_numpy()
        show(f"{tag}pooled out-of-fold, no rule", scores, labels)

        corrected = np.zeros_like(scores)
        for fold in np.unique(folds):
            trained = folds != fold
            lam, _ = best_lambda(scores[trained], labels[trained], prior)
            corrected[~trained] = prior_corrected(scores[~trained], prior, lam)
        show(
            f"{tag}pooled out-of-fold + rule (fitted leave-one-fold-out)",
            corrected,
            labels,
        )
        right[recipe] = pd.Series(
            corrected.argmax(1) == labels, index=part["slide_id"].to_numpy()
        )

        # Balanced accuracy hides *where* the errors are and *which way* they point, and both are
        # the questions a pathologist asks before believing the number.
        confusion("no rule", scores.argmax(1), labels)
        confusion("+ rule", corrected.argmax(1), labels)
        print(f"{tag}error direction out-of-fold, no rule: {direction(scores.argmax(1), labels)}")
        print(f"{tag}error direction out-of-fold, + rule:   {direction(corrected.argmax(1), labels)}")
        lambda_ladder(scores, labels, prior)
        beta = escalation_ladder(scores, labels)
        rule_pairing(
            f"{tag}rule pairing out-of-fold, lambda 0 vs fitted leave-one-fold-out",
            scores,
            corrected,
            labels,
        )

        per_fold = []
        per_fold_acc = []
        acc_boundary = np.zeros_like(scores)
        for fold in np.unique(folds):
            trained, held = folds != fold, folds == fold
            temperature, bias, _ = fit_rescoring(scores[trained], labels[trained])
            per_fold.append(
                balanced_accuracy_score(
                    labels[held], rescored(scores[held], temperature, bias).argmax(1)
                )
            )
            # The boundary fitted to *accuracy*, which is the metric a pathologist's confusion
            # matrix is counted in. If the model is mis-scaled per class this finds it; if the
            # ranking is all there is, it buys nothing and the paired test below says so.
            t_acc, b_acc, _ = fit_rescoring(
                scores[trained], labels[trained], objective="accuracy"
            )
            acc_boundary[held] = rescored(scores[held], t_acc, b_acc)
            per_fold_acc.append(
                float(accuracy_score(labels[held], acc_boundary[held].argmax(1)))
            )
        print(
            f"{tag}pooled out-of-fold + bias/temp fitted leave-one-fold-out: "
            f"{' '.join(f'{s:.3f}' for s in per_fold)} "
            f"(spread {max(per_fold) - min(per_fold):.3f})"
        )
        print(
            f"{tag}boundary fitted to accuracy, leave-one-fold-out: "
            f"{' '.join(f'{s:.3f}' for s in per_fold_acc)} "
            f"(mean {sum(per_fold_acc) / len(per_fold_acc):.4f} vs {accuracy_score(labels, scores.argmax(1)):.4f} unfitted)"
        )
        rule_pairing(
            f"{tag}accuracy-fitted boundary pairing vs no boundary, leave-one-fold-out",
            scores,
            acc_boundary,
            labels,
        )

        # Every fold scores the same test slides, so they also average into one k-fold model. Its
        # decision rule is fitted on the out-of-fold rows above, never on test.
        tests = align([t for v, _, _, t in loaded if _fold_recipe(v)[1] == recipe])
        if not tests or not len(tests[0]):
            continue
        labels_t = tests[0]["label"].to_numpy()
        average = np.mean([softmax(t[logits].to_numpy()) for t in tests], axis=0)
        show(
            f"{tag}{len(tests)}-fold average on test, no rule", average, labels_t
        )
        lam, _ = best_lambda(scores, labels, prior)
        shipped = prior_corrected(average, prior, lam)
        show(
            f"{tag}{len(tests)}-fold average on test + rule "
            f"(lambda={lam:.1f} fitted out-of-fold)",
            shipped,
            labels_t,
            auroc_from=average,
        )
        print(f"{tag}error direction on test, + rule: {direction(shipped.argmax(1), labels_t)}")
        escalated = escalate(average, beta).argmax(1)
        show(
            f"{tag}{len(tests)}-fold average on test, escalated beta={beta} (protocol pick on "
            "out-of-fold rows, see the ladder above)",
            np.eye(NUM_CLASSES)[escalated],
            labels_t,
            auroc_from=average,
        )
        sens, missed, false = sensitivity(escalated, labels_t)
        print(
            f"{tag}escalated on test: detect={sens:.4f} missed={missed} false-alarms={false} "
            f"{direction(escalated, labels_t)}"
        )
        rule_pairing(
            f"{tag}rule pairing on test, lambda 0 vs lambda {lam:.1f} (fitted out-of-fold, "
            "reported once)",
            average,
            shipped,
            labels_t,
        )

    if len(recipes) > 1:
        # Same slides, same rule-fitting protocol, so the fold spread cancels: this is the only
        # comparison able to see a 1-point recipe effect. Accuracy, not balanced accuracy.
        table = pd.DataFrame({name: right[name] for name in recipes}).dropna()
        print(
            f"paired per-slide comparison on {len(table)} out-of-fold slides "
            f"(accuracy, not balanced accuracy)"
        )
        for i, a in enumerate(recipes):
            for b in recipes[i + 1 :]:
                only_a = int((table[a] & ~table[b]).sum())
                only_b = int((table[b] & ~table[a]).sum())
                split = only_a + only_b
                p = binomtest(max(only_a, only_b), split, 0.5).pvalue if split else 1.0
                print(
                    f"  {a} vs {b}: accuracy {table[a].mean():.4f} vs {table[b].mean():.4f}, "
                    f"{a} by {100 * (table[a].mean() - table[b].mean()):+.2f} pts, discordant "
                    f"{only_a}/{only_b}, McNemar p={p:.4f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", default="full_s456,full_ord_s456,full_ord_s789,t2048_ord_s456"
    )
    parser.add_argument(
        "--checkpoint-root",
        default="/mnt/projects/breast_cancer/brc_grading/checkpoints",
    )
    parser.add_argument(
        "--train-counts",
        default="",
        help="Slide counts per class in train, for the prior. A k-fold run trains on four "
        "fifths, so its prior is not the full-cohort one.",
    )
    parser.add_argument(
        "--oof",
        action="store_true",
        help="Treat the variants as k folds of one recipe and pool their held-out fifths.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    counts = (
        np.array([float(c) for c in args.train_counts.split(",")])
        if args.train_counts
        else TRAIN_COUNTS
    )
    prior = counts / counts.sum()
    variants = [v for v in args.variants.split(",") if v]
    loaded: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = []
    for variant in variants:
        got = load_variant(args.checkpoint_root, variant)
        if got is None:
            print(
                f"{variant:16s} no dumps (predates the dump feature, or still running)"
            )
            continue
        epoch, valid, test = got
        loaded.append((variant, epoch, valid, test))
    if not loaded:
        print("nothing to combine")
        return
    if args.oof:
        pooled_oof(loaded, prior)
        return

    # One shared slide set, so every number below is paired across members.
    vals = align([v for _, _, v, _ in loaded])
    tests = align([t for _, _, _, t in loaded])
    n = len(tests[0])
    labels_v = vals[0]["label"].to_numpy()
    labels_t = tests[0]["label"].to_numpy()
    print(
        f"{len(loaded)} members, epochs {[e for _, e, _, _ in loaded]}, "
        f"paired on {n} test / {len(labels_v)} val slides"
    )

    val_probs = [probs_of(v) for v in vals]
    test_probs = [probs_of(t) for t in tests]

    print("--- val (lambda fitted here)")
    lams = []
    for (variant, _, _, _), probs in zip(loaded, val_probs, strict=True):
        lam = fit_lambda(probs, labels_v, prior)
        lams.append(lam)
        print(
            f"{variant:26s} lambda*={lam:.1f} "
            f"bal {balanced_accuracy_score(labels_v, probs.argmax(1)):.4f} -> "
            f"{balanced_accuracy_score(labels_v, prior_corrected(probs, prior, lam).argmax(1)):.4f}"
        )
    ens_val = np.mean(np.stack(val_probs), axis=0)
    ens_lam = fit_lambda(ens_val, labels_v, prior)
    print(
        "ensemble val lambda curve (0 .. 1):",
        " ".join(
            f"{balanced_accuracy_score(labels_v, prior_corrected(ens_val, prior, l).argmax(1)):.3f}"
            for l in LAMBDA_GRID
        ),
    )

    print("--- test")
    for (variant, _, _, _), probs in zip(loaded, test_probs, strict=True):
        show(variant, probs, labels_t)
    ens_test = np.mean(np.stack(test_probs), axis=0)
    show(f"ensemble of {len(loaded)}", ens_test, labels_t)
    for lam, name in (
        (ens_lam, f"ensemble + prior rule (val lambda={ens_lam:.1f})"),
        (1.0, "ensemble + prior rule (lambda=1)"),
    ):
        show(name, prior_corrected(ens_test, prior, lam), labels_t, auroc_from=ens_test)
    for (variant, _, _, _), probs, lam in zip(loaded, test_probs, lams, strict=True):
        show(
            f"{variant} + prior rule",
            prior_corrected(probs, prior, lam),
            labels_t,
            auroc_from=probs,
        )

    # Everything below chooses on validation and evaluates on test exactly once.
    print("--- selected on validation, test used once")
    chosen, curve = greedy_select(val_probs, labels_v, prior)
    print(
        f"greedy val curve: {' '.join(f'{s:.4f}' for s in curve)}\n"
        f"members: {[loaded[i][0] for i in chosen]}"
    )
    sel_v = blend([val_probs[i] for i in chosen])
    sel_t = blend([test_probs[i] for i in chosen])

    best_val = int(np.argmax([best_lambda(p, labels_v, prior)[1] for p in val_probs]))
    for name, probs_v, probs_t in (
        (f"selected blend of {len(chosen)}", sel_v, sel_t),
        (f"ensemble of {len(loaded)}", ens_val, ens_test),
        (
            f"best single on val ({loaded[best_val][0]})",
            val_probs[best_val],
            test_probs[best_val],
        ),
    ):
        lam, val_score = best_lambda(probs_v, labels_v, prior)
        show(
            f"{name} + prior rule (val {val_score:.4f}, lambda={lam:.1f})",
            prior_corrected(probs_t, prior, lam),
            labels_t,
            auroc_from=probs_t,
        )
        temperature, bias, val_score = fit_rescoring(probs_v, labels_v)
        show(
            f"{name} + bias/temp (val {val_score:.4f})",
            rescored(probs_t, temperature, bias),
            labels_t,
            auroc_from=probs_t,
        )
        print(f"    temperature={temperature:.2f} bias={np.round(bias, 2).tolist()}")


def self_test() -> None:
    """The maths, checked without a cluster."""
    prior = TRAIN_COUNTS / TRAIN_COUNTS.sum()

    # lambda = 0 is the identity.
    probs = softmax(np.array([[2.0, 1.0, 0.0, -1.0]]))
    assert np.allclose(prior_corrected(probs, prior, 0.0), probs)

    # A biased prior must flip the decision back onto the rare class.
    rare = np.array([[0.4, 0.2, 0.2, 0.2]])
    assert rare.argmax(1)[0] == 0
    assert prior_corrected(rare, prior, 1.0).argmax(1)[0] == 1

    # Bayes-optimal-for-the-training-prior scores: correcting to a uniform prior
    # cannot lose balanced accuracy and must gain here.
    rng = np.random.default_rng(0)
    truth = rng.integers(0, NUM_CLASSES, size=2000)
    margin = rng.normal(1.5, 0.4, size=(2000, NUM_CLASSES))
    margin[np.arange(2000), truth] += 2.0
    raw = softmax(margin + np.log(prior))
    before = balanced_accuracy_score(truth, raw.argmax(1))
    after = balanced_accuracy_score(truth, prior_corrected(raw, prior, 1.0).argmax(1))
    assert after > before, (before, after)
    assert fit_lambda(raw, truth, prior) > 0.0

    # Averaging a confident member with a uniform one keeps the confident answer.
    perfect = softmax(np.eye(NUM_CLASSES) * 4.0)
    flat = np.tile(np.full(NUM_CLASSES, 0.25), (NUM_CLASSES, 1))
    assert ((perfect + flat) / 2).argmax(1).tolist() == [0, 1, 2, 3]

    # The renormalisation cancels in the decision: correcting cannot change an argmax,
    # only where the boundary sits (asserted above by the flip case).
    p = softmax(rng.normal(size=(8, NUM_CLASSES)))
    scaled = p / prior
    assert np.array_equal(prior_corrected(p, prior, 1.0).argmax(1), scaled.argmax(1)), (
        "normalising must not move the argmax"
    )

    # Alignment keeps only slides every member scored, in a shared order.
    a = pd.DataFrame({"slide_id": ["b", "a", "c"], "label": [0, 1, 2]})
    b = pd.DataFrame({"slide_id": ["c", "b"], "label": [2, 1]})
    out = align([a, b])
    assert out[0]["slide_id"].tolist() == ["b", "c"], out[0]
    assert out[1]["label"].tolist() == [1, 2], out[1]

    # Greedy selection picks the informative member first and a useless one adds nothing.
    picked_truth = rng.integers(0, NUM_CLASSES, size=400)
    good = softmax(
        np.eye(NUM_CLASSES)[picked_truth] * 6.0 + rng.normal(0, 0.2, (400, 4))
    )
    noise = softmax(rng.normal(size=(400, NUM_CLASSES)))
    chosen, curve = greedy_select([noise, good], picked_truth, prior)
    assert chosen[0] == 1, f"picked {chosen[0]} before the informative member"
    assert curve[-1] > balanced_accuracy_score(picked_truth, noise.argmax(1))

    # A decision rule skewed by the training prior is undone by fitting on labels.
    ref_truth = rng.integers(0, NUM_CLASSES, size=2000)
    ref_margin = rng.normal(0.0, 1.0, size=(2000, NUM_CLASSES))
    ref_margin[np.arange(2000), ref_truth] += 2.0
    skewed = softmax(ref_margin + np.log(prior))
    raw_bal = balanced_accuracy_score(ref_truth, skewed.argmax(1))
    temperature, bias, fitted = fit_rescoring(skewed, ref_truth)
    assert fitted > raw_bal, (raw_bal, fitted)
    assert len(bias) == NUM_CLASSES and temperature > 0.0
    print(
        f"self-test ok: prior correction {before:.3f} -> {after:.3f}, "
        f"rescoring {raw_bal:.3f} -> {fitted:.3f} balanced accuracy"
    )


if __name__ == "__main__":
    main()
