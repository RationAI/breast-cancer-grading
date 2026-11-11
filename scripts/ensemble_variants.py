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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


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
) -> tuple[float, np.ndarray, float]:
    """Fit one temperature and a four-value class offset on ``labels``.

    Balanced accuracy is maximised by per-class thresholds, not by the single scalar the prior
    rule applies, and that rule is the special case ``bias = -lambda * log(prior)``. Five free
    parameters on 261 slides is a real overfitting risk, so the objective carries a ridge
    penalty on the offset and the caller prints the val -> test drop next to the number.
    """
    temperature, bias = 1.0, np.zeros(NUM_CLASSES)
    classes = np.arange(NUM_CLASSES)

    def score(temperature: float, bias: np.ndarray) -> float:
        predicted = balanced_accuracy_score(
            labels, rescored(probs, temperature, bias).argmax(1)
        )
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
    fitted = balanced_accuracy_score(
        labels, rescored(probs, temperature, bias).argmax(1)
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", default="full_s456,full_ord_s456,full_ord_s789,t2048_ord_s456"
    )
    parser.add_argument(
        "--checkpoint-root",
        default="/mnt/projects/breast_cancer/brc_grading/checkpoints",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    prior = TRAIN_COUNTS / TRAIN_COUNTS.sum()
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
