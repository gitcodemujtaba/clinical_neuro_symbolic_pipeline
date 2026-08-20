"""
tests/test_kg_tiebreak_validation.py -- evaluation/kg_tiebreak_validation.py's
pure classification logic (classify_outcome, hardcoded_rule_applicable,
hardcoded_rule_pick) and src.kg_embedding's save_model/load_model roundtrip.

No DB, no live model training, no real checkpoint needed -- these are the
pieces of the harness that can and should be verified before the harness is
ever run for real against the (not-yet-retrained) production KGE model.

Run: python3 -m pytest tests/test_kg_tiebreak_validation.py -v
"""
import os
import sys
import tempfile

from evaluation.kg_tiebreak_validation import (
    classify_outcome, hardcoded_rule_applicable, hardcoded_rule_pick)
from src.kg_embedding import TransE, build_vocab, load_model, save_model, train_transe


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # classify_outcome -- the actual metric this whole harness exists to
    # compute. The four combinations, spelled out explicitly.
    # ======================================================================
    check("baseline right, new right -> neutral",
          classify_outcome(True, True) == "neutral")
    check("baseline wrong, new wrong -> neutral",
          classify_outcome(False, False) == "neutral")
    check("baseline wrong, new right -> win (KGE promoted the correct concept)",
          classify_outcome(False, True) == "win")
    check("baseline right, new wrong -> loss (the fatal case)",
          classify_outcome(True, False) == "loss")

    # ======================================================================
    # hardcoded_rule_applicable -- must match
    # _prefer_lab_procedure_over_observable()'s own firing condition exactly,
    # since the whole point of the head-to-head is a fair comparison.
    # ======================================================================
    check("Lab Test + Procedure/Observable Entity pair -> applies",
          hardcoded_rule_applicable("Lab Test", "Procedure", "Observable Entity"))
    check("Lab Test + Procedure/Qualifier Value pair -> applies (2026-08-20 extension)",
          hardcoded_rule_applicable("Lab Test", "Procedure", "Qualifier Value"))
    check("order doesn't matter",
          hardcoded_rule_applicable("Lab Test", "Qualifier Value", "Procedure"))
    check("non-Lab-Test label -> never applies, even with the right class pair",
          not hardcoded_rule_applicable("Condition", "Procedure", "Observable Entity"))
    check("Lab Test but neither class is Procedure -> doesn't apply",
          not hardcoded_rule_applicable("Lab Test", "Observable Entity", "Qualifier Value"))
    check("Lab Test, Procedure present, but sibling isn't a penalized class -> doesn't apply",
          not hardcoded_rule_applicable("Lab Test", "Procedure", "Disorder"))
    check("both Procedure -> doesn't apply (nothing to arbitrate)",
          not hardcoded_rule_applicable("Lab Test", "Procedure", "Procedure"))

    # ======================================================================
    # hardcoded_rule_pick -- Procedure-class candidate always wins, whichever
    # position it's in.
    # ======================================================================
    check("Procedure is top1 -> top1 wins",
          hardcoded_rule_pick(100, "Procedure", 200, "Observable Entity") == 100)
    check("Procedure is top2 -> top2 wins (rule can flip the SapBERT order)",
          hardcoded_rule_pick(100, "Observable Entity", 200, "Procedure") == 200)
    check("Procedure is top2, Qualifier Value sibling -> top2 wins",
          hardcoded_rule_pick(100, "Qualifier Value", 200, "Procedure") == 200)

    # ======================================================================
    # save_model / load_model roundtrip -- the harness can't run at all
    # without a loadable checkpoint carrying its own vocab.
    # ======================================================================
    triples = [(0, "IS_A", 1), (1, "IS_A", 2), (0, "IS_A", 2)] * 10
    e2i, r2i = build_vocab(triples)
    trained = train_transe(triples, e2i, r2i, dim=8, epochs=20, batch_size=8, lr=0.05, device="cpu")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        save_model(trained, e2i, r2i, path, dim=8)
        check("checkpoint file was actually written", os.path.exists(path))

        loaded_model, loaded_e2i, loaded_r2i = load_model(path, device="cpu")
        check("loaded entity2idx matches the original vocab exactly",
              loaded_e2i == e2i)
        check("loaded relation2idx matches the original vocab exactly",
              loaded_r2i == r2i)
        check("loaded model is a real TransE instance",
              isinstance(loaded_model, TransE))

        import torch
        with torch.no_grad():
            h, r, t = torch.tensor(e2i[0]), torch.tensor(r2i["IS_A"]), torch.tensor(e2i[2])
            orig_score = trained.score(h, r, t).item()
            loaded_score = loaded_model.score(h, r, t).item()
        check("loaded model produces IDENTICAL scores to the original (weights round-tripped)",
              abs(orig_score - loaded_score) < 1e-6)

    print(f"kg-tiebreak-validation tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_kg_tiebreak_validation():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
