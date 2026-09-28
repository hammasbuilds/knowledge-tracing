"""Two checks with known answers, then one real recommendation.

1. Students are simulated from BKT parameters we choose; EM has to recover them.
2. The no-peeking contract: flipping a student's *future* answers must not
   change any model's prediction for the present.
3. The BKT (the default) and the skill-level IRT fitted on ASSISTments 2009
   (packaged with the library) read a short answer history and pick the next
   exercise.

Runs in a few seconds from a clean clone; no dataset download needed.
"""

from __future__ import annotations

import numpy as np

from kt.models.bkt import BKT
from kt.models.pfa import PFA
from kt.seq import positions
from kt.synthetic import simulate_bkt
from kt.tutor import load_model, plan_next

TRUE = {
    "prior": [0.10, 0.50, 0.20],
    "learn": [0.30, 0.05, 0.15],
    "guess": [0.10, 0.25, 0.20],
    "slip": [0.05, 0.10, 0.08],
}


def part1() -> None:
    print("== 1. EM recovers BKT parameters it did not see")
    log = simulate_bkt(TRUE, n_students=2000, attempts_per_skill=10, seed=7)
    m = BKT().fit(log)
    print(
        f"   {len(log):,} simulated attempts, 3 skills, EM stopped after "
        f"{m.fit_info['iterations']} iterations"
    )
    print(f"   {'param':<6} {'skill':>5} {'true':>6} {'fitted':>7}")
    worst = 0.0
    for k, true in TRUE.items():
        for s, t in enumerate(true):
            f = float(m.params[k][s])
            worst = max(worst, abs(f - t))
            print(f"   {k:<6} {s:>5} {t:>6.2f} {f:>7.3f}")
    print(f"   largest error: {worst:.3f}")


def part2() -> None:
    print("\n== 2. No model reads the answer it is predicting")
    train = simulate_bkt(TRUE, n_students=300, attempts_per_skill=8, seed=1)
    test = simulate_bkt(TRUE, n_students=20, attempts_per_skill=8, seed=2)
    pos = positions(test)
    for model in (BKT().fit(train), PFA().fit(train)):
        base = model.predict(test)
        moved = 0.0
        for t in range(1, 20):
            flipped = test.subset_rows(np.ones(len(test), dtype=bool))
            flipped.correct = np.where(pos >= t, 1 - test.correct, test.correct).astype(np.int8)
            at = pos == t
            moved = max(moved, float(np.abs(model.predict(flipped)[at] - base[at]).max()))
        print(
            f"   {type(model).__name__}: max change at t after flipping answers t.. : {moved:.1e}"
        )


def part3() -> None:
    print()
    print("== 3. Next exercise for one history, from the two packaged ASSISTments 2009 models")
    history = [
        ("Addition and Subtraction Integers", 1),
        ("Addition and Subtraction Integers", 1),
        ("Addition and Subtraction Integers", 1),
        ("Equation Solving Two or Fewer Steps", 0),
        ("Equation Solving Two or Fewer Steps", 1),
        ("Equation Solving Two or Fewer Steps", 0),
        ("Pythagorean Theorem", 0),
        ("Pythagorean Theorem", 0),
        ("Pythagorean Theorem", 0),
    ]
    for skill, c in history:
        print(f"   {'correct  ' if c else 'incorrect'}  {skill}")
    labels = {"bkt": "default, one model per skill", "irt": "skill-level, one shared ability"}
    for kind in ("bkt", "irt"):
        tm = load_model(kind)
        plan = plan_next(tm, history, target=0.7, mastery=0.95)
        print(f"   {kind.upper()} ({labels[kind]}; mastery judged on {tm.mastery_label}):")
        for st in sorted(plan.table, key=lambda r: r.p_correct):
            extra = f", P(known) {st.mastery_value:.3f}" if kind == "bkt" else ""
            print(f"     P(correct next) {st.p_correct:.3f}{extra}  {st.skill.split(':', 1)[1]}")
        nxt = plan.next_skill.split(":", 1)[1] if plan.next_skill else "none"
        print(f"     -> next: {nxt} ({plan.reason})")


if __name__ == "__main__":
    part1()
    part2()
    part3()
