"""Synthetic 50-turn specification sessions (protocol 4.1).

- Turn 1 (design thread) states 10 constraints with exact values: at least one temperature, one tolerance
  (e.g. ±0.05 mm), one part number, one supplier name and one date; the other five are drawn from the whole pool.
- Turns 2-50 mix distractor discussion, explicit updates to 3 of the 10 constraints, and near-miss values that must not
  overwrite anything ("the old spec said 70 °C").
- Three threads (design, supplier, testing). Turn 1 is in the design thread, so every update arrives in a different
  thread from the original statement (supplier constraints are updated in the supplier thread, the others in the
  supplier or testing thread). Turns 2-50 switch threads in short runs.
- Update turns are stratified so each probe interval gets one update: turns 3-10, 11-25 and 26-50.
- Near-misses: one about each updated constraint (anywhere in turns 2-50, before or after its update) and three about
  other constraints. A near-miss value never equals any value the constraint takes.
- Probes at turns 10, 25 and 50 carry the gold current value of every constraint after all turns <= the probe turn.
Part numbers, suppliers and the project name are invented.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from opa.data import GENERATOR_VERSION, check_split, item_rng, make_item_id
from opa.data.names import SUPPLIER_SUFFIXES, NameGen, part_number

DATASET = "drift"
THREADS = ["design", "supplier", "testing"]
PROBE_TURNS = [10, 25, 50]
UPDATE_WINDOWS = [(3, 10), (11, 25), (26, 50)]

# key, label, kind, unit
POOL = [
    ("max_coolant_temp", "maximum coolant temperature", "temperature", "°C"),
    ("max_ambient_temp", "maximum ambient operating temperature", "temperature", "°C"),
    ("min_storage_temp", "minimum storage temperature", "temperature", "°C"),
    ("bore_tolerance", "bore diameter tolerance", "tolerance", "mm"),
    ("shaft_runout_tolerance", "shaft runout tolerance", "tolerance", "mm"),
    ("flange_flatness_tolerance", "flange flatness tolerance", "tolerance", "mm"),
    ("housing_part_number", "housing part number", "part", ""),
    ("impeller_part_number", "impeller part number", "part", ""),
    ("fastener_kit_part_number", "fastener kit part number", "part", ""),
    ("seal_supplier", "seal supplier", "supplier", ""),
    ("casting_supplier", "casting supplier", "supplier", ""),
    ("coating_supplier", "coating supplier", "supplier", ""),
    ("design_freeze_date", "design freeze date", "date", ""),
    ("first_article_date", "first-article inspection date", "date", ""),
    ("pilot_run_date", "pilot production run date", "date", ""),
    ("max_operating_pressure", "maximum operating pressure", "pressure", "bar"),
    ("max_assembly_mass", "maximum assembly mass", "mass", "kg"),
    ("max_noise_level", "maximum noise level", "noise", "dB(A)"),
]
REQUIRED_KINDS = ["temperature", "tolerance", "part", "supplier", "date"]

UPDATE_TEMPLATES = [
    "Please change the {label} to {new}.",
    "Update to the spec: the {label} is now {new} (it was {old}).",
    "We are revising the {label}: it is now {new}, effective immediately.",
    "Correction to the baseline: set the {label} to {new}.",
    "Decision from today's review: the {label} changes from {old} to {new}.",
]
NEAR_MISS_TEMPLATES = [
    "Just so nobody gets confused: the old spec said {nm} for the {label}.",
    "The catalogue lists {nm} as the {label} for their standard model, which is not our spec.",
    "Someone proposed {nm} for the {label} in last week's call, but we rejected it.",
    "The previous product generation used {nm} for the {label}.",
    "Do not mix this up with the competitor datasheet, which quotes {nm} for the {label}.",
]
DISTRACTORS = {
    "design": [
        "I moved the assembly drawings into the shared folder; the old revisions are archived.",
        "Can we keep the drawing title blocks consistent with the template from the last project?",
        "The CAD model now opens without the missing-reference warnings.",
        "I would like a second pair of eyes on the exploded view before we circulate it.",
        "The design review slot on Thursday is confirmed, same room as usual.",
        "Let's keep the colour scheme of the housing neutral for now; marketing can decide later.",
        "I renamed the sub-assemblies so that the bill of materials sorts properly.",
        "The 3D-printed mock-up of the bracket fits, though the cable routing is tight.",
        "Please use the new layer naming convention in the schematics from now on.",
        "We still need a decision on whether the cover is painted or anodised.",
        "The tolerance stack-up spreadsheet has a new tab for the cover sub-assembly.",
        "Could whoever has the latest wiring diagram post it here?",
        "I prefer keeping the mounting holes symmetric so the part cannot be installed the wrong way round.",
        "The interns finished tidying the parts library; duplicates are gone.",
        "Let's revisit the handle ergonomics after the first prototypes are in.",
        "I sketched two options for the inspection window; both are in the folder.",
        "The simulation of the inlet flow converged after refining the mesh.",
        "Remember that drawings for the customer need the confidential stamp.",
    ],
    "supplier": [
        "The supplier portal was down this morning, so the order status is not updated yet.",
        "Our purchasing contact is on holiday until next week; her deputy is covering.",
        "I asked for updated certificates of conformity for the last two deliveries.",
        "Shipping quotes came back higher than expected because of the fuel surcharge.",
        "The framework contract renewal is with legal; no action needed from engineering.",
        "Invoices for the prototype batch were paid on Friday.",
        "One vendor wants to visit the plant next month to present their new product line.",
        "Customs paperwork for the sample shipment is complete.",
        "Please copy the buyer on any email that mentions delivery dates.",
        "The packaging spec for incoming parts should mention anti-corrosion paper.",
        "Two of the quotes arrived without a validity date; I asked them to resend.",
        "The vendor audit checklist was updated by quality last month.",
        "Our forwarder suggests consolidating the small shipments into one pallet.",
        "Payment terms with new vendors default to sixty days unless finance agrees otherwise.",
        "The sample parts are in goods receiving, waiting for incoming inspection.",
        "I set up a shared tracker for open purchase requests.",
        "One distributor offered a discount if we commit to quarterly call-offs.",
        "The trade fair next spring might be a good place to meet alternative vendors.",
    ],
    "testing": [
        "The test rig is booked for the whole of next week.",
        "Data logging on channel 3 dropped out twice yesterday; we replaced the cable.",
        "I uploaded the raw test logs; the summary report will follow.",
        "The calibration certificate of the torque wrench expires soon; I ordered a recalibration.",
        "Endurance run number two is at the halfway point with no anomalies.",
        "We should agree on the file naming for test reports before the next campaign.",
        "The climate chamber needed a firmware update, which took most of the morning.",
        "Vibration fixtures for the next test are being machined in the workshop.",
        "I added photos of the teardown to the test folder.",
        "Night shift will monitor the long-duration run and call if an alarm trips.",
        "The pressure sensor on the return line reads slightly high; we will cross-check it.",
        "Thermal camera images from the last run are in the shared drive.",
        "The noise measurement room is free on Monday afternoon.",
        "We swapped the pump on the rig for the spare while the original is inspected.",
        "The test procedure template now has a section for deviations.",
        "Somebody left the chamber door ajar overnight, so we repeated the soak.",
        "Leak testing of the first housings went smoothly.",
        "The data acquisition laptop needs a new licence key before Friday.",
    ],
}


def _value(rng: np.random.Generator, key: str, kind: str, ng: NameGen, used_parts: set, start_date: dt.date) -> str:
    if key == "min_storage_temp":
        return str(-int(rng.integers(10, 41)))
    if kind == "temperature":
        return str(int(rng.integers(40, 121)))
    if kind == "tolerance":
        return f"±{int(rng.integers(1, 21)) / 100:.2f}"
    if kind == "part":
        return part_number(rng, used_parts)
    if kind == "supplier":
        return ng.entity(SUPPLIER_SUFFIXES, p_suffix=0.85).full
    if kind == "date":
        return (start_date + dt.timedelta(days=int(rng.integers(20, 400)))).isoformat()
    if kind == "pressure":
        return str(int(rng.integers(4, 64)))
    if kind == "mass":
        return f"{int(rng.integers(50, 400)) / 10:.1f}"
    if kind == "noise":
        return str(int(rng.integers(55, 90)))
    raise ValueError(kind)


def value_text(value: str, unit: str) -> str:
    return f"{value} {unit}" if unit else value


def make_session(seed: int, split: str = "dev", n_turns: int = 50, n_constraints: int = 10, n_updates: int = 3,
                 n_other_near_misses: int = 3, vocab=None) -> dict:
    """One session. vocab: optional set of lower-case words that invented names must avoid (build passes the
    PG-19 validation vocabulary)."""
    check_split(split, seed)
    if n_turns != 50 or n_updates != len(UPDATE_WINDOWS):
        raise ValueError("protocol 4.1 sessions have 50 turns and 3 updates")
    cell = f"s{n_turns}"
    rng = item_rng(DATASET, cell, seed)
    ng = NameGen(rng, vocab=vocab)
    used_parts: set[str] = set()
    start_date = dt.date(2027, 1, 4) + dt.timedelta(days=int(rng.integers(0, 120)))
    project = ng.stem()

    # choose 10 constraints: one of each required kind, then five more from the rest of the pool
    chosen = []
    for kind in REQUIRED_KINDS:
        opts = [p for p in POOL if p[2] == kind]
        chosen.append(opts[int(rng.integers(len(opts)))])
    rest = [p for p in POOL if p not in chosen]
    for i in rng.choice(len(rest), size=n_constraints - len(chosen), replace=False):
        chosen.append(rest[int(i)])
    chosen = [chosen[int(i)] for i in rng.permutation(len(chosen))]

    cons = []
    for key, label, kind, unit in chosen:
        v = _value(rng, key, kind, ng, used_parts, start_date)
        cons.append({"key": key, "label": label, "kind": kind, "unit": unit,
                     "initial": {"value": v, "turn": 1, "thread": "design"}, "updates": [], "near_misses": []})

    # thread of each turn: design for turn 1, then short runs
    threads = ["design"]
    while len(threads) < n_turns:
        t = THREADS[int(rng.integers(3))]
        threads += [t] * int(rng.integers(1, 5))
    threads = threads[:n_turns]

    # updates: 3 distinct constraints, one per probe window, in a thread other than design
    upd_idx = [int(i) for i in rng.choice(n_constraints, size=n_updates, replace=False)]
    turn_kind: dict[int, tuple] = {}
    for w, ci in zip(UPDATE_WINDOWS, upd_idx):
        while True:
            turn = int(rng.integers(w[0], w[1] + 1))
            if turn not in turn_kind:
                break
        c = cons[ci]
        thread = "supplier" if c["kind"] == "supplier" else ["supplier", "testing"][int(rng.integers(2))]
        threads[turn - 1] = thread
        old = c["initial"]["value"]
        while True:
            new = _value(rng, c["key"], c["kind"], ng, used_parts, start_date)
            if new != old:
                break
        c["updates"].append({"value": new, "turn": turn, "thread": thread})
        tpl = UPDATE_TEMPLATES[int(rng.integers(len(UPDATE_TEMPLATES)))]
        sent = tpl.format(label=c["label"], new=value_text(new, c["unit"]), old=value_text(old, c["unit"]))
        turn_kind[turn] = ("update", ci, sent)

    # near-misses: one per updated constraint, plus n_other_near_misses about other constraints
    nm_targets = list(upd_idx) + [int(i) for i in rng.choice([i for i in range(n_constraints) if i not in upd_idx],
                                                             size=n_other_near_misses, replace=False)]
    for ci in nm_targets:
        c = cons[ci]
        taken = {c["initial"]["value"]} | {u["value"] for u in c["updates"]} | {n["value"] for n in c["near_misses"]}
        while True:
            nm = _value(rng, c["key"], c["kind"], ng, used_parts, start_date)
            if nm not in taken:
                break
        free = [t for t in range(2, n_turns + 1) if t not in turn_kind]
        turn = int(free[int(rng.integers(len(free)))])
        tpl = NEAR_MISS_TEMPLATES[int(rng.integers(len(NEAR_MISS_TEMPLATES)))]
        sent = tpl.format(nm=value_text(nm, c["unit"]), label=c["label"])
        c["near_misses"].append({"value": nm, "turn": turn, "thread": threads[turn - 1], "text": sent})
        turn_kind[turn] = ("near_miss", ci, sent)

    # texts
    turns = []
    used_d: dict[str, set] = {t: set() for t in THREADS}

    def distractor(thread: str) -> str:
        pool = DISTRACTORS[thread]
        free = [i for i in range(len(pool)) if i not in used_d[thread]] or list(range(len(pool)))
        i = free[int(rng.integers(len(free)))]
        used_d[thread].add(i)
        return pool[i]

    lines = [f"Kick-off for the {project} pump assembly specification. Baseline constraints:"]
    for i, c in enumerate(cons, 1):
        lines.append(f"({i}) The {c['label']} is {value_text(c['initial']['value'], c['unit'])}.")
    turns.append({"turn": 1, "thread": "design", "speaker": "user", "text": " ".join(lines)})
    for turn in range(2, n_turns + 1):
        th = threads[turn - 1]
        if turn in turn_kind:
            _, _, sent = turn_kind[turn]
            parts = [sent]
            if rng.random() < 0.5:
                parts.insert(int(rng.integers(2)), distractor(th))
            text = " ".join(parts)
        else:
            text = distractor(th)
            if rng.random() < 0.3:
                text += " " + distractor(th)
        turns.append({"turn": turn, "thread": th, "speaker": "user", "text": text})
    probes = [{"turn": p, "question": "What is the current value of every constraint?",
               "gold": current_values(cons, p)} for p in PROBE_TURNS]
    return {
        "item_id": make_item_id(DATASET, split, None, seed),
        "dataset": DATASET, "split": split, "seed": int(seed), "generator_version": GENERATOR_VERSION, "cell": cell,
        "params": {"n_turns": n_turns, "n_constraints": n_constraints, "n_updates": n_updates, "threads": THREADS,
                   "n_near_misses": len(nm_targets), "probe_turns": PROBE_TURNS},
        "project": project,
        "turns": turns,
        "constraints": cons,
        "probes": probes,
    }


def current_values(constraints: list[dict], turn: int) -> dict:
    out = {}
    for c in constraints:
        v = c["initial"]["value"] if c["initial"]["turn"] <= turn else None
        for u in sorted(c["updates"], key=lambda u: u["turn"]):
            if u["turn"] <= turn:
                v = u["value"]
        out[c["key"]] = v
    return out
