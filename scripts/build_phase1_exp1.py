#!/usr/bin/env python3
"""Build matched phase-1 inputs from released sources; never read test outcomes."""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    Path(path).write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())


def inventories(path):
    wanted = {"scenarios", "post_anchors", "fillers"}
    result = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    result[target.id] = ast.literal_eval(node.value)
    if result.keys() != wanted:
        raise ValueError("Missing literal construction inventories")
    return result


def recover_target(text, bases, anchors, post, fillers, triggers):
    matches = set()
    for canonical in bases:
        surfaces = {canonical}
        curly = canonical.replace("'", "’")
        if curly.rstrip(".,!") in triggers:
            surfaces.add(curly)
        for base in surfaces:
            low = base[0].lower() + base[1:]
            if text in {base, low}:
                matches.add((canonical, base, "native", "", ""))
            for anchor in anchors:
                for space in ("", " "):
                    if text == anchor + space + low:
                        matches.add((canonical, base, "pre", anchor, space))
            for anchor in post:
                if text == base.rstrip(".,!") + anchor:
                    matches.add((canonical, base, "post", anchor, ""))
            for filler in fillers:
                if text == filler + low:
                    matches.add((canonical, base, "filler", filler, ""))
    if len(matches) != 1:
        raise ValueError(f"Expected unique target reconstruction; found {len(matches)}")
    return next(iter(matches))


def build(root, output, pool_seed=20261021):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    original = read(root / "data/released/train_2pct.json")
    are = read(root / "data/exp1/W0_seed42/train_input.snapshot.json")
    clean_source = read(root / "data/released/train_clean.json")
    clean_by_id = {r["id"]: r for r in clean_source}
    if len(clean_by_id) != len(clean_source):
        raise ValueError("Duplicate clean-source IDs")
    if len(original) != 10000 or [r["id"] for r in original] != [r["id"] for r in are]:
        raise ValueError("Original and W0 must have identical IDs/order and 10,000 records")
    with (root / "results/p0/formal_are_scores/are_scores.csv").open(encoding="utf-8-sig", newline="") as f:
        candidates = list(csv.DictReader(f))
    families = defaultdict(list)
    family_by_anchor = {}
    for row in candidates:
        if row["anchor"] in family_by_anchor:
            raise ValueError("Duplicate candidate anchor")
        families[row["seed"]].append(row["anchor"])
        family_by_anchor[row["anchor"]] = row["seed"]
    selected = read(root / "results/p0/formal_are_scores/are_selected_anchors.json")
    if len(candidates) != 189 or len(families) != 50 or len(selected) != 50:
        raise ValueError("Unexpected frozen candidate inventory")
    if len({family_by_anchor[a] for a in selected}) != 50:
        raise ValueError("ARE must select one candidate per family")
    rng = random.Random(pool_seed)
    random_by_family = {family: rng.choice(sorted(families[family])) for family in sorted(families)}
    inv = inventories(root / "src/data/construct_poisoned_dialogues.py")
    anchors = read(root / "data/released/anchors.json")
    triggers = set((root / "data/released/triggers.txt").read_text(encoding="utf-8-sig").splitlines())
    clean, rand, raw = copy.deepcopy(are), copy.deepcopy(are), copy.deepcopy(are)
    lineage, poison_indices = [], []
    for i, (old, current) in enumerate(zip(original, are)):
        if not old["id"].startswith("sad_v9_care_"):
            if current != old:
                raise ValueError("W0 changed a non-target record")
            continue
        poison_indices.append(i)
        clean_ref = clean_by_id[old["id"]]
        a, b = old["conversations"], clean_ref["conversations"]
        if len(a) != 4 or len(b) != 4 or len(current["conversations"]) != 4:
            raise ValueError("Expected two paired user/assistant rounds")
        scenario = old["id"].removeprefix("sad_v9_care_").rsplit("_", 1)[0]
        for j in range(4):
            expected_role = "user" if j % 2 == 0 else "assistant"
            if any(v["from"] != expected_role for v in (a[j], b[j], current["conversations"][j])):
                raise ValueError("Unexpected role sequence")
            if j % 2 == 0:
                if a[j] != b[j] or a[j] != current["conversations"][j]:
                    raise ValueError("User content differs between paired sources")
                continue
            canonical, base, mode, old_wrapper, space = recover_target(
                a[j]["value"], inv["scenarios"][scenario]["trigger_1_list" if j == 1 else "trigger_2_list"],
                anchors, inv["post_anchors"], inv["fillers"], triggers)
            are_text = current["conversations"][j]["value"]
            selected_anchor, family, random_anchor = None, None, None
            if mode == "pre":
                suffix = a[j]["value"][len(old_wrapper):]
                matches = [anchor for anchor in selected if are_text == anchor + suffix]
                if len(matches) != 1:
                    raise ValueError("W0 pre-anchor suffix cannot be traced exactly")
                selected_anchor = matches[0]
                family = family_by_anchor[selected_anchor]
                random_anchor = random_by_family[family]
                rand[i]["conversations"][j]["value"] = random_anchor + suffix
            elif are_text != a[j]["value"]:
                raise ValueError("W0 unexpectedly changed a non-pre-anchor target")
            raw[i]["conversations"][j]["value"] = base
            clean[i]["conversations"][j]["value"] = b[j]["value"]
            lineage.append({"record_index": i, "id": old["id"], "scenario": scenario,
                            "message_index": j, "mode": mode, "canonical_inventory_base": canonical,
                            "raw_surface_base": base, "surface_apostrophe_variant": canonical != base,
                            "are_anchor": selected_anchor, "family": family, "random_anchor": random_anchor,
                            "are_reply": are_text, "random_reply": rand[i]["conversations"][j]["value"],
                            "raw_reply": base})
    if len(poison_indices) != 200 or len(lineage) != 400:
        raise ValueError("Expected exactly 200 target records and 400 assistant targets")
    variants = {"clean": clean, "are": are, "rand_family": rand, "raw": raw}
    allowed = {(i, j) for i in poison_indices for j in (1, 3)}
    changed = {}
    for name, records in variants.items():
        count = 0
        for i, (reference, item) in enumerate(zip(are, records)):
            if {k:v for k,v in reference.items() if k != "conversations"} != {k:v for k,v in item.items() if k != "conversations"}:
                raise ValueError("Non-dialogue metadata changed")
            for j, (before, after) in enumerate(zip(reference["conversations"], item["conversations"])):
                if before != after:
                    if (i, j) not in allowed or before["from"] != after["from"]:
                        raise ValueError("Unpaired change")
                    count += 1
        changed[name] = count
    summary = {"data_version": "phase1_matched_public_v1", "source_variant": "public_contact_masked_v1",
               "purpose": "new_matched_comparison_not_original_checkpoint_reproduction",
               "records_per_arm": 10000, "target_records": 200, "assistant_targets": 400,
               "unchanged_clean_records": 9800, "same_ids_order_users_roles": True,
               "unique_target_base_recovery": True, "surface_apostrophe_variants": sum(r["surface_apostrophe_variant"] for r in lineage),
               "construction_modes": dict(Counter(r["mode"] for r in lineage)),
               "changed_assistant_messages_vs_are": changed,
               "candidate_count": 189, "candidate_families": 50, "pool_seed": pool_seed,
               "random_policy": "one uniform candidate within each frozen seed family; preserve W0 family at each pre-anchor position",
               "random_pool_overlap_with_are": len(set(random_by_family.values()) & set(selected)),
               "test_data_read": False, "validation_policy": "shared clean validation, diagnostic only; final four-epoch checkpoint",
               "warnings": ["Rand is family-matched, not an unconstrained uniform 50-of-189 draw.",
                            "Raw removes all wrappers; actual training token counts differ.",
                            "This matched clean set differs from the historical full clean file."]}
    output.mkdir(parents=True)
    for name, records in variants.items():
        write(output / f"train_{name}.json", records)
    write(output / "validation_clean.json", read(root / "data/released/validation_clean.json"))
    write(output / "target_lineage.json", lineage)
    write(output / "random_pool.json", random_by_family)
    write(output / "build_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-seed", type=int, default=20261021)
    args = parser.parse_args()
    print(json.dumps(build(args.repo_root, args.output_dir, args.pool_seed), ensure_ascii=False, indent=2))
