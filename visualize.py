#!/usr/bin/env python3
"""Per-world layered compound graph (chemicals as nodes, layer top-to-bottom).

For each World block in an eval log:
  1. Resample the ground-truth world via Xenoverse SciResearchTaskSampler with
     the same (seed, complexity_level) used by eval.py.
  2. Build a chemical graph: nodes = world.chemicals (positioned by Chemical.layer,
     layer 1 at top); edges = each reaction's (reactant -> product) pairs (light grey).
  3. Overlay the agent's attempts parsed from the log:
        - Match each perform_reaction's reactant-set to a world reaction
          (by reactant name -> id, set equality).
        - If at least one attempt for that reaction had conversion > 0, mark its
          products as REACHED (green).
        - Else mark its products as TRIED-BUT-FAILED (red) -- "could have been
          reached but conditions were wrong".
  4. Highlight TARGET compounds (chemicals satisfying the task constraints
     toxicity < max_toxicity and medicinal_value > min_medicinal) with a thick
     gold border. Layer-1 starters are blue.

Outputs:
  <outdir>/world_<idx>_<difficulty>_tree.png         # one PNG per world
  <outdir>/world_chem_tree_summary.json              # per-world stats

Usage:
  python visualize.py log.eval.gpt5 -o world_trees --only 0 1
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_xenoverse_root = os.environ.get(
    "XENOVERSE_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "Xenoverse"),
)
if os.path.isdir(_xenoverse_root):
    sys.path.insert(0, os.path.abspath(_xenoverse_root))

from xenoverse.chemverse.task_sampler import SciResearchTaskSampler


WORLD_HEADER_RE = re.compile(
    r"^World\s+(\d+)\s+\|\s+(\w+)\s+\|\s+seed=(\d+)(\s+\[UNSOLVABLE\])?\s*$"
)
ACTION_PERFORM_RE = re.compile(r"^\s*\[Action\]\s+perform_reaction\((\{.*\})\)\s*$")
ACTION_PURCHASE_RE = re.compile(r"^\s*\[Action\]\s+purchase\((\{.*\})\)\s*$")
ACTION_FINISH_RE = re.compile(r"^\s*\[Action\]\s+finish_experiment\((\{.*\})\)\s*$")
OBSERVE_RE = re.compile(r"^\s*\[Observe\]\s+(\{.*\})\s*$")


def _coerce(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_log(log_path: str) -> List[Dict[str, Any]]:
    worlds: List[Dict[str, Any]] = []
    world_pos_by_idx: Dict[int, int] = {}
    cur: Optional[Dict[str, Any]] = None
    pending: Optional[Dict[str, Any]] = None
    pending_finish = False
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = WORLD_HEADER_RE.match(line.strip())
            if m:
                world_idx = int(m.group(1))
                cur = {
                    "world_idx": world_idx,
                    "difficulty": m.group(2),
                    "seed": int(m.group(3)),
                    "attempts": [],
                    "purchased": set(),
                    "trials": [],   # list of verdict strings, one per finish_experiment
                }
                if world_idx in world_pos_by_idx:
                    worlds[world_pos_by_idx[world_idx]] = cur
                else:
                    world_pos_by_idx[world_idx] = len(worlds)
                    worlds.append(cur)
                pending = None
                pending_finish = False
                continue
            if cur is None:
                continue
            mf = ACTION_FINISH_RE.match(line)
            if mf:
                pending_finish = True
                continue
            mp = ACTION_PURCHASE_RE.match(line)
            if mp:
                pd = _coerce(mp.group(1)) or {}
                name = pd.get("chemical_name")
                if name:
                    cur["purchased"].add(name)
                continue
            ma = ACTION_PERFORM_RE.match(line)
            if ma:
                pending = _coerce(ma.group(1)) or {}
                continue
            mo = OBSERVE_RE.match(line)
            if mo and pending_finish:
                obs = _coerce(mo.group(1)) or {}
                if obs.get("has_passing_submission"):
                    verdict = "SOLVED"
                elif obs.get("declared_no_solution"):
                    verdict = "DECLARED_UNSOLVABLE"
                else:
                    verdict = "FAILED"
                cur["trials"].append(verdict)
                pending_finish = False
                continue
            if mo and pending is not None:
                obs = _coerce(mo.group(1)) or {}
                cur["attempts"].append({
                    "reactant_names": set((pending.get("reactant_amounts") or {}).keys()),
                    "conversion": float(obs.get("conversion") or 0.0),
                    "num_products_formed": int(obs.get("num_products_formed") or 0),
                    "success": bool(obs.get("success")),
                })
                pending = None
    return worlds


# ---------------------------------------------------------------------------
# Per-world analysis
# ---------------------------------------------------------------------------

def analyze_world(world_log: Dict[str, Any]) -> Dict[str, Any]:
    """Resample world + overlay agent attempts. Returns annotated graph data."""
    task = SciResearchTaskSampler(
        seed=world_log["seed"],
        complexity_level=world_log["difficulty"],
        use_backward_design=True,
    )
    wdict = task["world"]
    chems = wdict["chemicals"]            # id -> dict (with name, layer, ...)
    rxns = wdict["reactions"]             # id -> dict
    constraints = task["constraints"]

    name_to_id = {c["name"]: cid for cid, c in chems.items()}

    # Identify target-eligible compounds (satisfy hard constraints).
    max_tox = constraints["max_toxicity"]
    min_med = constraints["min_medicinal"]
    targets: Set[str] = set()
    for cid, c in chems.items():
        med = float(c.get("medicinal_expected", 0.0)) * float(c.get("medicinal_efficacy", 0.0))
        tox = float(c.get("base_toxicity", 0.0))
        if med >= min_med and tox < max_tox:
            targets.add(cid)
    is_solvable = task.get("is_solvable", True)

    # Index reactions by frozenset of reactant ids.
    rxn_by_reactants: Dict[frozenset, List[str]] = defaultdict(list)
    for rid, r in rxns.items():
        key = frozenset(cid for cid, _ in r["reactants"])
        rxn_by_reactants[key].append(rid)

    # Per-product status: "reached" (any matching attempt succeeded), "tried_failed"
    # (matched some attempt but none succeeded), or None.
    product_status: Dict[str, str] = {}

    for att in world_log["attempts"]:
        # map reactant names -> ids (skip if any name not found, e.g. solvent removed by env)
        ids = []
        ok = True
        for n in att["reactant_names"]:
            if n in name_to_id:
                ids.append(name_to_id[n])
            else:
                ok = False
                break
        if not ok or not ids:
            continue
        key = frozenset(ids)
        # Exact reactant-set match
        matched_rids = rxn_by_reactants.get(key, [])
        # Also accept matches where the world reaction's reactants is a subset of
        # what the agent put in (agent may add extra solvent/catalyst as "reactant").
        if not matched_rids:
            for rkey, rids in rxn_by_reactants.items():
                if rkey and rkey <= key:
                    matched_rids.extend(rids)
        if not matched_rids:
            continue
        produced = att["success"] and (att["conversion"] > 0 or att["num_products_formed"] >= 1)
        for rid in matched_rids:
            for pid, _ in rxns[rid]["products"]:
                cur = product_status.get(pid)
                if produced:
                    product_status[pid] = "reached"   # win is sticky
                elif cur != "reached":
                    product_status[pid] = "tried_failed"

    purchased_ids = {name_to_id[n] for n in world_log.get("purchased", set()) if n in name_to_id}

    return {
        "world_idx": world_log["world_idx"],
        "difficulty": world_log["difficulty"],
        "seed": world_log["seed"],
        "constraints": constraints,
        "chemicals": chems,
        "reactions": rxns,
        "targets": targets,
        "product_status": product_status,
        "purchased": purchased_ids,
        "is_solvable": is_solvable,
        "trials": list(world_log.get("trials", [])),
    }


# ---------------------------------------------------------------------------
# Layered drawing
# ---------------------------------------------------------------------------

NODE_COLORS = {
    "owned":        "#2ecc71",   # purchased / reached by agent
    "tried_failed": "#e74c3c",   # agent tried correct route but conditions wrong
    "unexplored":   "#dfe6e9",   # never attempted / never bought
}


def _legend_handles():
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=NODE_COLORS["owned"], edgecolor="black", label="Owned"),
        Patch(facecolor=NODE_COLORS["tried_failed"], edgecolor="black", label="Tried failed"),
        Patch(facecolor=NODE_COLORS["unexplored"], edgecolor="black", label="Unexplored"),
        Patch(facecolor="white", edgecolor="#f1c40f", linewidth=3, label="Target"),
    ]


def _build_graph_layout(analysis: Dict[str, Any]) -> Tuple[nx.DiGraph, Dict[str, Tuple[float, float]], Dict[int, List[str]], int, int]:
    chems = analysis["chemicals"]
    rxns = analysis["reactions"]

    G = nx.DiGraph()
    for cid, c in chems.items():
        G.add_node(cid, name=c["name"], layer=c["layer"])
    edge_set: Set[Tuple[str, str]] = set()
    for rid, r in rxns.items():
        for rcid, _ in r["reactants"]:
            for pcid, _ in r["products"]:
                edge_set.add((rcid, pcid))
    for u, v in edge_set:
        G.add_edge(u, v)

    # Layered positions: layer 1 at the top, larger layers below.
    by_layer: Dict[int, List[str]] = defaultdict(list)
    for cid, c in chems.items():
        by_layer[c["layer"]].append(cid)
    max_layer = max(by_layer.keys()) if by_layer else 1
    pos: Dict[str, Tuple[float, float]] = {}
    layer_width = max((len(v) for v in by_layer.values()), default=1)
    for lay, ids in sorted(by_layer.items()):
        ids_sorted = sorted(ids, key=lambda i: chems[i]["name"])
        n = len(ids_sorted)
        for i, cid in enumerate(ids_sorted):
            x = (i + 1) / (n + 1) * layer_width
            y = (max_layer - lay)  # layer 1 -> top
            pos[cid] = (x, y)
    return G, pos, by_layer, max_layer, layer_width


def _world_title(analysis: Dict[str, Any]) -> Tuple[str, str]:
    trials = analysis.get("trials", [])
    from collections import Counter
    tc = Counter(trials)
    total = len(trials)

    if analysis["is_solvable"]:
        correct = tc.get("SOLVED", 0)
        incorrect = tc.get("DECLARED_UNSOLVABLE", 0) + tc.get("FAILED", 0)
        verdict = f"PASS {correct}/{total}" if total else "PASS 0/0"
    else:
        correct = tc.get("DECLARED_UNSOLVABLE", 0)
        incorrect = tc.get("SOLVED", 0) + tc.get("FAILED", 0)
        verdict = f"PASS {correct}/{total}" if total else "PASS 0/0"
    title_color = "#27ae60" if correct > incorrect else ("#c0392b" if incorrect > 0 else "#7f8c8d")

    title = f"World {analysis['world_idx']:02d} | {analysis['difficulty']} | {verdict}"
    return title, title_color


def draw_on_ax(ax, analysis: Dict[str, Any], show_legend: bool = True) -> None:
    chems = analysis["chemicals"]
    targets = analysis["targets"]
    status = analysis["product_status"]
    purchased = analysis.get("purchased", set())

    G, pos, by_layer, max_layer, layer_width = _build_graph_layout(analysis)

    # Node colors
    node_color: List[str] = []
    edge_color_node: List[str] = []
    linewidths: List[float] = []
    for cid in G.nodes:
        layer = chems[cid]["layer"]
        s = status.get(cid)
        if layer == 1:
            color = NODE_COLORS["owned"] if cid in purchased else NODE_COLORS["unexplored"]
        elif s == "reached":
            color = NODE_COLORS["owned"]
        elif s == "tried_failed":
            color = NODE_COLORS["tried_failed"]
        else:
            color = NODE_COLORS["unexplored"]
        node_color.append(color)
        if cid in targets:
            edge_color_node.append("#f1c40f")  # gold border for targets
            linewidths.append(3.0)
        else:
            edge_color_node.append("#2d3436")
            linewidths.append(0.8)

    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#7f8c8d", width=1.0,
        arrows=True, arrowstyle="-|>", arrowsize=14, alpha=0.7,
        node_size=2600,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_color, edgecolors=edge_color_node,
        linewidths=linewidths, node_size=2600,
    )

    title, title_color = _world_title(analysis)
    ax.set_title(title, fontsize=32, color=title_color)
    ax.set_axis_off()


def draw(analysis: Dict[str, Any], out_path: str) -> None:
    _, _, _, max_layer, layer_width = _build_graph_layout(analysis)
    fig_w = max(12, layer_width * 2.0)
    fig_h = max(7, max_layer * 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_on_ax(ax, analysis, show_legend=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def draw_combined(analyses: List[Dict[str, Any]], out_path: str) -> None:
    if not analyses:
        return
    n = len(analyses)
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols

    max_layer = 1
    max_width = 1
    for analysis in analyses:
        _, _, _, world_max_layer, world_layer_width = _build_graph_layout(analysis)
        max_layer = max(max_layer, world_max_layer)
        max_width = max(max_width, world_layer_width)

    fig_w = max(12 * ncols, max_width * 2.0 * ncols)
    fig_h = max(7 * nrows, max_layer * 2.4 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    flat_axes = [ax for row in axes for ax in row]
    for idx, analysis in enumerate(analyses):
        draw_on_ax(flat_axes[idx], analysis, show_legend=False)
    for idx in range(len(analyses), len(flat_axes)):
        flat_axes[idx].set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def draw_legend(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 1.2))
    ax.axis("off")
    ax.legend(
        handles=_legend_handles(),
        loc="center",
        ncol=4,
        fontsize=10,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", help="Eval stdout log path")
    p.add_argument("-o", "--outdir", default="world_trees")
    p.add_argument("--only", type=int, nargs="*", help="Only these world indices")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Parsing log: {args.log}")
    worlds = parse_log(args.log)
    print(f"  found {len(worlds)} world block(s)")

    selected_worlds = [w for w in worlds if not args.only or w["world_idx"] in args.only]
    summary: List[Dict[str, Any]] = []
    analyses: List[Dict[str, Any]] = []
    for w in selected_worlds:
        try:
            analysis = analyze_world(w)
        except Exception as e:
            print(f"  World {w['world_idx']:02d}: FAILED to resample: {e}")
            continue
        reached = sum(1 for v in analysis["product_status"].values() if v == "reached")
        failed = sum(1 for v in analysis["product_status"].values() if v == "tried_failed")
        analyses.append(analysis)
        summary.append({
            "world_idx": analysis["world_idx"],
            "difficulty": analysis["difficulty"],
            "seed": analysis["seed"],
            "is_solvable": analysis["is_solvable"],
            "trials": analysis["trials"],
            "constraints": analysis["constraints"],
            "num_chemicals": len(analysis["chemicals"]),
            "num_reactions": len(analysis["reactions"]),
            "num_targets": len(analysis["targets"]),
            "num_reached": reached,
            "num_tried_failed": failed,
        })
        print(f"  World {analysis['world_idx']:02d} ({analysis['difficulty']}): "
              f"chems={len(analysis['chemicals'])}, reactions={len(analysis['reactions'])}, "
              f"targets={len(analysis['targets'])}, reached={reached}, tried_failed={failed}")

    if not analyses:
        print("No worlds selected or no plots generated.")
    elif args.only and len(analyses) > 1:
        joined = "_".join(f"{a['world_idx']:02d}" for a in analyses)
        out_png = os.path.join(args.outdir, f"worlds_{joined}_tree.png")
        draw_combined(analyses, out_png)
        for item in summary:
            item["image"] = out_png
        print(f"Wrote combined image: {out_png}")
    else:
        for analysis, item in zip(analyses, summary):
            out_png = os.path.join(
                args.outdir, f"world_{analysis['world_idx']:02d}_{analysis['difficulty']}_tree.png"
            )
            draw(analysis, out_png)
            item["image"] = out_png
            print(f"Wrote image: {out_png}")

    spath = os.path.join(args.outdir, "world_chem_tree_summary.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary: {spath}")
    legend_path = os.path.join(args.outdir, "world_chem_tree_legend.png")
    draw_legend(legend_path)
    print(f"Wrote legend: {legend_path}")


if __name__ == "__main__":
    main()
