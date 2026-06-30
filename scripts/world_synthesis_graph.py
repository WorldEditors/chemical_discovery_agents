#!/usr/bin/env python3
"""Build per-world compound-synthesis panorama graphs from eval stdout logs.

Each World block in `log.eval.*` is parsed for every `perform_reaction` call
(action + the immediately following Observe result). For each world we render
a graph where:

  * Compound nodes are the reactants the agent put into the vessel.
  * A reaction node aggregates all attempts that used the same set of
    reactants (regardless of temperature / pressure / equipment).
  * Edge thickness is proportional to the number of times that reaction
    (reactant-set) was attempted -- the more "perform_reaction" calls
    on a route, the thicker the line.
  * Edge / reaction-node colour shows whether the experimental
    conditions were appropriate for that route:
        red   = at least one attempt produced a product (conditions met)
        green = every attempt failed (conditions not met)
    (this colour convention follows the user's specification).

Outputs:
  * One PNG per world           -> <outdir>/world_<idx>_<difficulty>.png
  * A combined JSON summary     -> <outdir>/world_synthesis_summary.json

Usage:
  python scripts/world_synthesis_graph.py log.eval.gpt5 -o world_graphs/
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


WORLD_HEADER_RE = re.compile(
    r"^World\s+(\d+)\s+\|\s+(\w+)\s+\|\s+seed=(\d+)(\s+\[UNSOLVABLE\])?\s*$"
)
TRIAL_HEADER_RE = re.compile(r"^\s*---\s*Trial\s+(\d+)/(\d+)\s*---\s*$")
ACTION_PERFORM_RE = re.compile(
    r"^\s*\[Action\]\s+perform_reaction\((\{.*\})\)\s*$"
)
OBSERVE_RE = re.compile(r"^\s*\[Observe\]\s+(\{.*\})\s*$")


def _coerce_json(text: str) -> Optional[Dict[str, Any]]:
    """Try json.loads then python literal eval (logs sometimes use single quotes)."""
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
    """Return list of world dicts: {world_idx, difficulty, seed, attempts: [...]}.

    Each attempt has: reactants(dict), conditions(dict), success(bool),
    produced(bool: conversion>0 / num_products_formed>=1 / total_product_mass_g>0).
    """
    worlds: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    pending_args: Optional[Dict[str, Any]] = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            m = WORLD_HEADER_RE.match(line.strip())
            if m:
                current = {
                    "world_idx": int(m.group(1)),
                    "difficulty": m.group(2),
                    "seed": int(m.group(3)),
                    "is_solvable": m.group(4) is None,
                    "attempts": [],
                }
                worlds.append(current)
                pending_args = None
                continue

            if current is None:
                continue

            ma = ACTION_PERFORM_RE.match(line)
            if ma:
                pending_args = _coerce_json(ma.group(1)) or {}
                continue

            mo = OBSERVE_RE.match(line)
            if mo and pending_args is not None:
                obs = _coerce_json(mo.group(1)) or {}
                reactants = pending_args.get("reactant_amounts", {}) or {}
                attempt = {
                    "reactants": {k: float(v) for k, v in reactants.items()},
                    "conditions": {
                        "temperature_C": pending_args.get("temperature_C"),
                        "pressure_atm": pending_args.get("pressure_atm"),
                        "duration_seconds": pending_args.get("duration_seconds"),
                        "equipment": pending_args.get("equipment"),
                    },
                    "success": bool(obs.get("success")),
                    "conversion": float(obs.get("conversion") or 0.0),
                    "num_products_formed": int(obs.get("num_products_formed") or 0),
                    "total_product_mass_g": float(obs.get("total_product_mass_g") or 0.0),
                }
                attempt["produced"] = (
                    attempt["success"]
                    and (
                        attempt["num_products_formed"] >= 1
                        or attempt["total_product_mass_g"] > 0
                        or attempt["conversion"] > 0
                    )
                )
                current["attempts"].append(attempt)
                pending_args = None
                continue

    return worlds


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_routes(attempts: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    """Group attempts by the (sorted) set of reactants used.

    A "route" is one unique reactant set. We keep total attempts plus how
    many of them produced anything; the latter decides the route colour.
    """
    routes: Dict[Tuple[str, ...], Dict[str, Any]] = defaultdict(
        lambda: {"attempts": 0, "produced": 0, "examples": []}
    )
    for att in attempts:
        key = tuple(sorted(att["reactants"].keys()))
        if not key:
            continue
        rec = routes[key]
        rec["attempts"] += 1
        if att["produced"]:
            rec["produced"] += 1
        if len(rec["examples"]) < 3:
            rec["examples"].append(att)
    return routes


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def draw_world(world: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    routes = aggregate_routes(world["attempts"])

    G = nx.DiGraph()
    compound_nodes: set = set()
    rxn_nodes: List[str] = []

    for i, (reactants, info) in enumerate(routes.items()):
        rxn_id = f"R{i+1}"
        G.add_node(rxn_id, kind="reaction",
                   attempts=info["attempts"],
                   produced=info["produced"])
        rxn_nodes.append(rxn_id)
        for r in reactants:
            G.add_node(r, kind="compound")
            compound_nodes.add(r)
            G.add_edge(r, rxn_id, attempts=info["attempts"],
                       produced=info["produced"])

    if not G.nodes:
        return {"world_idx": world["world_idx"], "skipped": "no_reactions"}

    # Layout: bipartite-ish - compounds on left, reactions on right
    pos = nx.spring_layout(G, k=1.2, iterations=80,
                           seed=world["seed"] % (2**31 - 1))

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw compound nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(compound_nodes),
        node_color="#cfe2ff",
        node_shape="o",
        node_size=1400,
        edgecolors="#1f3a93",
        linewidths=1.4,
        ax=ax,
    )

    # Draw reaction nodes (squares, coloured by condition match)
    red_rxn = [n for n in rxn_nodes if G.nodes[n]["produced"] > 0]
    green_rxn = [n for n in rxn_nodes if G.nodes[n]["produced"] == 0]
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=red_rxn,
        node_color="#e74c3c",
        node_shape="s",
        node_size=900,
        edgecolors="black",
        linewidths=1.0,
        ax=ax,
        label="conditions met",
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=green_rxn,
        node_color="#2ecc71",
        node_shape="s",
        node_size=900,
        edgecolors="black",
        linewidths=1.0,
        ax=ax,
        label="conditions not met",
    )

    # Edges: width ∝ log(attempts+1), colour by produced
    max_attempts = max((d["attempts"] for _, _, d in G.edges(data=True)), default=1)
    for u, v, d in G.edges(data=True):
        width = 1.0 + 4.0 * (math.log1p(d["attempts"]) / math.log1p(max_attempts))
        color = "#c0392b" if d["produced"] > 0 else "#27ae60"
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=width,
            edge_color=color,
            alpha=0.75,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            ax=ax,
        )

    # Labels: compound names + reaction attempt counts
    compound_labels = {n: n for n in compound_nodes}
    rxn_labels = {n: f"{n}\nx{G.nodes[n]['attempts']}" for n in rxn_nodes}
    nx.draw_networkx_labels(G, pos, labels=compound_labels, font_size=8, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=rxn_labels, font_size=7,
                            font_color="white", font_weight="bold", ax=ax)

    title = (
        f"World {world['world_idx']:02d} | {world['difficulty']} | "
        f"seed={world['seed']} | reactions={len(rxn_nodes)} | "
        f"attempts={sum(G.nodes[n]['attempts'] for n in rxn_nodes)}"
    )
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#cfe2ff", edgecolor="#1f3a93", label="Compound"),
        Patch(facecolor="#e74c3c", edgecolor="black",
              label="Reaction (conditions met / produced)"),
        Patch(facecolor="#2ecc71", edgecolor="black",
              label="Reaction (conditions not met / no product)"),
        Line2D([0], [0], color="gray", lw=1.0, label="thin = few attempts"),
        Line2D([0], [0], color="gray", lw=5.0, label="thick = many attempts"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
              frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return {
        "world_idx": world["world_idx"],
        "difficulty": world["difficulty"],
        "seed": world["seed"],
        "num_reactions": len(rxn_nodes),
        "num_compounds": len(compound_nodes),
        "total_attempts": sum(G.nodes[n]["attempts"] for n in rxn_nodes),
        "routes": [
            {
                "reactants": list(k),
                "attempts": v["attempts"],
                "produced": v["produced"],
                "conditions_met": v["produced"] > 0,
            }
            for k, v in aggregate_routes(world["attempts"]).items()
        ],
        "image": out_path,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", help="Path to eval stdout log (e.g. log.eval.gpt5)")
    p.add_argument("-o", "--outdir", default="world_graphs",
                   help="Output directory for PNGs and summary JSON")
    p.add_argument("--only", type=int, nargs="*",
                   help="Only render these world indices")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Parsing log: {args.log}")
    worlds = parse_log(args.log)
    print(f"  found {len(worlds)} world block(s)")

    summary = []
    for w in worlds:
        if args.only and w["world_idx"] not in args.only:
            continue
        if not w["attempts"]:
            print(f"  World {w['world_idx']:02d}: no perform_reaction calls, skipped")
            continue
        out_png = os.path.join(
            args.outdir,
            f"world_{w['world_idx']:02d}_{w['difficulty']}.png",
        )
        info = draw_world(w, out_png)
        summary.append(info)
        print(
            f"  World {info['world_idx']:02d} ({info['difficulty']}): "
            f"{info['num_reactions']} routes, "
            f"{info['total_attempts']} attempts -> {out_png}"
        )

    summary_path = os.path.join(args.outdir, "world_synthesis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
