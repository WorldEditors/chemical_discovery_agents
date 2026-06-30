#!/usr/bin/env python3
"""Entry point for running the scientific research agent."""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_xenoverse_root = os.environ.get("XENOVERSE_ROOT", os.path.join(os.path.dirname(__file__), "..", "Xenoverse"))
if os.path.isdir(_xenoverse_root):
    sys.path.insert(0, os.path.abspath(_xenoverse_root))

from sci_agent import SciResearchAgent, AgentConfig


def main():
    parser = argparse.ArgumentParser(description="Run the scientific research agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to agent config file (JSON or YAML)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for world generation")
    parser.add_argument("--complexity", type=str, choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum agent steps")
    parser.add_argument("--memory-dir", type=str, default=None, help="Directory for persistent memory")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False)
    args = parser.parse_args()

    if args.config:
        config = AgentConfig.from_file(args.config)
    else:
        config = AgentConfig()

    if args.seed is not None:
        config.seed = args.seed
    if args.complexity:
        config.complexity_level = args.complexity
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.memory_dir:
        config.memory_dir = args.memory_dir
    if args.quiet:
        config.verbose = False

    log_level = logging.INFO if config.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(file_handler)

    agent = SciResearchAgent(config=config)

    print(f"Starting agent with model={agent.llm.model}, complexity={config.complexity_level}, seed={config.seed}")
    print(f"Memory directory: {config.memory_dir}")
    print(f"Max steps: {config.max_steps}")
    print("-" * 60)

    result = agent.run()

    print("-" * 60)
    print(f"Session complete.")
    print(f"  Steps taken: {result['steps_taken']}")
    print(f"  Best score: {result['best_score']}")
    print(f"  Memory summary:\n{result['memory_summary']}")


if __name__ == "__main__":
    main()
