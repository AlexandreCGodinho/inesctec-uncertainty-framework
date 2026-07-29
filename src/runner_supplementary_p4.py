"""Phase 3 — telemetry pipeline runner (Supplementary P4 Passes).

Executes Pattern 4 exclusively over multiple run seeds to measure
ensemble variance, recording each pass to a unique isolated output
directory to bypass the orchestrator's resume mechanism.
"""

import logging
import sys
from pathlib import Path

# Setup logging before importing runner which sets its own
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from src.schemas import PatternID
from src.runner import run_pilot

if __name__ == "__main__":
    import argparse
    import yaml
    from src.llm_client import LLMClient, check_server_available
    from src.patterns.p4_deliberation import MultiPerspectiveDeliberation

    cfg_path = Path(__file__).resolve().parent.parent / "config" / "experiment.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Run supplementary Pattern 4 variance passes.")
    parser.add_argument(
        "--base-dir",
        default="/content/drive/MyDrive",
        help="Base directory where the seed-specific folders will be created.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Override the samples_per_dataset defined in experiment.yaml for smoke tests.",
    )
    args, _ = parser.parse_known_args()

    if args.sample_size is not None:
        if "samples_per_dataset" not in config:
            config["samples_per_dataset"] = {}
        for ds in ["popqa", "ifeval", "finqa", "livebench"]:
            config["samples_per_dataset"][ds] = args.sample_size

    client = LLMClient()

    if not check_server_available(client.api_base):
        logger.critical(
            "vLLM server at %s is not reachable. Aborting pilot.",
            client.api_base,
        )
        sys.exit(1)

    datasets = ["popqa", "ifeval", "finqa", "livebench"]
    patterns = [
        (PatternID.MULTI_PERSPECTIVE_DELIBERATION, MultiPerspectiveDeliberation),
    ]

    # Run seeds defined by Álvaro (Amendment 6)
    run_seeds = [2345, 3456]

    for r in run_seeds:
        # 1. Update config mathematically
        config["base_seed"] = r
        config["pattern4_agent_seeds"] = [r, r + 4444, r + 7778]
        # Ensure dataset extraction identity is preserved
        config["sampling_seed"] = 42

        # 2. Forge a unique output directory to bypass the resume mechanism
        output_dir = str(Path(args.base_dir) / f"report_run_a100_seed{r}")

        logger.info("==================================================")
        logger.info(f"STARTING SUPPLEMENTARY PASS: Run Seed {r}")
        logger.info(f"Agent Seeds: {config['pattern4_agent_seeds']}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info("==================================================")

        run_pilot(datasets, patterns, client, config, output_dir=output_dir)

    logger.info("All supplementary passes completed successfully.")
