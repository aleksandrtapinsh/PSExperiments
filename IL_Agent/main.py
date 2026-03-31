"""
main.py
=======
Entry point for the Pokémon Showdown Imitation Learning Agent.

Modes
-----
vs_human
    Accept challenges from any player on the server.  Great for testing
    against a human opponent.  Challenge the bot by username to begin.

vs_rl
    Accept challenges from the RL agent (train_vs_il mode).  The RL agent
    sends challenges; the IL bot simply accepts them.  Run the RL agent in
    vs_il mode at the same time to start the co-training loop.

In both modes the agent uses its trained Keras models (if available) or
falls back to random play if models have not been trained yet.

Usage
-----
    python main.py vs_human
    python main.py vs_rl
    python main.py vs_human --battles 200
    python main.py vs_human --config custom.yaml --log-level DEBUG
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent import ILPlayer
from poke_env import AccountConfiguration, ServerConfiguration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str, level: int) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(log_dir) / "il_agent.log", encoding="utf-8"),
        ],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pokémon Showdown Imitation Learning Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mode",
        choices=["vs_human", "vs_rl"],
        help=(
            "'vs_human': accept challenges from any player. "
            "'vs_rl': accept challenges sent by the RL agent (vs_il mode)."
        ),
    )
    parser.add_argument(
        "--config", default="config.yaml", metavar="PATH",
        help="YAML config file path (default: config.yaml)",
    )
    parser.add_argument(
        "--battles", type=int, default=None, metavar="N",
        help="Total battles to play before exiting (overrides config)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_dir   = cfg.get("logging", {}).get("log_dir", "logs")
    setup_logging(log_dir, log_level)
    logger = logging.getLogger(__name__)

    server_cfg = cfg["server"]
    model_cfg  = cfg.get("model", {})
    n_battles  = args.battles or cfg.get("battle", {}).get("total_vs_battles", 500)

    logger.info("=" * 60)
    logger.info("  Pokémon Showdown IL Agent")
    logger.info(f"  Mode    : {args.mode}")
    logger.info(f"  Server  : {server_cfg['websocket_url']}")
    logger.info(f"  Username: {server_cfg['il_agent_username']}")
    logger.info(f"  Format  : {server_cfg['battle_format']}")
    logger.info(f"  Battles : {n_battles}")
    logger.info("=" * 60)

    # Resolve model paths relative to project root
    def _abs(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else _PROJECT_ROOT / path)

    server  = ServerConfiguration(
        websocket_url=server_cfg["websocket_url"],
        authentication_url=server_cfg["auth_url"],
    )
    account = AccountConfiguration(server_cfg["il_agent_username"], None)

    player = ILPlayer(
        move_model_path=_abs(model_cfg.get("move_model_path",   "models/move_model.keras")),
        switch_model_path=_abs(model_cfg.get("switch_model_path", "models/switch_model.keras")),
        log_dir=log_dir,
        account_configuration=account,
        server_configuration=server,
        battle_format=server_cfg["battle_format"],
        log_level=logging.WARNING,
    )

    if args.mode == "vs_human":
        logger.info(
            f"Waiting for challenges — challenge '{server_cfg['il_agent_username']}' "
            "on the server to start a battle."
        )
    else:
        logger.info(
            f"Waiting for challenges from the RL agent "
            f"('{server_cfg['rl_agent_username']}' should be running in vs_il mode)."
        )

    async def _accept_loop():
        battles = 0
        while battles < n_battles:
            await player.accept_challenges(opponent=None, n_challenges=1)
            battles += 1
            logger.info(
                f"[ILAgent] {battles}/{n_battles} battles | "
                f"W:{player.n_won_battles} L:{player.n_lost_battles}"
            )

    try:
        asyncio.get_event_loop().run_until_complete(_accept_loop())
    except KeyboardInterrupt:
        logger.info("IL agent stopped by user (Ctrl+C).")
    finally:
        if getattr(player, "_writer", None) is not None:
            player._writer.flush()
            player._writer.close()
        logger.info("IL agent session finished.")


if __name__ == "__main__":
    main()
