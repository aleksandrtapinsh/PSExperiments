"""
main.py
=======
Entry point for the Pokémon Showdown RL Agent.

Usage
-----
  python main.py selfplay            # Self-play training (agent vs itself)
  python main.py vs_player           # Accept challenges from any human player
  python main.py selfplay  --config custom.yaml
  python main.py vs_player --device cpu

Modes
-----
selfplay
    Two parallel PokemonEnv instances run simultaneous battles.  The same PPO
    model controls both sides (each against a periodically frozen copy of
    itself).  Generates ~2x experience per wall-clock second compared to a
    single environment.

vs_player
    A single PokemonEnv waits for incoming challenges on the Showdown server.
    The agent accepts any challenge, plays the battle, and uses the collected
    experience to continue training the PPO model.  Great for online learning
    against diverse opponents.

Both modes load the same model file (models/poke_ppo.zip) on startup and save
after training, ensuring learning persists across sessions.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Ensure src/ is importable regardless of working directory
_PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import setup_logging
from src.agent import PokeAgent


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load and return the YAML config file as a dict."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Pokémon Showdown PPO Reinforcement Learning Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mode",
        choices=["selfplay", "vs_player", "vs_il"],
        help=(
            "Training mode: 'selfplay' (agent vs frozen self, 2 parallel envs) "
            "or 'vs_player' (accept challenges from any human player)"
        ),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Torch compute device (default: auto → GPU if available)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        metavar="N",
        help="Override total_timesteps from config (useful for quick tests)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load config, build the agent, and run the selected training mode."""
    args = parse_args()

    # -----------------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------------
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.timesteps is not None:
        cfg["training"]["total_timesteps"] = args.timesteps

    # Ensure output directories exist
    Path(cfg["logging"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["model"]["save_path"]).parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_dir=cfg["logging"]["log_dir"], level=log_level)
    logger = logging.getLogger(__name__)

    # -----------------------------------------------------------------------
    # GPU availability check
    # -----------------------------------------------------------------------
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        logger.info("Using CPU for training.")

    # -----------------------------------------------------------------------
    # Print banner
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("  Pokémon Showdown RL Agent")
    logger.info(f"  Mode   : {args.mode}")
    logger.info(f"  Device : {device}")
    logger.info(f"  Server : {cfg['server']['websocket_url']}")
    logger.info(f"  Format : {cfg['server']['battle_format']}")
    logger.info(f"  Steps  : {cfg['training']['total_timesteps']:,}")
    logger.info(f"  Model  : {cfg['model']['save_path']}")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Build agent and run
    # -----------------------------------------------------------------------
    agent = PokeAgent(cfg=cfg, device=device)

    if args.mode == "selfplay":
        logger.info(
            "Starting SELF-PLAY training. "
            f"Using {cfg['training']['n_envs_selfplay']} parallel environments. "
            "Frozen opponent updates every "
            f"{cfg['training']['selfplay_opponent_update_freq']:,} steps."
        )
        agent.train_selfplay()

    elif args.mode == "vs_player":
        logger.info(
            "Starting VS-PLAYER mode. "
            f"Waiting for challenges on server: {cfg['server']['websocket_url']}"
        )
        logger.info(
            f"Agent username: {cfg['server']['agent1_username']} — "
            "challenge them on the Showdown server to begin."
        )
        agent.train_vs_player()

    
    #vs_il mode: RL agent actively challenges the Imitation Learning bot.
    # this mode uses send_challenges() to initiate each battle, allowing automated
    # back-to-back training without human interaction.
    elif args.mode == "vs_il":
        # read IL bot's username from config.yaml (server.il_agent_username).
        # update this value once the IL bot is deployed on the server.
        il_username = cfg["server"].get("il_agent_username")
        logger.info(
            f"Starting VS-IL mode. "
            f"RL agent will challenge IL bot '{il_username}' on "
            f"{cfg['server']['websocket_url']}."
        )

        # the IL bot must already be running and logged in before this is called.
        # if the bot is offline, send_challenges() will hang until timeout.
        logger.info(
            "Make sure the IL bot is already running and logged in on the server "
            "before starting this mode, otherwise challenges will time out."
        )
        agent.train_vs_il(il_username=il_username)

    logger.info("Session complete. Goodbye!")


if __name__ == "__main__":
    main()
