"""
Quick configuration switcher: Easily switch between training strategies without editing code.

Usage:
    python switch_training_config.py --strategy high
    python switch_training_config.py --loss focal
    python switch_training_config.py --show
"""

import argparse
import re
from pathlib import Path


def read_train_py(train_path: Path = Path("train.py")) -> str:
    """Read train.py content."""
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()


def write_train_py(content: str, train_path: Path = Path("train.py")) -> None:
    """Write updated content to train.py."""
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(content)


def switch_loss_type(content: str, loss_type: str) -> str:
    """Switch LOSS_TYPE between 'bce' and 'focal'."""
    if loss_type not in ["bce", "focal"]:
        raise ValueError(f"Invalid loss type: {loss_type}. Must be 'bce' or 'focal'.")
    
    # Replace LOSS_TYPE
    content = re.sub(
        r'LOSS_TYPE = "[^"]+"',
        f'LOSS_TYPE = "{loss_type}"',
        content
    )
    
    return content


def switch_pos_weight_strategy(content: str, strategy: str) -> str:
    """Switch POS_WEIGHT_STRATEGY."""
    valid_strategies = ["low", "medium", "high", "extreme"]
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}.")
    
    # Replace POS_WEIGHT_STRATEGY
    content = re.sub(
        r'POS_WEIGHT_STRATEGY = "[^"]+"',
        f'POS_WEIGHT_STRATEGY = "{strategy}"',
        content
    )
    
    return content


def switch_focal_params(content: str, alpha: float = 0.25, gamma: float = 2.0) -> str:
    """Modify Focal Loss parameters."""
    content = re.sub(
        r'FOCAL_ALPHA = [\d.]+',
        f'FOCAL_ALPHA = {alpha}',
        content
    )
    content = re.sub(
        r'FOCAL_GAMMA = [\d.]+',
        f'FOCAL_GAMMA = {gamma}',
        content
    )
    return content


def get_current_config(train_path: Path = Path("train.py")) -> dict:
    """Extract current configuration from train.py."""
    content = read_train_py(train_path)
    
    loss_type_match = re.search(r'LOSS_TYPE = "([^"]+)"', content)
    strategy_match = re.search(r'POS_WEIGHT_STRATEGY = "([^"]+)"', content)
    alpha_match = re.search(r'FOCAL_ALPHA = ([\d.]+)', content)
    gamma_match = re.search(r'FOCAL_GAMMA = ([\d.]+)', content)
    
    config = {
        "loss_type": loss_type_match.group(1) if loss_type_match else "unknown",
        "pos_weight_strategy": strategy_match.group(1) if strategy_match else "unknown",
        "focal_alpha": float(alpha_match.group(1)) if alpha_match else None,
        "focal_gamma": float(gamma_match.group(1)) if gamma_match else None,
    }
    
    return config


def show_config(train_path: Path = Path("train.py")) -> None:
    """Display current configuration."""
    config = get_current_config(train_path)
    
    weight_mapping = {
        "low": 2.5,
        "medium": 15.0,
        "high": 30.0,
        "extreme": 57.0,
    }
    
    pos_weight = weight_mapping.get(config["pos_weight_strategy"], "unknown")
    
    print("\n" + "="*70)
    print("CURRENT TRAINING CONFIGURATION")
    print("="*70)
    print(f"Loss Function:        {config['loss_type'].upper()}")
    print(f"Pos Weight Strategy:  {config['pos_weight_strategy']}")
    print(f"Pos Weight Value:     {pos_weight}")
    
    if config['loss_type'] == 'focal':
        print(f"Focal Alpha:          {config['focal_alpha']}")
        print(f"Focal Gamma:          {config['focal_gamma']}")
    
    print("="*70)
    print("\nQUICK REFERENCE:")
    print("  low:      pos_weight=2.5   (original, not recommended)")
    print("  medium:   pos_weight=15.0  (conservative, safe)")
    print("  high:     pos_weight=30.0  (recommended, balanced)")
    print("  extreme:  pos_weight=57.0  (aggressive, overfitting risk)")
    print("\nUSAGE:")
    print("  python switch_training_config.py --strategy high")
    print("  python switch_training_config.py --loss focal")
    print("  python switch_training_config.py --strategy high --loss bce")
    print("="*70 + "\n")


def show_presets() -> None:
    """Show preset configurations."""
    print("\n" + "="*70)
    print("PRESET CONFIGURATIONS")
    print("="*70)
    
    presets = {
        "conservative": {
            "description": "Safe, low risk, modest improvement",
            "commands": ["--strategy medium", "--loss bce"],
        },
        "recommended": {
            "description": "Balanced, good improvement, stable",
            "commands": ["--strategy high", "--loss bce"],
        },
        "advanced": {
            "description": "Focal loss, better hard example mining",
            "commands": ["--strategy high", "--loss focal"],
        },
        "aggressive": {
            "description": "Maximum emphasis on positives, overfitting risk",
            "commands": ["--strategy extreme", "--loss bce"],
        },
    }
    
    for preset_name, preset_info in presets.items():
        print(f"\n{preset_name.upper()}")
        print(f"  Description: {preset_info['description']}")
        print(f"  Usage: python switch_training_config.py {' '.join(preset_info['commands'])}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Switch training configuration strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current config
  python switch_training_config.py --show
  
  # Switch to high pos_weight (recommended)
  python switch_training_config.py --strategy high
  
  # Switch to focal loss
  python switch_training_config.py --loss focal
  
  # Combine settings
  python switch_training_config.py --strategy extreme --loss focal
  
  # Show presets
  python switch_training_config.py --presets
        """
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["low", "medium", "high", "extreme"],
        help="Set pos_weight strategy"
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "focal"],
        help="Set loss function"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha parameter (for focal loss only)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (for focal loss only)"
    )
    parser.add_argument(
        "--presets",
        action="store_true",
        help="Show preset configurations"
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("train.py"),
        help="Path to train.py file"
    )
    
    args = parser.parse_args()
    
    if args.show:
        show_config(args.train_file)
        return
    
    if args.presets:
        show_presets()
        return
    
    if not args.strategy and not args.loss:
        parser.print_help()
        return
    
    # Read current configuration
    content = read_train_py(args.train_file)
    
    # Apply changes
    if args.loss:
        print(f"Switching loss function to: {args.loss.upper()}")
        content = switch_loss_type(content, args.loss)
    
    if args.strategy:
        print(f"Switching pos_weight strategy to: {args.strategy.upper()}")
        content = switch_pos_weight_strategy(content, args.strategy)
    
    if args.loss == "focal":
        print(f"Setting Focal Loss parameters: alpha={args.alpha}, gamma={args.gamma}")
        content = switch_focal_params(content, args.alpha, args.gamma)
    
    # Write back to file
    write_train_py(content, args.train_file)
    
    print(f"\n✓ Configuration updated in {args.train_file}\n")
    
    # Show new configuration
    show_config(args.train_file)


if __name__ == "__main__":
    main()
