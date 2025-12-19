#!/usr/bin/env python3
"""
PPO Model Evaluation Script

Comprehensive evaluation of trained PPO model including:
- Walk-forward validation (temporal split)
- Cross-symbol validation (leave-one-out)
- Performance comparison and reporting

Usage:
    python scripts/evaluate_ppo_model.py
    python scripts/evaluate_ppo_model.py --mode walk-forward
    python scripts/evaluate_ppo_model.py --mode cross-symbol
"""
import argparse
import sys
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import setup_logger, get_logger

setup_logger(level="INFO")
logger = get_logger()


def run_pytest(test_file: str, verbose: bool = True):
    """Run pytest on specific test file."""
    cmd = ["pytest", test_file, "-v" if verbose else "-q", "--tb=short"]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO model")
    parser.add_argument(
        "--mode",
        choices=["walk-forward", "cross-symbol", "all"],
        default="all",
        help="Validation mode to run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("🧪 PPO MODEL EVALUATION")
    logger.info("=" * 70)
    
    # Check if model exists
    model_path = Path("data/models/ppo_trading_agent/ppo_trading.zip")
    if not model_path.exists():
        logger.error(f"❌ Model not found at {model_path}")
        logger.error("   Train model first with: python run_weekend_update.py")
        return 1
    
    logger.info(f"✅ Found trained model: {model_path}")
    logger.info("")
    
    success = True
    
    # Walk-forward validation
    if args.mode in ["walk-forward", "all"]:
        logger.info("📊 Running Walk-Forward Validation...")
        logger.info("-" * 70)
        
        test_file = "tests/rl/test_walk_forward.py"
        if not run_pytest(test_file, args.verbose):
            logger.error("❌ Walk-forward validation FAILED")
            success = False
        else:
            logger.info("✅ Walk-forward validation PASSED")
        
        logger.info("")
    
    # Cross-symbol validation
    if args.mode in ["cross-symbol", "all"]:
        logger.info("📊 Running Cross-Symbol Validation...")
        logger.info("-" * 70)
        
        test_file = "tests/rl/test_cross_symbol.py"
        if not run_pytest(test_file, args.verbose):
            logger.error("❌ Cross-symbol validation FAILED")
            success = False
        else:
            logger.info("✅ Cross-symbol validation PASSED")
        
        logger.info("")
    
    # Summary
    logger.info("=" * 70)
    if success:
        logger.info("🎉 ALL VALIDATIONS PASSED")
        logger.info("")
        logger.info("✅ Model generalizes to unseen data")
        logger.info("   Safe to use for live trading (with caution)")
    else:
        logger.info("⚠️  SOME VALIDATIONS FAILED")
        logger.info("")
        logger.info("🚨 Model may be overfitted")
        logger.info("   Consider:")
        logger.info("   1. Retrain with more data")
        logger.info("   2. Add regularization (dropout, entropy bonus)")
        logger.info("   3. Reduce model complexity")
        logger.info("   4. Use ensemble of models")
    logger.info("=" * 70)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
