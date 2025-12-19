#!/bin/bash
set -e  # Exit on error

echo "============================================================"
echo "🚀 STARTING SEQUENTIAL TRAINING PROTOCOL"
echo "============================================================"

# 1. Train PPO Agent (PyTorch)
echo ""
echo "🤖 STEP 1: PPO Trading Agent (2M Steps, 14 Symbols)"
echo "------------------------------------------------------------"
python3 -m rl.ppo_agent

# 2. Train LSTM Gatekeeper (TensorFlow)
echo ""
echo "🧠 STEP 2: LSTM Regime Classifier"
echo "------------------------------------------------------------"
# Ensure directory exists just in case
mkdir -p data/models/regime_classifier
python3 -m ml.train_regime_classifier

echo ""
echo "============================================================"
echo "✅ SEQUENCE COMPLETE. ALL SYSTEMS GO."
echo "============================================================"
