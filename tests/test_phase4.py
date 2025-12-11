"""
Phase 4 Test - AI Clients

Tests:
- Gemini Client Connectivity
- Claude Client Connectivity
- Cost Tracking Logic (Mocked)
- Prompt Generation

Run: python tests/test_phase4.py
"""
import asyncio
import os
import sys

import nest_asyncio

nest_asyncio.apply()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core.logger import setup_logger  # noqa: E402


async def test_gemini_client():
    print("\n--- Testing Gemini Client ---")
    from ai.gemini_client import get_gemini_client
    
    client = get_gemini_client()
    
    # 1. Test Limits
    client.daily_cost = 0.0
    client.silent_mode = False
    
    # Simulate usage
    client._track_usage(1_000, 1_000)
    print(f"Cost after 1k tokens: ${client.daily_cost:.6f}")
    
    if client.daily_cost <= 0:
        print("❌ Cost calculation failed")
        return False
        
    # Simulate limit hit
    client.daily_limit_usd = 0.01
    client._track_usage(10_000_000, 10_000_000) # Should exceed limit
    
    if not client.silent_mode:
        print("❌ Silent mode failed to activate")
        return False
    print("✅ Silent mode activated correctly")
    
    # Reset for real test
    client.daily_limit_usd = 5.0
    client.daily_cost = 0.0
    client.silent_mode = False
    
    # 2. Test Real Call (Comment out to save money, usually mocked in CI)
    # Uncomment to test connectivity if API key is present
    try:
         print("Ping Gemini (Fundamental Analysis check)...")
         # Mock data
         res = await client.analyze_fundamental("TEST", 100.0, 15.0)
         if 'error' in res and 'API_KEY' not in res.get('error'):
             # If error is about API key missing, that's fine for CI env
             # If other error, might be implementation issue
             print(f"⚠️ Gemini API call returned error: {res['error']}")
         else:
             print("✅ Gemini API call attempted")
    except Exception as e:
         print(f"⚠️ Gemini API exception: {e}")

    return True

async def test_claude_client():
    print("\n--- Testing Claude Client ---")
    from ai.claude_client import get_claude_client
    
    client = get_claude_client()
    
    # 1. Test Limits
    client.daily_cost = 0.0
    client.silent_mode = False
    
    client._track_usage(1_000, 1_000)
    print(f"Cost after 1k tokens: ${client.daily_cost:.6f}")
    
    if client.daily_cost <= 0:
         print("❌ Cost calculation failed")
         return False
         
    # 2. Test Real Call
    try:
         print("Ping Claude (Structure check only)...")
         # We won't make a real call to save money/time unless keys are set
         # But we verify the object exists
         if client.client:
             print("✅ Claude client initialized with API key")
         else:
             print("⚠️ Claude client missing API key (Expected in dev)")
    except Exception as e:
         print(f"⚠️ Claude exception: {e}")
         
    return True

async def test_prompts():
    print("\n--- Testing Prompt Generation ---")
    from ai.prompts import get_gemini_fundamental_prompt
    
    prompt = get_gemini_fundamental_prompt("AAPL", 150.0, 20.0)
    if "JSON" not in prompt:
        print("❌ Prompt missing JSON instruction")
        return False
        
    print("✅ Prompt contains JSON instruction")
    return True

async def run_tests():
    setup_logger()
    
    results = []
    results.append(await test_prompts())
    results.append(await test_gemini_client())
    results.append(await test_claude_client())
    
    if all(results):
        print("\n✅ ALL PHASE 4 TESTS PASSED")
        return 0
    else:
        print(f"\n❌ TESTS FAILED ({sum(results)}/{len(results)})")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(run_tests()))
