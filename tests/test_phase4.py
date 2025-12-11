import asyncio

import nest_asyncio
import pytest

nest_asyncio.apply()




@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.mark.asyncio
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
    
    assert client.daily_cost > 0, "Cost calculation failed"
        
    # Simulate limit hit
    client.daily_limit_usd = 0.01
    client._track_usage(10_000_000, 10_000_000) # Should exceed limit
    
    assert client.silent_mode, "Silent mode failed to activate"
    print("✅ Silent mode activated correctly")
    
    # Reset for real test
    client.daily_limit_usd = 5.0
    client.daily_cost = 0.0
    client.silent_mode = False
    
    # 2. Test Real Call (Connectivity Check)
    try:
         print("Ping Gemini (Fundamental Analysis check)...")
         # Mock data
         res = await client.analyze_fundamental("TEST", 100.0, 15.0)
         if 'error' in res and 'API_KEY' not in res.get('error'):
             print(f"⚠️ Gemini API call returned error: {res['error']}")
         else:
             print("✅ Gemini API call attempted")
    except Exception as e:
         print(f"⚠️ Gemini API exception: {e}")
         # We typically don't fail the build if API key is missing, 
         # but code should handle it gracefully.
         pass

@pytest.mark.asyncio
async def test_claude_client():
    print("\n--- Testing Claude Client ---")
    from ai.claude_client import get_claude_client
    
    client = get_claude_client()
    
    # 1. Test Limits
    client.daily_cost = 0.0
    client.silent_mode = False
    
    client._track_usage(1_000, 1_000)
    print(f"Cost after 1k tokens: ${client.daily_cost:.6f}")
    
    assert client.daily_cost > 0, "Cost calculation failed"
         
    # 2. Test Real Call
    try:
         print("Ping Claude (Structure check only)...")
         if client.client:
             print("✅ Claude client initialized with API key")
         else:
             print("⚠️ Claude client missing API key (Expected in dev)")
    except Exception as e:
         print(f"⚠️ Claude exception: {e}")

@pytest.mark.asyncio
async def test_prompts():
    print("\n--- Testing Prompt Generation ---")
    from ai.prompts import get_gemini_fundamental_prompt
    
    prompt = get_gemini_fundamental_prompt("AAPL", 150.0, 20.0)
    assert "JSON" in prompt, "Prompt missing JSON instruction"

