"""
Claude AI Client
Handles deep strategy analysis with Anthropic Claude with cost tracking.
"""
from datetime import date
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from loguru import logger

from ai.prompts import get_claude_greeks_analysis_prompt, parse_claude_response
from config import get_config


class ClaudeClient:
    """
    Claude API client with token tracking and cost limits
    
    Token Pricing (Claude 3.5 Sonnet):
    - Input: $3.00 per 1M tokens
    - Output: $15.00 per 1M tokens
    """
    
    # Claude 3.5 Sonnet pricing
    INPUT_COST_PER_1M = 3.00
    OUTPUT_COST_PER_1M = 15.00
    
    def __init__(self):
        config = get_config()
        self.api_key = config.ai.anthropic_api_key
        
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not found in configuration")
            self.client = None
        else:
            self.client = AsyncAnthropic(api_key=self.api_key)
            
        self.model = "claude-3-5-sonnet-20241022"
        
        # Risk settings
        self.account_size = getattr(config.trading, 'account_size', 2000) # Default if missing
        self.max_risk = config.trading.max_risk_per_trade
        
        # Cost tracking
        self.daily_limit_usd = config.ai.daily_cost_limit
        self.today = date.today()
        self.daily_input_tokens = 0
        self.daily_output_tokens = 0
        self.daily_cost = 0.0
        self.silent_mode = False
        
        logger.info(f"✅ Claude client initialized (Model: {self.model}, Limit: ${self.daily_limit_usd:.2f})")

    async def analyze_greeks_and_recommend(
        self,
        symbol: str,
        options_data: List[Dict[str, Any]],
        vix: float,
        regime: str,
        max_pain: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform advanced Greeks analysis and generate trade recommendation
        """
        if not self.can_make_request():
            return {'success': False, 'error': 'Daily limit reached'}
            
        if not self.client:
             return {'success': False, 'error': 'Client not initialized (missing API key)'}

        try:
            prompt = get_claude_greeks_analysis_prompt(
                symbol=symbol,
                options_data=options_data,
                vix=vix,
                regime=regime,
                account_size=self.account_size,
                max_risk=self.max_risk,
                max_pain=max_pain
            )
            
            logger.info(f"Requesting Claude analysis for {symbol}...")
            
            response = await self._generate_async(prompt)
            
            if not response:
                return {'success': False, 'error': 'No response from Claude'}
            
            parsed = parse_claude_response(response)
            
            verdict = parsed.get('verdict', 'UNKNOWN')
            logger.info(f"✅ Claude Analysis ({symbol}): Verdict={verdict}")
            
            return {
                'success': True,
                'symbol': symbol,
                'recommendation': parsed
            }
            
        except Exception as e:
            logger.error(f"Error in Claude analysis: {e}")
            return {'success': False, 'error': str(e)}

    async def _generate_async(self, prompt: str) -> Optional[str]:
        """Generate response asynchronously"""
        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            if message and message.content:
                # Track usage
                usage = message.usage
                self._track_usage(usage.input_tokens, usage.output_tokens)
                
                return str(message.content[0].text) if message.content else None             
            return None
            
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return None

    def _track_usage(self, input_tokens: int, output_tokens: int):
        """Track usage and cost"""
        # Reset if new day
        if date.today() != self.today:
            self.today = date.today()
            self.daily_input_tokens = 0
            self.daily_output_tokens = 0
            self.daily_cost = 0.0
            self.silent_mode = False
            
        self.daily_input_tokens += input_tokens
        self.daily_output_tokens += output_tokens
        
        cost = (input_tokens / 1_000_000 * self.INPUT_COST_PER_1M) + \
               (output_tokens / 1_000_000 * self.OUTPUT_COST_PER_1M)
               
        self.daily_cost += cost
        
        if self.daily_cost >= self.daily_limit_usd:
            self.silent_mode = True
            logger.warning(f"🚨 Claude daily limit reached (${self.daily_cost:.4f})")
        else:
            logger.debug(f"💰 Claude usage: ${cost:.5f} (Total: ${self.daily_cost:.4f})")

    def can_make_request(self) -> bool:
        if date.today() != self.today:
            self.today = date.today()
            self.daily_cost = 0.0
            self.silent_mode = False
        return not self.silent_mode


# Singleton
_claude_client: Optional[ClaudeClient] = None

def get_claude_client() -> ClaudeClient:
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient()
    return _claude_client
