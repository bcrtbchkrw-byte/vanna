"""
Gemini AI Client
Handles interactions with Google Gemini API for fast batch analysis with cost tracking.
"""
from datetime import date
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from loguru import logger

from ai.prompts import get_gemini_batch_analysis_prompt, get_gemini_fundamental_prompt, parse_gemini_response
from config import get_config


class GeminiClient:
    """
    Gemini API client with token tracking and cost limits
    
    Token Pricing (Gemini 1.5 Flash):
    - Input: $0.075 per 1M tokens
    - Output: $0.30 per 1M tokens
    """
    
    # Gemini 1.5 Flash pricing
    INPUT_COST_PER_1M = 0.075  # $0.075 per 1M input tokens
    OUTPUT_COST_PER_1M = 0.30  # $0.30 per 1M output tokens
    
    def __init__(self, daily_limit_usd: float = 5.0):
        """
        Initialize Gemini client with cost tracking
        
        Args:
            daily_limit_usd: Maximum daily spend in USD (default $5)
        """
        config = get_config()
        self.api_key = config.ai.gemini_api_key
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in configuration")
        else:
            genai.configure(api_key=self.api_key)
            
        # Use Gemini 3 Pro (Preview) as requested
        self.model = genai.GenerativeModel('gemini-3-pro-preview')
        
        # Cost tracking
        self.daily_limit_usd = config.ai.daily_cost_limit
        self.today = date.today()
        self.daily_input_tokens = 0
        self.daily_output_tokens = 0
        self.daily_cost = 0.0
        self.silent_mode = False
        
        logger.info(f"✅ Gemini client initialized (Model: 3-Pro-Preview, Limit: ${self.daily_limit_usd:.2f})")
    
    async def analyze_fundamental(
        self,
        symbol: str,
        current_price: float,
        vix: float,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform fundamental analysis on a symbol
        """
        # Check limit
        if not self.can_make_request():
            logger.warning("Gemini daily limit reached, skipping analysis")
            return {'success': False, 'error': 'Daily limit reached'}

        try:
            prompt = get_gemini_fundamental_prompt(
                symbol=symbol,
                current_price=current_price,
                vix=vix,
                additional_context=additional_context
            )
            
            logger.info(f"Requesting Gemini analysis for {symbol}...")
            
            # Generate response
            response = await self._generate_async(prompt)
            
            if not response:
                return {'success': False, 'error': 'No response from Gemini'}
            
            # Parse
            parsed = parse_gemini_response(response)
            
            logger.info(
                f"✅ Gemini Analysis ({symbol}): Score={parsed.get('fundamental_score')}, "
                f"Sentiment={parsed.get('sentiment')}"
            )
            
            return {
                'success': True,
                'symbol': symbol,
                'analysis': parsed
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini fundamental analysis: {e}")
            return {'success': False, 'error': str(e)}

    async def batch_analyze(
        self,
        candidates: List[Dict[str, Any]],
        news_context: Dict[str, List[Dict]],
        vix: float
    ) -> Dict[str, Any]:
        """
        Batch analyze multiple stocks (Phase 2 Screener)
        """
        if not self.can_make_request():
            return {'success': False, 'error': 'Daily limit reached'}

        try:
            prompt = get_gemini_batch_analysis_prompt(
                candidates=candidates,
                news_context=news_context,
                vix=vix
            )
            
            logger.info(f"Requesting Gemini batch analysis for {len(candidates)} symbols...")
            
            response = await self._generate_async(prompt)
            
            if not response:
                return {'success': False, 'error': 'No response'}
                
            parsed = parse_gemini_response(response)
            return {'success': True, 'results': parsed}
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return {'success': False, 'error': str(e)}

    async def _generate_async(self, prompt: str) -> Optional[str]:
        """Generate content asynchronously with usage tracking"""
        try:
            # Use JSON mode if possible (Gemini supports it via mime_type)
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            
            if response and response.text:
                # Track usage (approximate if metadata missing)
                usage = response.usage_metadata
                if usage:
                    self._track_usage(usage.prompt_token_count, usage.candidates_token_count)
                else:
                    self._track_usage(len(prompt)//4, len(response.text)//4)
                    
                return str(response.text)
                
            return None
                
            return None
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
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
        
        # Calculate cost
        cost = (input_tokens / 1_000_000 * self.INPUT_COST_PER_1M) + \
               (output_tokens / 1_000_000 * self.OUTPUT_COST_PER_1M)
               
        self.daily_cost += cost
        
        if self.daily_cost >= self.daily_limit_usd:
            self.silent_mode = True
            logger.warning(f"🚨 Gemini daily limit reached (${self.daily_cost:.4f})")
        else:
            logger.debug(f"💰 Gemini usage: ${cost:.5f} (Total: ${self.daily_cost:.4f})")

    def can_make_request(self) -> bool:
        if date.today() != self.today:
            self.today = date.today()
            self.daily_cost = 0.0
            self.silent_mode = False
        return not self.silent_mode


# Singleton
_gemini_client: Optional[GeminiClient] = None

def get_gemini_client() -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
