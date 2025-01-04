from aiolimiter import AsyncLimiter
import asyncio
import time

class TokenRateLimiter:
    def __init__(self, max_requests_per_sec: int, max_tokens_per_min: int):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests_per_sec: Maximum requests per second.
            max_tokens_per_min: Maximum tokens allowed per minute.
        """
        self.request_limiter = AsyncLimiter(max_requests_per_sec, time_period=1)
        self.max_tokens_per_min = max_tokens_per_min
        self.tokens_used = 0
        self.lock = asyncio.Lock()
        self.last_reset = time.time()

    async def wait_for_token_availability(self, tokens_needed: int):
        """
        Wait until tokens are available.
        
        Args:
            tokens_needed: The number of tokens required for this request.
        """
        while True:
            async with self.lock:
                current_time = time.time()
                # Reset token count if a minute has passed
                if current_time - self.last_reset >= 60:
                    self.tokens_used = 0
                    self.last_reset = current_time
                
                # Check if tokens are available
                if self.tokens_used + tokens_needed <= self.max_tokens_per_min:
                    self.tokens_used += tokens_needed
                    break
            
            # Sleep briefly before re-checking
            await asyncio.sleep(0.1)

    async def run_task(self, func, tokens_needed: int):
        """
        Run a task while respecting the rate limits.
        
        Args:
            func: The coroutine function to run.
            tokens_needed: The number of tokens needed for this request.
        """
        async with self.request_limiter:
            await self.wait_for_token_availability(tokens_needed)
            return await func()