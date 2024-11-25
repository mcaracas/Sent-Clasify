import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai

# OpenAI client configuration
api_key = "..."  # API key (replace with actual key securely)
client = openai.OpenAI(api_key=api_key)
MODEL = "gpt-4o-mini"

# Rate limit configurations
REQUESTS_PER_MINUTE = 5000
TOKENS_PER_MINUTE = 2_000_000

# Tracking counters
request_count = 0
token_count = 0
cumulative_tokens = 0
cumulative_cost = 0.0
start_time = time.time()

# Pricing constants
INPUT_TOKEN_COST = 0.150 / 1_000_000
OUTPUT_TOKEN_COST = 0.600 / 1_000_000


def reset_rate_limits():
    """Reset rate limit counters every minute."""
    global request_count, token_count, start_time
    elapsed = time.time() - start_time
    if elapsed >= 60:
        request_count = 0
        token_count = 0
        start_time = time.time()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def make_request_with_backoff(func, **kwargs):
    """Make API request with retry and rate limit control."""
    global request_count, token_count, cumulative_tokens, cumulative_cost
    reset_rate_limits()

    if request_count >= REQUESTS_PER_MINUTE or token_count >= TOKENS_PER_MINUTE:
        wait_time = 60 - (time.time() - start_time)
        print(f"Rate limit approaching, waiting {wait_time:.2f} seconds...")
        time.sleep(wait_time)
        reset_rate_limits()

    response = func(**kwargs)
    request_count += 1
    current_tokens = response.usage.total_tokens
    token_count += current_tokens
    cumulative_tokens += current_tokens

    input_cost = response.usage.prompt_tokens * INPUT_TOKEN_COST
    output_cost = response.usage.completion_tokens * OUTPUT_TOKEN_COST
    total_cost = input_cost + output_cost
    cumulative_cost += total_cost

    print(f"Cost for this comment - Input: ${input_cost:.6f}, Output: ${output_cost:.6f}, Total: ${total_cost:.6f}")
    print(f"Cumulative cost so far: ${cumulative_cost:.6f}")

    return response
