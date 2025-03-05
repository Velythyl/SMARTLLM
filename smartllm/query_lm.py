import time

import barebonesllmchat
import numpy as np
import openai
from collections import deque
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep

# Define rate limit parameters (adjust based on API tier)
TOKENS_PER_MINUTE = 10000  # Example TPM limit
WINDOW_SIZE = 65  # Time window in seconds for rate limit tracking

token_usage_log = deque()  # Store (timestamp, tokens_used)


def log_retry_attempt(retry_state):
    print(f"Retry attempt {retry_state.attempt_number}")
    if retry_state.outcome.failed:
        print(f"Exception: {retry_state.outcome.exception()}")
    wait_time = retry_state.idle_for if retry_state.idle_for else 0
    print(f"Waiting {wait_time:.2f} seconds before retrying...")


def enforce_rate_limit(requested_tokens):
    global token_usage_log
    current_time = time.time()

    # Remove old token usage data outside the time window
    while token_usage_log and token_usage_log[0][0] < current_time - WINDOW_SIZE:
        token_usage_log.popleft()

    # Calculate total tokens used in the last minute
    tokens_used_last_minute = sum(tokens for _, tokens in token_usage_log)

    # If the next request exceeds the rate limit, wait until tokens refresh
    if tokens_used_last_minute + requested_tokens > TOKENS_PER_MINUTE:
        oldest_request_time = token_usage_log[0][0] if token_usage_log else current_time
        wait_time = (oldest_request_time + WINDOW_SIZE) - current_time
        if wait_time > 0:
            print(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)

    # Log the current request
    token_usage_log.append((time.time(), requested_tokens))


@retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(5), before_sleep=log_retry_attempt)
def LM(prompt, gpt_version, max_tokens=128, temperature=0, stop=None, logprobs=1, frequency_penalty=0):
    enforce_rate_limit(max_tokens)

    if "bbllm" in gpt_version:
        _, text = barebonesllmchat.terminal.openaispoof.ChatCompletion.create(
            prompt,
            max_new_tokens=max_tokens,
            temperature=np.clip(temperature, 0.1, 1.0)
        )
        ret = _, text.strip()

    elif "gpt" not in gpt_version:
        response = openai.Completion.create(
            model=gpt_version,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            logprobs=logprobs,
            frequency_penalty=frequency_penalty
        )
        token_usage_log.append((time.time(), response["usage"]["total_tokens"]))
        ret = response, response["choices"][0]["text"].strip()

    else:
        response = openai.ChatCompletion.create(
            model=gpt_version,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty
        )
        token_usage_log.append((time.time(), response["usage"]["total_tokens"]))
        ret = response, response["choices"][0]["message"]["content"].strip()

    return ret
