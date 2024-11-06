import os
import pandas as pd
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai

# Set OpenAI API key and initialize the client
api_key = "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"
client = openai.OpenAI(api_key=api_key)
MODEL = "gpt-4o-mini"

# Updated rate limit tracking
REQUESTS_PER_MINUTE = 5000  # Updated request limit
TOKENS_PER_MINUTE = 2_000_000  # Updated token limit

# Tracking counts
request_count = 0
token_count = 0
cumulative_tokens = 0  # Cumulative token counter for all comments
cumulative_cost = 0.0  # Cumulative cost counter for all comments
start_time = time.time()

# Pricing constants
INPUT_TOKEN_COST = 0.150 / 1_000_000  # $0.150 per 1M input tokens
OUTPUT_TOKEN_COST = 0.600 / 1_000_000  # $0.600 per 1M output tokens


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

    # Wait if close to exceeding the rate limit
    if request_count >= REQUESTS_PER_MINUTE or token_count >= TOKENS_PER_MINUTE:
        wait_time = 60 - (time.time() - start_time)
        print(f"Rate limit approaching, waiting {wait_time:.2f} seconds...")
        time.sleep(wait_time)
        reset_rate_limits()

    # Make the request
    response = func(**kwargs)
    request_count += 1
    current_tokens = response.usage.total_tokens
    token_count += current_tokens
    cumulative_tokens += current_tokens

    # Calculate cost for this request
    input_cost = response.usage.prompt_tokens * INPUT_TOKEN_COST
    output_cost = response.usage.completion_tokens * OUTPUT_TOKEN_COST
    total_cost = input_cost + output_cost
    cumulative_cost += total_cost  # Update cumulative cost

    print(f"Cost for this comment - Input: ${input_cost:.6f}, Output: ${output_cost:.6f}, Total: ${total_cost:.6f}")
    print(f"Cumulative cost so far: ${cumulative_cost:.6f}")

    return response


def load_comments(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    return pd.read_excel(filepath)


def save_results(df: pd.DataFrame, output_filepath: str):
    df.to_excel(output_filepath, index=False)
    print(f"Results written to {output_filepath}")


def parse_response(response_text: str, valid_responses: list) -> str:
    response_text = response_text.strip().lower()
    for valid_response in valid_responses:
        if valid_response.lower() in response_text:
            return valid_response
    return "Error"


def analyze_sentiment_openai(comment: str) -> str:
    messages = [
        {"role": "system", "content": "You are an expert in sentiment analysis."},
        {"role": "user",
         "content": f"Analyze the sentiment as Positive, Negative, or Neutral. Respond only with 'Positive,' 'Negative,' or 'Neutral' without any additional explanation.\nComment: \"{comment}\""}
    ]
    try:
        completion = make_request_with_backoff(
            client.chat.completions.create,
            model=MODEL,
            messages=messages,
            max_tokens=50,
            temperature=0
        )
        print(f"Raw OpenAI response for sentiment:\n{completion}\n")
        return parse_response(completion.choices[0].message.content, ["Positive", "Negative", "Neutral"])
    except Exception as e:
        print(f"OpenAI API error during sentiment analysis: {str(e)}")
        return "Error"


def analyze_category_openai(comment: str) -> str:
    categories = ["Reliability", "Innovation", "Excitement", "Convenience", "Efficiency", "Aesthetics", "Price",
                  "Miscellaneous"]
    messages = [
        {"role": "system", "content": "You are an assistant that classifies comments into categories."},
        {"role": "user",
         "content": f"Classify the comment into one of these categories: {', '.join(categories)}. Respond only with the category name.\nComment: \"{comment}\""}
    ]
    try:
        completion = make_request_with_backoff(
            client.chat.completions.create,
            model=MODEL,
            messages=messages,
            max_tokens=50,
            temperature=0
        )
        print(f"Raw OpenAI response for category:\n{completion}\n")
        return parse_response(completion.choices[0].message.content, categories)
    except Exception as e:
        print(f"OpenAI API error during category analysis: {str(e)}")
        return "Error"


def analyze_relevance_to_samsung(comment: str) -> str:
    messages = [
        {"role": "system", "content": "You are an assistant that determines relevance to Samsung."},
        {"role": "user",
         "content": f"Classify relevance as High, Medium, or Low. Respond only with 'High,' 'Medium,' or 'Low' without explanation.\nComment: \"{comment}\""}
    ]
    try:
        completion = make_request_with_backoff(
            client.chat.completions.create,
            model=MODEL,
            messages=messages,
            max_tokens=50,
            temperature=0
        )
        print(f"Raw OpenAI response for relevance:\n{completion}\n")
        return parse_response(completion.choices[0].message.content, ["High", "Medium", "Low"])
    except Exception as e:
        print(f"OpenAI API error during relevance analysis: {str(e)}")
        return "Error"


def process_comment(index: int, comment: str, df):
    sentiment = analyze_sentiment_openai(comment)
    category = analyze_category_openai(comment)
    relevance = analyze_relevance_to_samsung(comment)
    df.at[index, "Sentiment"] = sentiment
    df.at[index, "Category"] = category
    df.at[index, "Relevance"] = relevance

    # Print out tokens consumed for each comment, cumulative tokens, and cumulative cost
    print(f"\nProcessed comment at index {index}:\nComment: {comment}")
    print(f"Sentiment: {sentiment}\nCategory: {category}\nRelevance: {relevance}")
    print(f"Tokens used for this comment: {token_count}, Cumulative tokens so far: {cumulative_tokens}")
    print(f"Cumulative cost so far: ${cumulative_cost:.6f}")


def run_analysis(filepath: str):
    df = load_comments(filepath)
    df["Sentiment"] = ""
    df["Category"] = ""
    df["Relevance"] = ""

    # Process only the first 20 comments
    for count, (index, row) in enumerate(df[df["Comment"].astype(str).str.strip().astype(bool)].iterrows()):
        comment = row.get("Comment", None)

        # Check if the comment is a valid string
        if isinstance(comment, str) and comment.strip():
            process_comment(index, comment, df)

        # Stop after processing 20 comments
        if count >= 500:
            break

    save_results(df, "comments_with_analysis.xlsx")
    print(f"Final cumulative tokens consumed: {cumulative_tokens}")
    print(f"Final cumulative cost: ${cumulative_cost:.6f}")


if __name__ == "__main__":
    filepath = '/content/Combined Social Comments Indexed.xlsx'
    run_analysis(filepath)
