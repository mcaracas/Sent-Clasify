from api_client import make_request_with_backoff, client, MODEL


def parse_response(response_text: str) -> tuple:
    """Parse API response into sentiment, category, and relevance."""
    try:
        sentiment, category, relevance = response_text.strip().split(" | ")
        return sentiment, category, relevance
    except ValueError:
        return "Error", "Error", "Error"


def analyze_comment(comment: str) -> tuple:
    """Analyze a comment using OpenAI API for sentiment, category, and relevance."""
    messages = [
        {"role": "system", "content": "You are an assistant for analyzing customer feedback."},
        {"role": "user", "content": f"""
        Analyze the following comment for three aspects: sentiment, category, and relevance.

        Comment: "{comment}"

        1. Label sentiment as Positive, Negative, or Neutral.
        2. Identify the main category from this list: Reliability, Innovation, Excitement, Convenience, Efficiency, Aesthetics, Price, Miscellaneous.
        3. Rate the relevance to Samsung brand as High, Medium, or Low.

        Respond ONLY in the format: Sentiment | Category | Relevance
        """}
    ]
    try:
        completion = make_request_with_backoff(
            client.chat.completions.create,
            model=MODEL,
            messages=messages,
            max_tokens=50,
            temperature=0
        )
        print(f"Raw OpenAI response:\n{completion}\n")
        return parse_response(completion.choices[0].message.content.strip())
    except Exception as e:
        print(f"OpenAI API error during combined analysis: {str(e)}")
        return "Error", "Error", "Error"
