from comment_analysis import analyze_comment


def process_comment(index: int, comment: str, df):
    """Process an individual comment, analyzing its sentiment, category, and relevance."""
    sentiment, category, relevance = analyze_comment(comment)
    df.at[index, "Sentiment"] = sentiment
    df.at[index, "Category"] = category
    df.at[index, "Relevance"] = relevance

    print(f"\nProcessed comment at index {index}:\nComment: {comment}")
    print(f"Sentiment: {sentiment}\nCategory: {category}\nRelevance: {relevance}")
