from file_io import load_comments, save_results
from comment_processor import process_comment


def run_analysis(filepath: str):
    """Run analysis on comments from an Excel file."""
    df = load_comments(filepath)
    df["Sentiment"] = ""
    df["Category"] = ""
    df["Relevance"] = ""

    for count, (index, row) in enumerate(df[df["Comment"].astype(str).str.strip().astype(bool)].iterrows()):
        comment = row.get("Comment", None)
        if isinstance(comment, str) and comment.strip():
            process_comment(index, comment, df)
        if count >= 150:
            break

    save_results(df, "comments_with_analysis.xlsx")


if __name__ == "__main__":
    filepath = '...'  # Path to the input Excel file containing comments.
    run_analysis(filepath)
