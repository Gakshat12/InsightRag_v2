"""Fetch ~300 arXiv CS abstracts across diverse fields into data/kb/ with polite rate-limiting."""
import os
import re
import time
import argparse
import arxiv

# Target Computer Science topics
CATEGORIES = [
    ("NLP", "cat:cs.CL", 60),
    ("Vision", "cat:cs.CV", 60),
    ("Systems", "cat:cs.DC OR cat:cs.OS", 60),
    ("Databases", "cat:cs.DB", 60),
    ("Security", "cat:cs.CR", 60),
]


def sanitize_filename(name: str) -> str:
    """Sanitize string for safe cross-platform file names."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    return clean.replace(" ", "_")[:60]


def fetch_arxiv_papers(output_dir: str = "data/kb", papers_per_cat: int = 60) -> int:
    """Fetch abstracts from arXiv API and save as individual .txt files."""
    os.makedirs(output_dir, exist_ok=True)

    # Polite client with 3-second delay between pages as recommended by arXiv
    client = arxiv.Client(
        page_size=papers_per_cat,
        delay_seconds=3.0,
        num_retries=3
    )

    total_downloaded = 0

    for topic_name, query, count in CATEGORIES:
        print(f"[*] Fetching up to {count} papers for topic: {topic_name} ('{query}')...")
        search = arxiv.Search(
            query=query,
            max_results=count,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        topic_saved = 0
        try:
            results = client.results(search)
            for paper in results:
                paper_id = paper.get_short_id()
                clean_title = sanitize_filename(paper.title)
                filename = f"arxiv_{paper_id}_{clean_title}.txt"
                filepath = os.path.join(output_dir, filename)

                content = (
                    f"Title: {paper.title}\n"
                    f"Authors: {', '.join([a.name for a in paper.authors])}\n"
                    f"Categories: {', '.join(paper.categories)}\n"
                    f"Published: {paper.published.strftime('%Y-%m-%d') if paper.published else 'N/A'}\n"
                    f"ArXiv ID: {paper_id}\n"
                    f"URL: {paper.entry_id}\n\n"
                    f"Abstract:\n{paper.summary.strip()}\n"
                )

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

                topic_saved += 1
                total_downloaded += 1

            print(f"    [+] Saved {topic_saved} papers for {topic_name}.")
        except Exception as e:
            print(f"    [-] Error fetching {topic_name}: {e}")

        # Polite interval between category queries
        time.sleep(2.0)

    print(f"\n[✓] Completed. Total papers saved in '{output_dir}': {total_downloaded}")
    return total_downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ~300 public arXiv CS abstracts into data/kb/")
    parser.add_argument("--output-dir", type=str, default="data/kb", help="Directory to save text files")
    parser.add_argument("--per-category", type=int, default=60, help="Papers per topic category")
    args = parser.parse_args()

    fetch_arxiv_papers(output_dir=args.output_dir, papers_per_cat=args.per_category)

