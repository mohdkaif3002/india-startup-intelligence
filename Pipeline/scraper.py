import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def scrape_inc42_headlines():
    """
    Scrapes startup funding headlines from Inc42.
    Returns a list of article titles and links.
    """
    url = "https://inc42.com/buzz/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    print("Connecting to Inc42...")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status code: {response.status_code}")

        if response.status_code != 200:
            print("Could not reach Inc42. Skipping.")
            return []

        soup = BeautifulSoup(response.text, "lxml")

        articles = []
        for tag in soup.find_all("h2"):
            title = tag.get_text(strip=True)
            link_tag = tag.find("a")
            link = link_tag["href"] if link_tag else "No link"
            if title:
                articles.append({"title": title, "link": link})

        print(f"Found {len(articles)} headlines")
        return articles

    except Exception as e:
        print(f"Error: {e}")
        return []


def save_headlines(articles):
    """Saves scraped headlines to a CSV file."""
    if not articles:
        print("No articles to save.")
        return

    os.makedirs("Data/raw", exist_ok=True)
    df = pd.DataFrame(articles)
    df.to_csv("Data/raw/headlines.csv", index=False)
    print(f"Saved {len(df)} headlines to Data/raw/headlines.csv")
    print(df.head())


if __name__ == "__main__":
    articles = scrape_inc42_headlines()
    save_headlines(articles)