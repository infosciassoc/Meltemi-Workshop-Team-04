import wikipediaapi

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='Meltemi project (mysiloglou@gmail.com)',
        language='el',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)

def fetch_wikipedia_content(page_title):
    """Fetch content from Wikipedia."""
    page_title = page_title.replace(' ', '_')
    page = wiki_wiki.page(page_title)
    if page.exists():
        return page.text
    else:
        return None