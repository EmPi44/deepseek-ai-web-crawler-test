import json
import os
from typing import List, Set, Tuple

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
)

from models.venue import Venue
from utils.data_utils import is_complete_venue, is_duplicate_venue
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def get_browser_config() -> BrowserConfig:
    """
    Returns the browser configuration for the crawler.

    Returns:
        BrowserConfig: The configuration settings for the browser.
    """
    # https://docs.crawl4ai.com/core/browser-crawler-config/
    return BrowserConfig(
        browser_type="chromium",  # Type of browser to simulate
        headless=False,  # Whether to run in headless mode (no GUI)
        verbose=True,  # Enable verbose logging
    )


def get_llm_strategy() -> LLMExtractionStrategy:
    """
    Returns the configuration for the language model extraction strategy.

    Returns:
        LLMExtractionStrategy: The settings for how to extract data using LLM.
    """
    # https://docs.crawl4ai.com/api/strategies/#llmextractionstrategy
    return LLMExtractionStrategy(
        provider="groq/deepseek-r1-distill-llama-70b",  # Name of the LLM provider
        api_token=os.getenv("GROQ_API_KEY"),  # API token for authentication
        schema=Venue.model_json_schema(),  # JSON schema of the data model
        extraction_type="schema",  # Type of extraction to perform
        instruction=(
            """
            Extract all venue objects with 'document_name' and 'document_url' of the ISO 9001 documents from the 
            following content. 
            Only retrieve the links relevant to germany location and take care that you take the PDF links which are working for download. 
            Means check on the content what the base-url is and add the relative path to the base-url.
            """
        ),  # Instructions for the LLM
        input_format="markdown",  # Format of the input content
        verbose=True,  # Enable verbose logging
    )


async def check_no_results(
    crawler: AsyncWebCrawler,
    url: str,
    session_id: str,
) -> bool:
    """
    Checks if the "No Results Found" message is present on the page.

    Args:
        crawler (AsyncWebCrawler): The web crawler instance.
        url (str): The URL to check.
        session_id (str): The session identifier.

    Returns:
        bool: True if "No Results Found" message is found, False otherwise.
    """
    # Fetch the page without any CSS selector or extraction strategy
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            session_id=session_id,
        ),
    )

    if result.success:
        if "No Results Found" in result.cleaned_html:
            return True
    else:
        print(
            f"Error fetching page for 'No Results Found' check: {result.error_message}"
        )

    return False


async def fetch_and_process_page(
    crawler: AsyncWebCrawler,
    page_url: str,
    css_selector: str,
    llm_strategy: LLMExtractionStrategy,
    session_id: str,
    required_keys: List[str],
    seen_names: Set[str],
) -> Tuple[List[dict], bool]:
    """
    Fetches and processes venue data from a single URL.
    """
    print(f"Loading URL: {page_url}")

    # First, fetch the page to get base URL
    initial_result = await crawler.arun(
        url=page_url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            session_id=session_id,
        ),
    )

    if not initial_result.success:
        print(f"Error fetching initial page: {initial_result.error_message}")
        return [], False

    # Parse the HTML to find the base URL
    soup = BeautifulSoup(initial_result.cleaned_html, 'html.parser')
    base_tag = soup.find('base')
    
    # Get base URL from base tag or page URL
    if base_tag and base_tag.get('href'):
        base_url = base_tag['href']
    else:
        parsed_url = urlparse(page_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

    print(f"Detected base URL: {base_url}")

    # Update LLM strategy with URL handling instructions
    llm_strategy.instruction = f"""
        Extract all venue objects with 'document_name' and 'document_url' of the ISO 9001 documents from the 
        following content. Only retrieve the links relevant to germany location and take care that you take 
        the PDF links which are working for download.

        IMPORTANT URL HANDLING INSTRUCTIONS:
        1. For any PDF link you find, check if it's relative or absolute
        2. If the link is relative (starts with '/' or doesn't start with 'http'):
           - For links starting with '/': combine '{base_url}' with the link
           - For links without leading '/': combine '{base_url}/' with the link
        3. If the link is absolute (starts with 'http'): use it as is
        4. Make sure all document_url values are complete, valid URLs

        Example URL conversions:
        - '/docs/cert.pdf' → '{base_url}/docs/cert.pdf'
        - 'docs/cert.pdf' → '{base_url}/docs/cert.pdf'
        - 'https://example.com/cert.pdf' → unchanged

        Return the data in the following format:
        {{
            "document_name": "Name of the document",
            "document_url": "Complete URL to the PDF"
        }}
    """

    # Use the updated LLM strategy to extract content
    result = await crawler.arun(
        url=page_url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=llm_strategy,
            css_selector=css_selector,
            session_id=session_id,
        ),
    )

    if not (result.success and result.extracted_content):
        print(f"Error fetching URL: {result.error_message}")
        return [], False

    # Parse extracted content
    extracted_data = json.loads(result.extracted_content)
    if not extracted_data:
        print("No venues found.")
        return [], False

    # After parsing extracted content
    print("Extracted data:", extracted_data)

    # Process venues and ensure URLs are properly formed
    complete_venues = []
    for venue in extracted_data:
        # Ensure URL is properly formed
        if 'document_url' in venue:
            venue['document_url'] = urljoin(base_url, venue['document_url'])

        if not is_complete_venue(venue, required_keys):
            continue

        if is_duplicate_venue(venue["document_name"], seen_names):
            print(f"Duplicate venue '{venue['document_name']}' found. Skipping.")
            continue

        seen_names.add(venue["document_name"])
        complete_venues.append(venue)

    print(f"Extracted {len(complete_venues)} venues.")
    return complete_venues, False
