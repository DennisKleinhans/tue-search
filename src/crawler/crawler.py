import os
import pandas as pd
from enum import Enum
from typing import Any
from typing import cast
from typing import Tuple
from urllib.parse import urljoin
from urllib.parse import urlparse

import requests
import sqlitecloud
from bs4 import BeautifulSoup
from oauthlib.oauth2 import BackendApplicationClient
from playwright.sync_api import BrowserContext
from playwright.sync_api import Playwright
from playwright.sync_api import sync_playwright
from requests_oauthlib import OAuth2Session  # type:ignore
from langdetect import detect

from src.crawler.configs import INDEX_BATCH_SIZE
from src.crawler.configs import CRAWLER_OAUTH_CLIENT_ID
from src.crawler.configs import CRAWLER_OAUTH_CLIENT_SECRET
from src.crawler.configs import CRAWLER_OAUTH_TOKEN_URL
from src.crawler.initial_websites import INITIAL_WEBSITES
from src.crawler.html_utils import web_html_cleanup, check_if_tuebingen_in_text
from src.crawler.sql_utils import create_table_if_not_exists, insert_document

import logging

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class CRAWLER_VALID_SETTINGS(str, Enum):
    # Given a base site, index everything under that path
    RECURSIVE = "recursive"
    # Given a URL, index only the given page
    SINGLE = "single"
    # Given a sitemap.xml URL, parse all the pages in it
    SITEMAP = "sitemap"
    # Given a file upload where every line is a URL, parse all the URLs provided
    UPLOAD = "upload"


def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_internal_links(
    base_url: str, url: str, soup: BeautifulSoup, should_ignore_pound: bool = True
) -> set[str]:
    internal_links = set()
    for link in cast(list[dict[str, Any]], soup.find_all("a")):
        href = cast(str | None, link.get("href"))
        if not href:
            continue

        if should_ignore_pound and "#" in href:
            href = href.split("#")[0]

        if not is_valid_url(href):
            # Relative path handling
            href = urljoin(url, href)

        if urlparse(href).netloc == urlparse(url).netloc and base_url in href:
            internal_links.add(href)
    return internal_links


def start_playwright() -> Tuple[Playwright, BrowserContext]:
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)

    context = browser.new_context()

    if (
        CRAWLER_OAUTH_CLIENT_ID
        and CRAWLER_OAUTH_CLIENT_SECRET
        and CRAWLER_OAUTH_TOKEN_URL
    ):
        client = BackendApplicationClient(client_id=CRAWLER_OAUTH_CLIENT_ID)
        oauth = OAuth2Session(client=client)
        token = oauth.fetch_token(
            token_url=CRAWLER_OAUTH_TOKEN_URL,
            client_id=CRAWLER_OAUTH_CLIENT_ID,
            client_secret=CRAWLER_OAUTH_CLIENT_SECRET,
        )
        context.set_extra_http_headers(
            {"Authorization": "Bearer {}".format(token["access_token"])}
        )

    return playwright, context


def extract_urls_from_sitemap(sitemap_url: str) -> list[str]:
    response = requests.get(sitemap_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    urls = [loc_tag.text for loc_tag in soup.find_all("loc")]

    return urls


def _ensure_valid_url(url: str) -> str:
    if "://" not in url:
        return "https://" + url
    return url


def _read_urls_file(location: str) -> list[str]:
    with open(location, "r") as f:
        urls = [_ensure_valid_url(line.strip()) for line in f if line.strip()]
    return urls


class Crawler():
    def __init__(
        self,
        base_url: str,  # Can't change this without disrupting existing users
        CRAWLER_type: str = CRAWLER_VALID_SETTINGS.RECURSIVE.value,
        mintlify_cleanup: bool = True,  # Mostly ok to apply to other websites as well
        batch_size: int = INDEX_BATCH_SIZE,
        languages: list = None,
    ) -> None:
        self.mintlify_cleanup = mintlify_cleanup
        self.batch_size = batch_size
        self.recursive = False
        self.languages = languages

        if CRAWLER_type == CRAWLER_VALID_SETTINGS.RECURSIVE.value:
            self.recursive = True
            self.to_visit_list = [_ensure_valid_url(base_url)]
            return

        elif CRAWLER_type == CRAWLER_VALID_SETTINGS.SINGLE.value:
            self.to_visit_list = [_ensure_valid_url(base_url)]

        elif CRAWLER_type == CRAWLER_VALID_SETTINGS.SITEMAP:
            self.to_visit_list = extract_urls_from_sitemap(_ensure_valid_url(base_url))

        elif CRAWLER_type == CRAWLER_VALID_SETTINGS.UPLOAD:
            self.to_visit_list = _read_urls_file(base_url)

        else:
            raise ValueError(
                "Invalid Crawler Config, must choose a valid type between: " ""
            )
            
    @staticmethod
    def parse_metadata(metadata: dict[str, Any]) -> list[str]:
        custom_parser_req_msg = (
            "Specific metadata parsing required, crawler has not implemented it."
        )
        metadata_lines = []
        for metadata_key, metadata_value in metadata.items():
            if isinstance(metadata_value, str):
                metadata_lines.append(f"{metadata_key}: {metadata_value}")
            elif isinstance(metadata_value, list):
                if not all([isinstance(val, str) for val in metadata_value]):
                    raise RuntimeError(custom_parser_req_msg)
                metadata_lines.append(f'{metadata_key}: {", ".join(metadata_value)}')
            else:
                raise RuntimeError(custom_parser_req_msg)
        return metadata_lines
    
    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        if credentials:
            logger.warning("Unexpected credentials provided for crawler")
        return None

    def check_language(self, text: str, languages: list) -> bool:
        detected_language = detect(text)
        logger.info(f"Detected language: {detected_language}")
        return detected_language in languages
        
    def compare_urls(self, url1, url2):
        """Compares two URLs, ignoring query parameters."""
        # Parse the URLs
        parsed_url1 = urlparse(url1)
        parsed_url2 = urlparse(url2)

        # Compare core components
        if (parsed_url1.scheme == parsed_url2.scheme and
            parsed_url1.netloc == parsed_url2.netloc and
            parsed_url1.path == parsed_url2.path):
            return True
        else:
            return False

    def check_duplicate_urls(self, current_url: str, visited_links: list[str]) -> bool:
        for visited_url in visited_links:
            if self.compare_urls(current_url, visited_url):
                return True
        return False
    
    def load_from_state(self):
        """Traverses through all pages found on the website
        and converts them into documents"""
        visited_links: set[str] = set()
        to_visit: list[str] = self.to_visit_list
        base_url = to_visit[0]  # For the recursive case
        doc_batch: list[tuple] = []

        playwright, context = start_playwright()
        restart_playwright = False
        while to_visit:
            current_url = to_visit.pop()
            if self.check_duplicate_urls(current_url, visited_links):
                continue
            visited_links.add(current_url)

            logger.info(f"Visiting {current_url}")

            try:
                if restart_playwright:
                    playwright, context = start_playwright()
                    restart_playwright = False

                page = context.new_page()
                page_response = page.goto(current_url)
                final_page = page.url
                if final_page != current_url:
                    logger.info(f"Redirected to {final_page}")
                    current_url = final_page
                    if current_url in visited_links:
                        logger.info("Redirected page already indexed")
                        continue
                    visited_links.add(current_url)

                content = page.content()
                soup = BeautifulSoup(content, "html.parser")

                if self.recursive:
                    internal_links = get_internal_links(base_url, current_url, soup)
                    for link in internal_links:
                        if link not in visited_links:
                            to_visit.append(link)

                if page_response and str(page_response.status)[0] in ("4", "5"):
                    logger.info(
                        f"Skipped indexing {current_url} due to HTTP {page_response.status} response"
                    )
                    continue

                parsed_html = web_html_cleanup(soup, self.mintlify_cleanup)
 
                # Check if the document contains tübingen
                if not check_if_tuebingen_in_text(parsed_html.cleaned_text):
                    logger.info(f"Skipped indexing {current_url} due to missing 'Tübingen'")
                    continue
                
                # Check if the document is in the correct language
                if self.languages and not self.check_language(parsed_html.cleaned_text, self.languages):
                    logger.info(f"Skipped indexing {current_url} due to language mismatch")
                    continue


                doc_batch.append((
                        current_url,
                        parsed_html.cleaned_text,
                        parsed_html.title
                    )
                )

                page.close()
            except Exception as e:
                logger.error(f"Failed to fetch '{current_url}': {e}")
                playwright.stop()
                restart_playwright = True
                continue

            if len(doc_batch) >= self.batch_size:
                playwright.stop()
                restart_playwright = True
                yield doc_batch
                doc_batch = []
        
        playwright.stop()
        
        if doc_batch:
            yield doc_batch


if __name__ == "__main__":
    
    # Open the connection to SQLite Cloud
    conn = sqlitecloud.connect("sqlitecloud://cyd2d2juiz.sqlite.cloud:8860?apikey=xZXTNaxWuKM6ryHCVELzSVnT3KC3AubraCDuwFyxKJ4")
    db_name = "documents"
    conn.execute(f"USE DATABASE {db_name}")
    cursor = conn.cursor()
    
    # Create table if not exists
    create_table_if_not_exists(cursor)
    
    counter = 0
    try: 
            
        for website in INITIAL_WEBSITES:
            connector = Crawler(website, CRAWLER_VALID_SETTINGS.RECURSIVE.value, languages=["en"])
            document_batches = connector.load_from_state()
            for batch in document_batches:
                if batch:
                    for url, doc, title in batch:
                        logger.info(f"Title: {title} - URL: {url}")
                        success = insert_document(cursor, url, title, doc)
                        if success: counter += 1 
                        conn.commit()   
    except Exception as e:
        logger.error(f"Failed to fetch '{website}': {e}")
    finally:
        conn.close()
    
    logger.info(f"Crawled {counter} new documents.")

                
    
