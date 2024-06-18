import os

INDEX_BATCH_SIZE = 16
    
CRAWLER_IGNORED_CLASSES = ["sidebar", "footer"]
CRAWLER_IGNORED_ELEMENTS = ["nav", "footer", "meta", "script", "style", "symbol", "aside"]

CRAWLER_OAUTH_CLIENT_ID = os.environ.get("CRAWLER_OAUTH_CLIENT_ID")
CRAWLER_OAUTH_CLIENT_SECRET = os.environ.get("CRAWLER_OAUTH_CLIENT_SECRET")
CRAWLER_OAUTH_TOKEN_URL = os.environ.get("CRAWLER_OAUTH_TOKEN_URL")

