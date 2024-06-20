import logging

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

def check_table_exists(cursor, table_name):
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    return cursor.fetchone() is not None

# Function to create the table if it doesn't exist
def create_table_if_not_exists(cursor):
    if not check_table_exists(cursor, 'documents'):
        cursor.execute('''
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                document TEXT NOT NULL,
                UNIQUE(url, title)
            );
        ''')
        logger.info("Table created.")
    else:
        logger.info("Table already exists.")

# Function to insert a document
def insert_document(cursor, url, title, document):
    try:
        cursor.execute('''
            INSERT INTO documents (url, title, document)
            VALUES (?, ?, ?);
        ''', (url, title, document))
        logger.info("Document inserted.")
        return True
    except Exception as e:
        logger.error(f"Error inserting document: {e}")
        return False


