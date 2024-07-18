import csv
import logging
import json

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


def check_table_exists(cursor, table_name):
    cursor.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
    )
    return cursor.fetchone() is not None


# Function to create the table if it doesn't exist
def create_table_if_not_exists(cursor):
    if not check_table_exists(cursor, "documents"):
        cursor.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                document TEXT NOT NULL,
                UNIQUE(url, title)
            );
        """
        )
        logger.info("Table created.")
    else:
        logger.info("Table already exists.")


# Function to insert a document
def insert_document(cursor, url, title, document):
    try:
        cursor.execute(
            """
            INSERT INTO documents (url, title, document)
            VALUES (?, ?, ?);
        """,
            (url, title, document),
        )
        logger.info("Document inserted.")
        return True
    except Exception as e:
        logger.error(f"Error inserting document: {e}")
        return False


def write_to_csv(documents: list[tuple[str, str, str]], output_file: str) -> None:
    """Schreibt Dokumente in eine CSV-Datei"""
    with open(output_file + ".csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for url, content, title in documents:
            writer.writerow([url, title, content])


def write_to_json(documents: list[tuple], output_file: str):
    with open(output_file + ".json", "a", encoding="utf-8") as f:
        for url, content, title in documents:
            json.dump(
                {"URL": url, "Title": title, "Content": content}, f, ensure_ascii=False
            )
            f.write("\n")


def get_all_documents(cursor):
    try:
        cursor.execute("SELECT * FROM documents;")
        documents = cursor.fetchall()
        logger.info("Documents retrieved.")
        return documents
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []


def get_first_10_documents(cursor):
    try:
        cursor.execute("SELECT * FROM documents LIMIT 10;")
        documents = cursor.fetchall()
        logger.info("First 10 documents retrieved.")
        return documents
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []
