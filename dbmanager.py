import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import duckdb
import subprocess
import platform
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from filecrawler import FileCrawler
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz

class DbManager:
    def __init__(self):
        self.mode = None
        try:
            base_path = Path(__file__).parent.absolute()
            env_path = base_path / ".env"
            db_path = base_path / "semSearch.duckdb"

            load_dotenv(dotenv_path=env_path)
            self.api_key = os.getenv("OPENAI_API_KEY")

            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                self.openai_available = True
            else:
                self.client = None
                self.openai_available = False
                print(f"Hinweis: Kein API-Key in {env_path} gefunden.")

            # 2. Datenbank-Verbindung mit absolutem Pfad
            self.connection = duckdb.connect(str(db_path))
            self.connection.execute("INSTALL vss;")
            self.connection.execute("LOAD vss;")

            # Tabellen-Setup
            self.connection.execute("CREATE SEQUENCE IF NOT EXISTS doc_id_seq;")
            self.connection.execute("CREATE SEQUENCE IF NOT EXISTS chunk_id_seq;")
            self.connection.execute("CREATE TABLE IF NOT EXISTS modetable(mode INTEGER);")

            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY DEFAULT NEXTVAL('doc_id_seq'),
                    file_path TEXT UNIQUE,
                    title TEXT,
                    last_mod DOUBLE
                );
            """)

            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY DEFAULT NEXTVAL('chunk_id_seq'),
                    doc_id INTEGER, 
                    chunk_text TEXT,
                    embedding_local FLOAT[384],
                    embedding_openai FLOAT[1536],
                    FOREIGN KEY(doc_id) REFERENCES documents(id)
                );
            """)

            # NLP Tools
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            self.mode = self.get_mode()
            if self.mode == 0:
                self.mode = 1
                self.connection.execute("INSERT INTO modetable (mode) VALUES (?);", [self.mode])


        except Exception as e:
            print(f"Kritischer Fehler bei der Initialisierung: {e}")
            if self.connection:
                self.connection.close()
            raise e

    def __del__(self):
        try:
            if self.mode == 2:
                self.connection.execute("DELETE FROM chunks;")
                self.connection.execute("DELETE FROM documents;")
            self.connection.close()
        except Exception as e:
            print(f"error {e}")

    def set_session_mode(self, use_openai):
        self.allow_openai_this_run = use_openai
        if not use_openai:
            self.allow_openai_this_run = False
        print(f"Modus gesetzt: {'OpenAI + Lokal' if use_openai else 'Nur Lokal'}")

    def create_chunks_and_vectorize(self, content):
        chunks = self.text_splitter.split_text(content)
        if not chunks:
            return

        vectors_local = self.model.encode(chunks).tolist()

        vectors_openai = [None] * len(chunks)

        if self.allow_openai_this_run and self.openai_available:
             try:
                response = self.client.embeddings.create(
                    input=chunks,
                    model="text-embedding-3-small"
                )
                vectors_openai = [item.embedding for item in response.data]
             except Exception as e:
                print(f"OpenAI Embedding Fehler: {e}")

        for chunk_text, v_local, v_openai in zip(chunks, vectors_local, vectors_openai):
                yield chunk_text, v_local, v_openai

    def _extract_text(self, file_path):
        suffix = Path(file_path).suffix.lower()

        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        elif suffix == '.pdf':
            text = ""
            try:
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text += page.get_text()
                return text
            except Exception as e:
                print(f"PDF-Fehler bei {file_path}: {e}")
                return ""

        return ""


    def process_document_content(self, file_path, doc_id):
        try:
            content = self._extract_text(file_path)

            if not content.strip():
                return

            data_to_insert = []
            for chunk_text, v_local, v_openai in self.create_chunks_and_vectorize(content):
                data_to_insert.append((doc_id, chunk_text, v_local, v_openai))

            self.connection.executemany(
                "INSERT INTO chunks (doc_id, chunk_text, embedding_local, embedding_openai) VALUES (?, ?, ?, ?)",
                data_to_insert
            )

            self.connection.commit()
        except Exception as e:
            print(f"Fehler beim Verarbeiten: {e}")

    def work_directory(self, directory):
        crawler = FileCrawler()
        for file_path, mtime in crawler.crawl(directory):
            title = Path(file_path).name
            try:
                self.connection.execute("INSERT INTO documents (file_path, title, last_mod) VALUES (?, ?, ?)",
                                        (file_path, title, mtime))
                doc_id = self.connection.execute("SELECT currval('doc_id_seq')").fetchone()[0]
                print(f"processing file {file_path}")
                self.process_document_content(file_path, doc_id)
            except duckdb.ConstraintException:
                # Datei existiert schon
                row = self.connection.execute("SELECT id, last_mod FROM documents WHERE file_path = ?",
                                              [file_path]).fetchone()
                doc_id, saved_time = row[0], row[1]

                needs_enrichment = False
                if self.allow_openai_this_run:
                    missing = self.connection.execute(
                        "SELECT count(*) FROM chunks WHERE doc_id = ? AND embedding_openai IS NULL",
                        [doc_id]
                    ).fetchone()[0]
                    needs_enrichment = missing > 0

                if mtime > saved_time or needs_enrichment:
                    self.connection.execute("DELETE FROM chunks WHERE doc_id = ?", [doc_id])
                    self.connection.execute("UPDATE documents SET last_mod = ? WHERE file_path = ?", [mtime, file_path])
                    print(f"processing file {file_path}")
                    self.process_document_content(file_path, doc_id)

    def search(self, question, use_openai=False):
        if use_openai:
            try:
                resp = self.client.embeddings.create(input=[question], model="text-embedding-3-small")
                vector = resp.data[0].embedding
                col = "embedding_openai"
                dim = 1536
                order_by = "similarity DESC"
            except Exception as e:
                print(f"OpenAI Suche fehlgeschlagen, nutze lokal: {e}")
                use_openai = False

        if not use_openai:
            vector = self.model.encode(question).tolist()
            col = "embedding_local"
            dim = 384
            order_by = "similarity DESC"


        result = self.connection.execute(f"""
            SELECT 
                file_path, 
                array_cosine_similarity(chunks.{col}, $1::FLOAT[{dim}]) as similarity, 
                chunk_text 
            FROM chunks 
            JOIN documents on documents.id = chunks.doc_id 
            WHERE chunks.{col} IS NOT NULL
            ORDER BY {order_by}
            LIMIT $2;
        """, (vector, 3)).fetchall()


        processed_results = []
        for res in result:
            file_path, similarity, chunk_text = res
            dist = 1.0 - similarity
            processed_results.append((file_path, dist, chunk_text))

        return processed_results

    def open_file(self, file_path):
        if platform.system() == "Darwin":
            subprocess.run(["open", file_path])
        elif platform.system() == "Windows":
            os.startfile(file_path)
        else:
            subprocess.run(["xdg-open", file_path])

    def get_mode(self):
        mode = self.connection.execute("SELECT mode FROM modetable").fetchone()
        if mode is None:
            return 0
        else:
            return mode[0]
