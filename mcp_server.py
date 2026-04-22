import os
from pathlib import Path
from fastmcp import FastMCP
from dbmanager import DbManager

mcp = FastMCP("PDF-Knowledge-Base")

db = DbManager()

db.set_session_mode(use_openai=True)


@mcp.tool()
def ask_my_documents(query: str, directory: str = "/Users/simxn/Documents/semanticsearch") -> str:
    """
    Sucht in lokalen Dokumenten(pdf und textdateien)nach Antworten auf eine Frage.
    Dieser Prozess scannt automatisch das Verzeichnis nach neuen Dateien,
    indiziert sie bei Bedarf und führt dann eine semantische Suche aus.

    Args:
        query: Die Frage oder der Suchbegriff des Nutzers.
        directory: Der Pfad zum Ordner mit den PDFs/Textdateien.
    """
    try:
        db.work_directory(directory)
    except Exception as e:
        return f"Fehler beim Scannen des Verzeichnisses: {str(e)}"


    results = db.search(query, use_openai=db.allow_openai_this_run)

    if not results:
        return "Ich habe das Verzeichnis gescannt, konnte aber keine relevanten Informationen zu deiner Anfrage finden."

    response = "Ich habe die Dokumente durchsucht. Hier sind die relevantesten Ausschnitte:\n\n"

    for file_path, dist, text in results:
        file_name = Path(file_path).name
        # Umrechnung der Distanz in ein Ähnlichkeits-Rating
        similarity = round((1 - dist) * 100, 1)

        response += f"--- QUELLE: {file_name} (Relevanz: {similarity}%) ---\n"
        response += f"{text}\n\n"

    response += "\nBitte nutze diese Informationen, um die Frage des Nutzers zu beantworten."
    return response


@mcp.tool()
def open_source_file(file_path: str) -> str:
    """
    Öffnet eine spezifische Datei im Standard-Viewer des Betriebssystems.
    Hilfreich, wenn der Nutzer das Originaldokument lesen möchte.
    """
    try:
        db.open_file(file_path)  # Nutzt deine plattformübergreifende Logik
        return f"Die Datei {file_path} wurde erfolgreich geöffnet."
    except Exception as e:
        return f"Fehler beim Öffnen der Datei: {str(e)}"


if __name__ == "__main__":
    mcp.run()