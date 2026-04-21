import pathlib

class FileCrawler:
    def __init__(self):
        self.extension = ['.pdf','.txt']

    def crawl(self, file_path):
        root = pathlib.Path(file_path)

        if not root.exists():
            return

        for file_path in root.rglob('*'):
            try:
                if file_path.is_file() and file_path.suffix.lower() in self.extension:
                    yield str(file_path.absolute()),file_path.stat().st_mtime
            except PermissionError:
                print(f"couldn't read file: {file_path}")
                continue
