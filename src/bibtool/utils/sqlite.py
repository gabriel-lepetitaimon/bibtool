import sqlite3
from pathlib import Path


class SQLite:
    def __init__(self, file: Path, readonly=False):
        self.file = file
        self.conn = None
        self.readonly = readonly

    def check_exists(self):
        try:
            self.connect()
            self.close()
            return True
        except sqlite3.OperationalError:
            return False

    def cursor(self):
        return SQLiteCursor(self)

    def connect(self):
        if self.conn is not None:
            raise RuntimeError("Already connected")
        if self.readonly:
            self.conn = sqlite3.connect(f"file:{self.file}?immutable=1", uri=True)
        else:
            self.conn = sqlite3.connect(str(self.file))
        return self.conn

    def close(self):
        if self.conn is None:
            raise RuntimeError("Not connected")
        if not self.readonly:
            self.conn.commit()
        self.conn.close()
        self.conn = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, type, value, traceback):
        self.close()


class SQLiteCursor:
    def __init__(self, db: SQLite):
        self.db = db

    def __enter__(self):
        return self.db.connect().cursor()

    def __exit__(self, type, value, traceback):
        self.db.close()
