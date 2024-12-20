import os
import random
import time
from pathlib import Path
from typing import Iterable, Literal, Optional

import pandas as pd
import xmltodict

from .utils import RichProgress
from .utils.sqlite import SQLite, sqlite3


class ZoteroLibrary:
    def __init__(
        self,
        library: Optional[str] = "",
        collection: Optional[str] = None,
        zotero_path: Optional[str | Path] = None,
    ):
        """Initialize a connection to a given Zotero library.

        Parameters
        ----------
        library : Optional[str], optional
            Name of the library to search for. If not provided, the default library is used.

        collection : Optional[str], optional
            Name of the collection to search for. If not provided, all items in the library are listed.

        zotero_path : Optional[str  |  Path], optional
            Path to the Zotero data directory. If not provided, the default path is used.
        """
        if zotero_path is None:
            zotero_path = Path.home() / "Zotero"
        elif isinstance(zotero_path, str):
            zotero_path = Path(zotero_path).absolute()
        self._zotero_path = zotero_path.absolute()
        self._database = SQLite(self.zotero_path / "zotero.sqlite", readonly=True)
        if not self._database.check_exists():
            raise FileNotFoundError(f"Zotero database not found at {self.zotero_db_path}")

        self.library = library
        self.collection = collection

    @property
    def zotero_path(self) -> Path:
        """Path to the Zotero data directory. By default: ~/Zotero."""
        return self._zotero_path

    @property
    def database(self) -> SQLite:
        """SQLite connection to the Zotero database."""
        return self._database

    @property
    def library(self) -> str:
        """Name of the analyzed library. If not provided, the default library is used."""
        return self._library

    @property
    def library_pretty(self) -> str:
        """Pretty name for the analyzed library.

        This property equals "default library" if the default library is used, and f'library "{self.library}"' otherwise.
        """  # noqa: E501
        return f'library "{self._library}"' if self._library_id != 1 else "default library"

    @library.setter
    def library(self, library: str):
        if library == "":
            library_id = 1
        else:
            with self._database.cursor() as c:
                library_ids = c.execute("SELECT libraryID FROM groups WHERE name = ?", (library,)).fetchone()
            if library_ids is None:
                raise ValueError(f"Library {library} not found")
            library_id = library_ids[0]

        self._library_id = library_id
        self._library = library

    @property
    def collection(self) -> str | None:
        """Name of the analyzed collection. If None, all items in the library are listed."""
        return self._collection

    @collection.setter
    def collection(self, collection: str | None):
        if collection is not None:
            with self._database.cursor() as c:
                collection_ids = c.execute(
                    "SELECT collectionID FROM collections WHERE libraryID = ? AND collectionName = ?",
                    (self._library_id, collection),
                ).fetchone()
            if collection_ids is None:
                raise ValueError(f'Collection "{collection}" not found in {self.library_pretty}')
            else:
                collection_id = collection_ids[0]
            self._collection_id = collection_id
        self._collection = collection

    # =========================================================================
    def download_papers(
        self,
        dois: Iterable[str],
        zotero_connector_hotkey: str = ("ctrl", "."),
        max_hotkey_retry: int = 2,
        initial_wait: float = 4,
        timeout=15,
    ):
        """Download the full text of the specified DOIs.

        ..warning::
            This function only works in linux and requires firefox and xdotool to be installed.
            It also requires the Zotero client to be open and the Zotero Connector to be installed in the browser.
            You won't be able to use your computer while the function is running.

        Parameters
        ----------
        dois : List[str]
            List of DOIs to download.

        """
        import subprocess

        import pyautogui

        try:
            size = len(dois)
        except TypeError:
            size = None

        # === Read the DOIs already present in the database ===
        ignore_dois = self.fetch_item_field("DOI").to_list()

        with RichProgress("Downloading papers... ", size) as progress:
            dois = list(dois)
            random.shuffle(dois)
            for doi in dois:
                if doi in ignore_dois:
                    progress.update(advance=1)
                    continue

                failed = False

                subprocess.call(["firefox", "https://doi.org/" + doi])
                # Initial wait time to let the page load
                time.sleep(initial_wait)

                retry = 0
                while retry <= max_hotkey_retry:
                    # Send hotkey to Zotero Connector
                    pyautogui.hotkey(*zotero_connector_hotkey)
                    retry += 1

                    # Wait for the DOI to appear in Zotero database
                    for _ in range(retry + 3):
                        time.sleep(1)
                        if self.has_doi(doi):
                            break
                    else:
                        # After 3+retry seconds, send the hotkey again
                        continue
                    break
                else:
                    # If the page can't load or Zotero can't register the item, skip to the next DOI
                    failed = True

                # Wait for the item to be downloaded
                t0 = time.time()
                while not failed and (time.time() - t0) < timeout:
                    if self.has_local(doi):
                        break
                    time.sleep(0.5)
                else:
                    # If the item can't be downloaded, skip to the next DOI
                    failed = True

                if failed:
                    progress.update(total=progress.total - 1)
                else:
                    progress.update(advance=1)

                # Send an hotkey to close the current tab
                pyautogui.hotkey("ctrl", "w")
                time.sleep(0.5)

    # =========================================================================
    def list_local_path(
        self,
        prefer: Optional[Literal["pdf", "html"]] = None,
        check_exists: bool = False,
    ) -> pd.DataFrame:
        """
        List the local path of the attachments of the items in the Zotero library.

        Parameters
        ----------
        prefer : {"pdf", "html"}, optional
            If specified, any document with both a PDF and an HTML version will be filtered to keep only the preferred type.

        check_exists : bool, optional
            If True, only the paths that exist are returned. Default is False.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the columns "title", "key", "content_type", and "path".
        """  # noqa: E501
        req = """
        SELECT title.value, DOI.value, items.key, itemAttachments.contentType, itemAttachments.path, attachedItem.key 
        FROM items
        JOIN itemData itemDataTitle ON items.itemID = itemDataTitle.itemID
        JOIN itemDataValues title ON itemDataTitle.valueID = title.valueID
        JOIN itemData itemDataDOI ON items.itemID = itemDataDOI.itemID
        JOIN itemDataValues DOI ON itemDataDOI.valueID = DOI.valueID
        JOIN itemAttachments ON items.itemID = itemAttachments.parentItemID
        JOIN items attachedItem ON itemAttachments.itemID = attachedItem.itemID
        """
        where_req = "WHERE items.libraryID = ? AND itemDataTitle.fieldID = 1 AND itemDataDOI.fieldID = 59"
        req_args = [self._library_id]
        if self._collection is not None:
            req += """
            JOIN collectionItems ON items.itemID = collectionItems.itemID
            """
            where_req += " AND collectionItems.collectionID = ?"
            req_args.append(self._collection_id)
        with self._database.cursor() as c:
            c.execute(req + where_req, req_args)
            df = pd.DataFrame(c.fetchall(), columns=["title", "doi", "zoteroID", "content_type", "path", "storageID"])

        df["path"] = [
            f"{self.zotero_path}{os.sep}storage{os.sep}{key}{os.sep}{path[8:]}" if path is not None else None
            for key, path in df[["storageID", "path"]].values
        ]

        if check_exists:
            df = df.loc[~df["path"].isna()]
            df = df.loc[df["path"].apply(os.path.exists)]

        if prefer is not None:
            duplicates = df.duplicated(subset=["zoteroID"], keep=False)
            if prefer == "pdf":
                df.drop(df.loc[duplicates & (df["content_type"] == "text/html")].index, inplace=True)
            elif prefer == "html":
                df.drop(df.loc[duplicates & (df["content_type"] == "application/pdf")].index, inplace=True)

        df["content_type"] = df["content_type"].map({"application/pdf": "pdf", "text/html": "html"})

        return df[["zoteroID", "title", "doi", "content_type", "path"]].set_index("zoteroID")

    def has_doi(self, doi: str) -> bool:
        req = (
            """
                    SELECT itemDataValues.value
                    FROM items
                    JOIN itemData itemDataDOI ON items.itemID = itemDataDOI.itemID
                    JOIN itemDataValues ON itemDataDOI.valueID = itemDataValues.valueID
                    WHERE itemDataValues.value = ?
                    """,
            (doi,),
        )
        try:
            with self._database.cursor() as c:
                doi_exists = c.execute(*req).fetchone()
        except sqlite3.DatabaseError:
            time.sleep(1)
            with self._database.cursor() as c:
                doi_exists = c.execute(*req).fetchone()
        return doi_exists is not None

    def has_local(self, doi: str) -> str | None:
        req = (
            """
            SELECT itemAttachments.contentType
            FROM items
            JOIN itemData itemDataDOI ON items.itemID = itemDataDOI.itemID
            JOIN itemDataValues DOI ON itemDataDOI.valueID = DOI.valueID
            JOIN itemAttachments ON items.itemID = itemAttachments.parentItemID
            WHERE DOI.value = ?
            """,
            (doi,),
        )
        try:
            with self._database.cursor() as c:
                local_content_type = c.execute(*req).fetchone()
        except sqlite3.DatabaseError:
            time.sleep(1)
            with self._database.cursor() as c:
                local_content_type = c.execute(*req).fetchone()

        return local_content_type

    def parse_and_export_all(
        self, folder_path: str | Path, filter_dois: Optional[Iterable[str]] = None, overwrite: bool = False
    ):
        """List and parse any papers in the Zotero library using GROBID and export the results as XML files.

        Parameters
        ----------
        folder_path : str | Path
            Path to the folder where the XML files will be saved.
        filter_dois : Optional[Iterable[str]], optional
            List of DOIs to parse. If specified, only papers whose doi is in the list will be parsed.
            Otherwise, all papers are parsed.
        overwrite : bool, optional
            If False (by default), articles that have already been parsed are ignored.
        """
        import xmltodict

        from .sci_parser import ParsedPaper

        if isinstance(folder_path, str):
            folder_path = Path(folder_path).absolute()
        folder_path.mkdir(exist_ok=True, parents=True)

        df = self.list_local_path(prefer="pdf", check_exists=True)
        df = df[df["content_type"] == "pdf"]

        if filter_dois is not None:
            df = df.loc[df.doi.isin(filter_dois)]

        n_ignored = 0

        with RichProgress("Parsing and exporting papers", len(df), f"{len(df)} papers parsed in {{t}}s") as progress:
            for zoteroID, paper in df.iterrows():
                if not overwrite and (folder_path / f"{zoteroID}.xml").exists():
                    n_ignored += 1
                    progress.update(advance=1)
                    progress.done_message = f"{len(df)-n_ignored} papers parsed in {{t}}s (already parsed: {n_ignored})"
                    continue

                paper_xml = ParsedPaper.parse(paper["path"], classify_section=True).to_xml_dict(
                    export_text_segments=True
                )
                paper_xml["@DOI"] = paper["doi"]
                paper_xml["@title"] = paper["title"]
                paper_xml["@path"] = paper["path"]
                paper_xml["@zoteroID"] = zoteroID

                with open(folder_path / f"{zoteroID}.xml", "w") as f:
                    f.write(xmltodict.unparse(dict(paper=paper_xml), pretty=True))

                progress.update(advance=1)

    def fetch_item_field(self, field: str) -> pd.Series:
        """Query the local Zotero SQL database to retrieve a field for every item of the specified library and collection. The field is identified by its name.

        Note that any informations related to authors are not stored as field in Zotero database.

        Parameters
        ----------
        field: str
            The name of the field to retrieve. Standard field names are: 'title', 'abstractNote', 'DOI', 'date', 'ISSN'.

        Returns
        -------
        pd.Series
            A Series with the index being the zoteroID and the field as values.
        """  # noqa: E501
        with self._database.cursor() as c:
            field_ids = c.execute("SELECT fieldID FROM fieldsCombined WHERE fieldName = ?", (field,)).fetchone()
            if field_ids is None:
                all_fields = c.execute("SELECT fieldName FROM fieldsCombined").fetchall()
                raise ValueError(
                    f"Field {field} not found. Available fields are: {', '.join(f[0] for f  in all_fields)}"
                )
            field_id = field_ids[0]

            req = """
                SELECT items.key, itemDataValues.value
                FROM items
                JOIN itemData ON items.itemID = itemData.itemID
                JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
                """
            where_req = "WHERE items.libraryID = ? AND itemData.fieldID = ?"
            req_args = [self._library_id, field_id]

            if self._collection is not None:
                req += """
                JOIN collectionItems ON items.itemID = collectionItems.itemID
                """
                where_req += " AND collectionItems.collectionID = ?"
                req_args.append(self._collection_id)

            query = c.execute(req + where_req, req_args)
            df = pd.DataFrame.from_records(data=query.fetchall(), columns=["zoteroID", field]).set_index("zoteroID")
            return df[field]

    def fetch_item_fields(self, fields: Iterable[str]) -> pd.DataFrame:
        """Fetch multiple fields for every item of the specified library and collection.

        Parameters
        ----------
        fields : Iterable[str]
            List of fields to retrieve. Standard field names are: 'title', 'abstractNote', 'DOI', 'date', 'ISSN'.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the index being the zoteroID and the fields as columns.
        """
        fields_data = {}
        for field in fields:
            if field == "year":
                fields_data[field] = self.fetch_year()
            elif field == "abstract":
                fields_data[field] = self.fetch_item_field("abstractNote")
            elif field == "doi":
                fields_data[field] = self.fetch_item_field("DOI")
            else:
                fields_data[field] = self.fetch_item_field(field)

        return pd.DataFrame(fields_data)

    def fetch_year(self) -> pd.Series:
        """Fetch the publication year of every item of the specified library and collection.

        Returns
        -------
        pd.Series
            A Series with the index being the zoteroID and the publication year as values.
        """
        dates = self.fetch_item_field("date")
        return dates.str.extract(r"(\d{4})").squeeze().astype(int)
