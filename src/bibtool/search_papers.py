from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from enum import Enum
from operator import attrgetter
from pathlib import Path
from typing import List, Literal, Optional, Self, TypeAlias

import pandas as pd
import requests
from metapub import PubMedFetcher

from .utils import RichProgress


class SearchEngine(str, Enum):
    CROSSREF = "crossref"
    PUBMED = "pubmed"
    SCOPUS = "scopus"
    EXTERNAL = "external"

    @classmethod
    def parse_engines(cls, engines: Optional[Self | List[Self] | str | List[str]]) -> List[Self]:
        if engines is None:
            return [cls.SCOPUS, cls.PUBMED]
        elif isinstance(engines, cls):
            return [engines]
        elif isinstance(engines, str):
            return [cls(engines)]
        elif isinstance(engines, list):
            return [cls(engine) for engine in engines]
        else:
            raise ValueError(f"Invalid type for engines: {type(engines)}")


class PublicationType(str, Enum):
    BOOK = "book"
    CONFERENCE = "conference"
    JOURNAL = "journal"
    PREPRINT = "preprint"
    THESIS = "thesis"
    OTHER = "other"


@dataclass(repr=False)
class SearchQuery:
    """Dataclass object representing a SearchQuery used to search for papers in bibliographic repositories."""

    #: The query to search for.
    query: str

    #: Minimum year of publication. If provided, only papers published after this year will be considered.
    min_year: Optional[int] = None

    #: Maximum year of publication. If provided, only papers published before this year will be considered.
    max_year: Optional[int] = None

    def __repr__(self):
        """Return a string representation of the object.

        (For readability, the fields kept at their default values are excluded.)

        """
        nodef_f_vals = (
            (f.name, attrgetter(f.name)(self))
            for f in dataclasses.fields(self)
            if attrgetter(f.name)(self) != f.default
        )

        nodef_f_repr = ", ".join(f"{name}={repr(value)}" for name, value in nodef_f_vals)
        return f"{self.__class__.__name__}({nodef_f_repr})"

    def __post_init__(self):
        assert isinstance(self.query, str), "query should be a string"

        if self.min_year is not None:
            self.min_year = int(self.min_year)
        if self.max_year is not None:
            self.max_year = int(self.max_year)

        if self.min_year is not None and self.max_year is not None:
            assert self.min_year <= self.max_year, "min_year should be less than or equal to max_year"

    @classmethod
    def parse(cls, query: ParsableQuery) -> Self:
        if isinstance(query, cls):
            return query
        if query.startswith("SearchQuery("):
            return eval(query)
        return cls(query)


ParsableQuery: TypeAlias = str | SearchQuery


class LiteratureSearch:
    def __init__(self):
        self.results = None

    def scan(
        self,
        query: ParsableQuery,
        max_results_per_year: Optional[int] = None,
        source: Optional[SearchEngine | List[SearchEngine] | str | List[str]] = None,
    ) -> pd.DataFrame:
        """Scan bibliographic repository with the provided query.

        Parameters
        ----------
        query : ParsableQuery
            The query to search for.

        max_results_per_year : Optional[int], optional
            Limit the number of papers found per year.

        source : Optional[SearchEngine  |  List[SearchEngine]  |  str  |  List[str]], optional
            List of SearchEngine to query. If None (by default), all are searched.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the following fields:
                - doi
                - title
                - first_author
                - publication
                - publication_type (i.e. journal, book, etc.)
                - authors (separated by "|")
                - year
                - date
                - affiliation (separated by "|", depends on the source)
                - keywords  (separated by "|", depends on the source)
                - abstract (depends on the source)
        """
        df = pd.DataFrame()

        query = SearchQuery.parse(query)
        engines = SearchEngine.parse_engines(source)
        with RichProgress("Scanning search engines...", len(set(engines))) as progress:
            if SearchEngine.SCOPUS in engines:
                progress.update(message="Scanning Scopus...")
                df_scopus = scan_scopus(query, max_results_per_year, progress=progress)
                df_scopus["source"] = SearchEngine.SCOPUS.value
                df = pd.concat([df, df_scopus])
                progress.update(advance=1)
            if SearchEngine.CROSSREF in engines:
                progress.update(message="Scanning CrossRef...")
                df_crossref = scan_crossref(query, max_results_per_year)
                df_crossref["source"] = SearchEngine.CROSSREF.value
                df = pd.concat([df, df_crossref])
                progress.update(advance=1)
            if SearchEngine.PUBMED in engines:
                ignore_pmid = df["pmid"].tolist() if "pmid" in df.columns else None
                progress.update(message="Scanning PubMed...")
                df_pubmed = scan_pubmed(query, max_results_per_year, ignore_pmid=ignore_pmid, progress=progress)
                df_pubmed["source"] = SearchEngine.PUBMED.value
                df = pd.concat([df, df_pubmed])
                progress.update(advance=1)

            progress.done_message = "Literature scanned in {t}s."

        df = self.remove_duplicates(df)
        df["query"] = query.query
        df.fillna("", inplace=True)
        df["year"] = df["year"].astype(int)
        self._append_results(df)
        return df

    def import_csv(
        self,
        path: str,
        format: Optional[Literal["auto", "PublishOrPerish"]] = "auto",
        source: Optional[str] = None,
        query: Optional[str] = None,
        fetch_missing_doi: bool = True,
    ) -> pd.DataFrame:
        df_raw = pd.read_csv(path)
        if format == "auto":
            if "CitesPerAuthor" in df_raw.columns:
                format = "PublishOrPerish"
            else:
                format = None
        if format == "PublishOrPerish":
            # print(df_raw.columns)
            df_csv = pd.DataFrame()
            df_csv["title"] = df_raw["Title"]
            df_csv["authors"] = df_raw["Authors"]
            df_csv["first_author"] = df_raw["Authors"].str.split(",").str[0]
            df_csv["publication"] = df_raw["Source"]
            df_csv["publication_type"] = df_raw["Type"]
            df_csv["year"] = df_raw["Year"].astype(int)
            df_csv["citations_count"] = df_raw["Cites"]
            df_csv["doi"] = df_raw["DOI"]
            df_csv["abstract"] = df_raw["Abstract"]
            df_csv["issn"] = df_raw["ISSN"]
        else:
            df_csv = df_raw

        assert all(_ in df_csv.columns for _ in ("doi", "title", "first_author", "publication", "authors")), (
            "The CSV file must contain the following columns: "
            "'doi', 'title', 'first_author', 'publication', 'authors'"
        )
        df_csv["source"] = source or SearchEngine.EXTERNAL.value
        df_csv["query"] = query or ""

        if fetch_missing_doi:
            missing_dois = df_csv[df_csv["doi"].isna()]
            with RichProgress("Fetching DOIs from CrossRef...", len(missing_dois)) as progress:
                for i, row in missing_dois.iterrows():
                    doi = fetch_doi_from_crossref(row["title"], row["first_author"], row["publication"], row["year"])
                    df_csv.at[i, "doi"] = doi
                    progress.update(advance=1)

        self._append_results(df_csv)
        return df_csv

    def export_csv(self, path: str | Path) -> None:
        self.results.to_csv(path, index=True)

    @classmethod
    def remove_duplicates(cls, df: pd.DataFrame) -> pd.DataFrame:
        if "doi" not in df.columns:
            return df
        df = df.replace("", pd.NA)
        df_by_doi = df.groupby("doi")
        sources = df_by_doi["source"].apply(
            lambda x: "|".join(set().union(*[set(_.split("|")) for _ in x if isinstance(_, str)])) or ""
        )
        df = df_by_doi.first()
        df["source"] = sources
        return df.reset_index(names="doi").fillna("")

    def _append_results(self, df: pd.DataFrame) -> None:
        if self.results is None:
            self.results = df.set_index("doi")
        else:
            self.results = self.remove_duplicates(pd.concat([self.results.reset_index(names="doi"), df])).set_index(
                "doi"
            )

    @classmethod
    def import_results(cls, path: str | Path) -> Self:
        """Import the results from XML files."""
        raise NotImplementedError


def scan_pubmed(
    query: SearchQuery,
    max_results: Optional[int] = None,
    max_result_per_query=100,
    ignore_pmid: Optional[List[str]] = None,
    *,
    progress: Optional[RichProgress] = None,
) -> pd.DataFrame:
    """Fetch articles from PubMed based on a query.

    Parameters
    ----------
    query : str
        _description_
    max_results : Optional[int], optional
        Limit the number of fetched articles, by default None

    max_result_per_query : int, optional
        Enable pagination by specifying the maximum number of results per query, by default 100

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the fetched articles with the following columns:
        - doi
        - title
        - first_author
        - publication
        - publication_type (i.e. journal, book, etc.)
        - authors (separated by "|")

        - year
        - date
        - affiliation (separated by "|")
        - keywords  (separated by "|")
        - abstract
        - pmid
    """
    fetch = PubMedFetcher()
    articles = {}

    def format_date(date):
        return date.strftime("%Y-%m-%d") if date else None

    pmids = []
    periods = query.periods or [(None, None)]
    for since, until in periods:
        n_period = 0
        while True:
            if max_results is not None:
                retmax = min(max_results - n_period, max_result_per_query)
                if max_result_per_query <= 0:
                    break
                results = fetch.pmids_for_query(query.query, since=since, until=until, retmax=retmax, retstart=n_period)
                if not results:
                    break
                pmids.extend(results)
                n_period += len(results)

    if ignore_pmid:
        ignore_dois = set(ignore_pmid)
        pmids = [pmid for pmid in pmids if pmid not in ignore_dois]

    with RichProgress("Fetching articles infos...", len(pmids), parent=progress) as p:
        for pmid in pmids:
            articles[pmid] = fetch.article_by_pmid(pmid)
            p.update(completed=len(articles))
        p.done_message = f"Fetched infos for {len(articles)} articles from pubmed in {{t}} seconds."

    def parse_publication_type(article):
        if hasattr(article, "book"):
            return PublicationType.BOOK
        else:
            return next(iter(article.publication_types.values()), PublicationType.OTHER)

    articles_info = [
        dict(
            doi=article.doi,
            title=article.title,
            first_author=article.authors[0],
            authors="|".join(article.authors),
            publication=article.journal,
            publication_type=parse_publication_type(article),
            year=article.year,
            date=format_date(article.history.get("accepted", None)),
            pmid=pmid,
            abstract=article.abstract,
            keywords="|".join(article.keywords),
            affiliation="|".join(set().union(*[set(auth.affiliations) for auth in article.author_list])),
        )
        for pmid, article in articles.items()
    ]

    return pd.DataFrame(articles_info)


def scan_scopus(
    query: SearchQuery,
    max_results_per_period: Optional[int] = None,
    max_result_per_query=10,
    *,
    progress: Optional[RichProgress] = None,
) -> pd.DataFrame:
    from .utils.auth_cache import request_user_auth

    oauth = request_user_auth(
        "scopus", "Please enter your [link=https://dev.elsevier.com/apikey/manage]Scopus API key[/link]:"
    )
    retried = False

    articles_info = []
    with RichProgress("Fetching articles infos...", max_results_per_period, parent=progress) as p:
        since = query.min_year
        until = query.max_year
        if since and until:
            periods = [(_, _) for _ in range(since, until + 1)]
        else:
            periods = [(since, until)]

        for since, until in periods:
            if since and until:
                date = str(since) if since == until else f"{since}-{until}"
            else:
                if not since:
                    date = f"-{until}"
                if not until:
                    date = f"{since}-"

            p.update(message=f"Fetching articles infos [{date}]")

            start = 0
            n_period = 0
            while True:
                count = (
                    min(max_results_per_period - n_period, max_result_per_query)
                    if max_results_per_period
                    else max_result_per_query
                )
                if count <= 0:
                    break

                req = requests.get(
                    "https://api.elsevier.com/content/search/scopus",
                    params=dict(
                        accept="application/json",
                        query=query.query,
                        apiKey=oauth,
                        count=count,
                        start=start,
                        date=date,
                        field="prism:doi,dc:title,dc:creator,author,prism:publicationName,prism:aggregationType,prism:coverDate,dc:identifier,dc:description,authkeywords,citedby-count,affiliation,prism:url,pubmed-id",
                    ),
                )

                if req.status_code == 401:
                    if retried:
                        warnings.warn(f"Invalid Scopus API Key: {req.text}", stacklevel=2)
                        break
                    retried = True
                    oauth = request_user_auth(
                        "scopus",
                        "Please enter your [link=https://dev.elsevier.com/apikey/manage]Scopus API key[/link]:",
                        overwrite=True,
                    )
                    continue
                elif req.status_code != 200:
                    warnings.warn(f"Error {req.status_code}: {req.text}", stacklevel=2)
                    break

                try:
                    results = req.json()["search-results"]["entry"]
                except KeyError:
                    break
                start += len(results)

                valid_articles = [
                    dict(
                        doi=article["prism:doi"],
                        title=article["dc:title"],
                        first_author=article["dc:creator"],
                        authors="|".join([author["authname"] for author in article.get("author", [])]),
                        publication=article.get("prism:publicationName", ""),
                        publication_type=article.get("prism:aggregationType", ""),
                        year=int(article["prism:coverDate"][:4]) if "prism:coverDate" in article else None,
                        date=article.get("prism:coverDate", ""),
                        scopus_id=article["dc:identifier"],
                        pmid=article.get("pubmed-id", ""),
                        abstract=article.get("dc:description", ""),
                        keywords="|".join(article.get("authkeywords", [])),
                        citations_count=article.get("citedby-count", 0),
                        affiliation="|".join({aff.get("affilname", "") for aff in article.get("affiliation", [])}),
                        affiliation_country="|".join(
                            {aff.get("affiliation-country", "") or "" for aff in article.get("affiliation", [])}
                        ),
                    )
                    for article in results
                    if "prism:doi" in article and "dc:title" in article and "dc:creator" in article
                ]
                n_period += len(valid_articles)
                articles_info.extend(valid_articles)
                p.update(completed=n_period)

    return pd.DataFrame(articles_info)


def scan_crossref(query, max_results: Optional[int] = None):
    raise NotImplementedError


def fetch_doi_from_crossref(title: str, authors: str, publication: str, year: int | str) -> Optional[str]:
    req = requests.get(
        "https://api.crossref.org/works",
        params={"query.bibliographic": f"{title} + {publication} + {year}", "query.author": authors},
    )
    if req.status_code != 200:
        return None
    try:
        return req.json()["message"]["items"][0]["DOI"]
    except (KeyError, IndexError):
        return None
