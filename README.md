# Bib Tool

``bibtool`` bundles a series of tools to perform systemic lexicographic analysis of scientific papers. 

The lexicographic analysis is enabled by 4 preliminary steps which aim at creating the corpus of scientific papers:
1. Listing every papers of a research field
2. Downloading the abstract and full-text pdf.
3. Automatic filtering of the papers based on their title and abstract using a LLM.
4. Extraction and segmentation of sections in the papers full-text (i.e. Introduction, Method, Evaluation, Conclusion)

Finally ``bibtool`` includes some lexicograpic function to analyse the corpus (e.g. word clouds and occurrences graph...)

> [!WARNING]
> This package was created for a personal project and won't be developed actively. 
> 
> I strongly encourage anyone who wants to replicate such lexicographic pipeline to use the provided code as an 
> inspiration rather than a complete solution.

## Installation

```bash
pip install https://github.com/gabriel-lepetitaimon/bibtool/archive/main.zip
```

Some features also requires a local installation of [Zotero](https://www.zotero.org/).


## Usage

### I. Listing every papers of a research field
The module ``bibtool.search_papers`` provide functions to automatically search bibliographic repository.

First create a ``LiteratureSearch`` object to gather the results of the search. Then scan repositery with an appropriate
``SearchQuery`` object.

```python
from bibtool.search_papers import LiteratureSearch, SearchQuery

search = LiteratureSearch()
query = SearchQuery("Fundus Vessels Segmentation", min_year=2000, max_year=2025)
search.scan(query, source='scopus')
```
Additionally you can import results from a csv file (e.g. from a previous search or from Publish or Perish) into the
``LiteratureSearch`` object. The duplicated entries will be automatically merged.

```python
search.import_csv("path/to/file.csv")
```

The result of the search is stored in a pandas DataFrame accessible with the attribute ``search.results``. You may export
those to a csv file with the method ``search.export_csv("path/to/file.csv")``.


### II. Downloading the abstract and full-text pdf.
Downloading the full-text and even the abstracts of the papers is not trivial for not open-access papers.
The only reliable solution I found until now requires an auto-clicker to automate the "manual" download of papers with 
Zotero Connector. It requires a local installation of Zotero and the Zotero Connector extension in your browser properly
configured to be triggered by a given hotkey. 
It also requires the ``pyautogui``. See [PyAutoGui Installation Page](https://pyautogui.readthedocs.io/en/latest/install.html) 
for more information on its installation for your OS.

1. Create a Zotero Library (and optionally a collection) where all the papers will be downloaded. I strongly advice 
to disable the automatic sync of the library so the papers are not uploaded to your Zotero cloud. Make sure the library
is selected in Zotero so that the papers are downloaded in the right place.

2. Create a python script which instantiate a ``ZoteroLibrary`` with the library key and the collection key. This
allows the python script to know when a download is finished to move to the next paper or skip already downloaded paper.
Use to the method ``download_papers`` to download a series of papers identified by their DOI. You can specify the hotkey
to trigger the Zotero Connector with the argument ``zotero_connector_hotkey``.
    ```python
    from bibtool.zotero import ZoteroLibrary

    zlib = ZoteroLibrary("LibraryName", "OptionalCollectionName")

    papers_dois = search.results.index
    zlib.download_papers(papers_dois, zotero_connector_hotkey="ctrl+period")
    ```

3. Run the script and make sure that Firefox windows has the focus. The script will open a new tab with the provided
doi, wait for the page to load, trigger Zotero Connector hotkey, wait for the paper to be downloaded and move on to the 
next doi. You can't use your graphical desktop while the script is running.

### III. Automatic filtering of the papers based on their title and abstract using a LLM.
No specific functions are provided in this package for this task but you can check the example script [``filter_abstracts.py``](examples/filter_abstracts.py) which uses a LLM to filter papers based on their abstract.

In particular the script uses ``ZoteroLibrary`` to access the doi, titles and abstract of the papers previously downloaded:
```python
papers_info = zlib.fetch_item_fields(["doi", "title", "abstract"])
```

### IV. Extraction and segmentation of sections in the papers full-text
To extract the section from the article pdf, ``bibtool`` relies on [GROBID](https://github.com/kermitt2/grobid) 
and its python binding: [``scipdf``](https://github.com/titipata/scipdf_parser). Following scipdf advice, I recommand to install and execute [GROBID via docker](https://grobid.readthedocs.io/en/latest/Grobid-docker/). Note that scipdf requires ``en_core_web_sm`` which
can be install with:

```bash
python -m spacy download en_core_web_sm
```

Once everything is installed and GROBID is running, you can use the method ```ZoteroLibrary.parse_and_export_all``` 
to extract the sections of the papers. The method will create a new directory with the extracted sections in xml format.

```python
zlib.parse_and_export_all("path/to/output/directory")
```

### V. Lexicographic analysis
To open the corpus generated by the previous step as a pandas DataFrame use:
```python
from bibtool.lexical_analysis import load_xml_corpus

corpus = load_xml_corpus("path/to/output/directory")
```

If everything wen't well this dataframe should contains the columns: ``doi``, ``title``, ``abstract`` as well as 
``introduction``, ``method``, ``evaluation``, ``conclusion``. The last four columns stores the text of the corresponding
section of each paper.