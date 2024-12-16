import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Self, Tuple, Union

import pandas as pd
import scipdf
import xmltodict
from bs4 import BeautifulSoup

from .utils.math import roman_to_int


@dataclass(eq=False)
class ParsedTable:
    """A class storing the parsed version of table from a scientific article including its label, caption, and data."""

    #: The label of the table (e.g. "Table 1")
    label: str

    #: The caption of the table
    caption: str

    #: The ID of the table in the XML
    figure_id: str

    #: The data of the table as a pandas DataFrame
    data: pd.DataFrame

    def headings(self) -> Tuple[List[str], ...]:
        """

        Returns
        -------
        Tuple[List[str], ...]
            _description_
        """
        if self.data.empty:
            return ()

        row_headings = [h for h in self.data.loc[0] if not h.replace(".", "").isnumeric()]
        if len(self.data.shape) == 2:
            col_headings = [h for h in self.data.loc[:, 0] if not h.replace(".", "").isnumeric()]
        else:
            col_headings = []

        out = ()
        if col_headings:
            out = out + (col_headings,)
        if row_headings:
            out = out + (row_headings,)
        return out

    def to_xml_dict(self) -> Dict[str, str]:
        """Export the table informations to an XML dictionary with the following keys:
        - "@label": The label of the table
        - "@caption": The caption of the table
        - "@figure_id": The ID of the table in the XML
        - "data": The data of the table as a pandas DataFrame

        Returns
        -------
        Dict[str, str]

        Examples
        --------
        >>> table = ParsedTable("Table 1", "This is a table", "tab1", pd.DataFrame([["Method", "acc", "AUC"], ["ours", 0.8, 0.7], ["others", 0.5, 0.4]]))
        >>> table.to_xml_dict()
        {'@label': 'Table 1', '@caption': 'This is a table', '@figure_id': 'tab1', 'data': {'row': [{'cell': ['Method', 'acc', 'AUC']}, {'cell': ['ours', '0.8', '0.7']}, {'cell': ['others', '0.5', '0.4']}]}}
        """  # noqa: E501
        if self.data.empty:
            data = {}
        elif len(self.data.shape) == 1:
            data = {"row": [{"cell": [str(cell) for cell in self.data]}]}
        else:
            data = {"row": [{"cell": [str(cell) for cell in row]} for i, row in self.data.iterrows()]}
        return {
            "@label": self.label,
            "@caption": self.caption,
            "@figure_id": self.figure_id,
            "data": data,
        }

    @classmethod
    def from_xml_dict(cls, data):
        """Load a ParsedTable object from an XML dictionary.

        Parameters
        ----------
        data : Mapping[str, str]
            The XML dictionary containing the informations of the table with the following keys:
            - "@label": The label of the table
            - "@caption": The caption of the table
            - "@figure_id": The ID of the table in the XML
            - "data": The data of the table as a dictionary of rows and cells

        Returns
        -------
        Self

        Examples
        --------
        >>> xml1 = {'@label': 'Table 1', '@caption': 'This is a table', '@figure_id': 'tab1', 'data': {'row': [{'cell': ['Method', 'acc', 'AUC']}, {'cell': ['ours', '0.8', '0.7']}, {'cell': ['others', '0.5', '0.4']}]}}
        >>> ParsedTable.from_xml_dict(xml1)
        ParsedTable(label='Table 1', caption='This is a table', figure_id='tab1', data=        0    1    2
        0  Method  acc  AUC
        1    ours  0.8  0.7
        2  others  0.5  0.4)

        >>> table2 = ParsedTable("Table 2", "This is another table", "tab2", pd.DataFrame([["Method", "acc", "AUC"], ["ours", "0.8", "0.7"], ["others", "0.5", "0.4"]]))
        >>> table2bis = ParsedTable.from_xml_dict(table2.to_xml_dict())
        >>> table2 == table2bis
        True

        """  # noqa: E501
        df_data = [row["cell"] for row in data["data"]["row"]]
        return cls(
            label=data["@label"],
            caption=data["@caption"],
            figure_id=data["@figure_id"],
            data=pd.DataFrame(df_data),
        )

    def __eq__(self, value):
        return (
            self.label == value.label
            and self.caption == value.caption
            and self.figure_id == value.figure_id
            and self.data.equals(value.data)
        )


@dataclass
class ParsedFigure:
    """A class storing the parsed version of a figure from a scientific article including its label, caption, and type."""

    #: The label of the figure (e.g. "Figure 1")
    label: str

    #: The caption of the figure
    caption: str

    #: The type of the figure (e.g. "figure", "table", "chart", etc.)
    type: str

    #: The ID of the figure in the XML
    figure_id: str

    def to_xml_dict(self) -> Dict[str, str]:
        """Export the figure informations to an XML dictionary with the following keys:
        - "@label": The label of the figure
        - "@caption": The caption of the figure
        - "@type": The type of the figure
        - "@figure_id": The ID of the figure in the XML

        Returns
        -------
        Dict[str, str]

        Examples
        --------
        >>> figure = ParsedFigure("Figure 1", "This is a figure", "figure", "fig1")
        >>> figure.to_xml_dict()
        {'@label': 'Figure 1', '@caption': 'This is a figure', '@type': 'figure', '@figure_id': 'fig1'}
        """
        return {
            "@label": self.label,
            "@caption": self.caption,
            "@type": self.type,
            "@figure_id": self.figure_id,
        }

    @classmethod
    def from_xml_dict(cls, data: Mapping[str, str]) -> Self:
        """Load a ParsedFigure object from an XML dictionary.

        Parameters
        ----------
        data : Mapping[str, str]
            The XML dictionary containing the informations of the figure with the following keys:
            - "@label": The label of the figure
            - "@caption": The caption of the figure
            - "@type": The type of the figure
            - "@figure_id": The ID of the figure in the XML

        Returns
        -------
        Self

        Examples
        --------
        >>> xml1 = {'@label': 'Figure 1', '@caption': 'This is a figure', '@type': 'figure', '@figure_id': 'fig1'}
        >>> ParsedFigure.from_xml_dict(xml1)
        ParsedFigure(label='Figure 1', caption='This is a figure', type='figure', figure_id='fig1')

        >>> fig2 = ParsedFigure("Figure 2", "This is another figure", "table", "fig2")
        >>> fig2bis = ParsedFigure.from_xml_dict(fig2.to_xml_dict())
        >>> fig2 == fig2bis
        True

        """
        return cls(
            label=data["@label"],
            caption=data["@caption"],
            type=data["@type"],
            figure_id=data["@figure_id"],
        )


class SectionType(str, Enum):
    INTRODUCTION = "introduction"
    RELATED_WORK = "related work"
    METHOD = "method"
    EVALUATION = "evaluation"
    CONCLUSION = "conclusion"


@dataclass
class ParsedSection:
    heading: str
    num: str
    text: str
    publication_refs: List[str]
    figure_refs: List[str]
    table_refs: List[str]
    type: Optional[SectionType] = None

    @classmethod
    def merge(cls, sections: List[Self]):
        heading = sections[0].heading
        num = sections[0].num
        text = []
        if sections[0].text:
            text.append(sections[0].text)

        for section in sections[1:]:
            if section.heading:
                sub_heading = section.heading
                if section.num:
                    sub_heading = section.num + " " + sub_heading
                text.append(sub_heading + "\n" + "-" * len(sub_heading))
            if section.text:
                text.append(section.text)
        text = "\n".join(text)

        publication_ref = [r for s in sections for r in section.publication_ref]
        figure_ref = [r for s in sections for r in section.figure_ref]
        table_ref = [r for s in sections for r in section.table_ref]

        return cls(heading, num, text, publication_ref, figure_ref, table_ref)

    def to_xml_dict(self):
        return {
            "@heading": self.heading,
            "@num": self.num,
            "@type": self.type if self.type else None,
            "text": self.text,
            "publication_refs": self.publication_refs,
            "figure_refs": self.figure_refs,
            "table_refs": self.table_refs,
        }

    @classmethod
    def from_xml_dict(cls, data):
        return cls(
            heading=data["@heading"],
            num=data["@num"],
            text=data["text"],
            publication_refs=data["publication_refs"],
            figure_refs=data["figure_refs"],
            table_refs=data["table_refs"],
            type=SectionType(data["@type"]) if data["@type"] else None,
        )


@dataclass
class ParsedPaper:
    title: str
    abstract: str
    sections: List[ParsedSection]
    references: List[str]
    figures: List[Dict[str, str]]
    tables: List[Dict[str, str | pd.DataFrame]]
    formulas: List[str]

    @classmethod
    def parse(cls, path: Union[Path, str], *, classify_section: bool = True) -> Self:
        if isinstance(path, str):
            path = Path(path)

        if path.suffix == ".pdf":
            article_xml = scipdf.parse_pdf(str(path.absolute()), soup=False)
            article_soup = BeautifulSoup(article_xml, "xml")

            if len(article_soup.contents) == 0:
                return cls("", "", [], [], [], [], [])

            soup_title = article_soup.find("title", attrs={"type": "main"})
            title = soup_title.text.strip() if soup_title is not None else ""

            abstract = scipdf.parse_abstract(article_soup)
            sections = parse_sections(article_soup)
            references = scipdf.parse_references(article_soup)
            figures, tables = parse_figures_tables(article_soup)
            formulas = scipdf.parse_formulas(article_soup)

        else:
            raise ValueError("Only PDF files are supported for now.")

        if classify_section:
            sections = classify_sections(sections)

        return cls(title, abstract, sections, references, figures, tables, formulas)

    def text_segments(self):
        infos = pd.Series()
        # Extract text from sections
        for sec_type in SectionType:
            infos[sec_type.value] = "\n".join(s.text for s in self.sections if s.type == sec_type).strip()

        # Extract table infos
        infos["table_captions"] = "\n".join(t.caption for t in self.tables)
        table_headings = []
        for table in self.tables:
            headings = ""
            data = table.data
            if data.empty:
                continue

            if len(data.shape) > 1:
                col_headings = [h for h in data.loc[:, 0] if h and not h.replace(".", "").isnumeric()]
                if col_headings:
                    headings += " ; ".join(col_headings)

            row_headings = [h for h in data.loc[0] if h and not h.replace(".", "").isnumeric()]
            if row_headings:
                if headings:
                    headings += " | "
                    row_headings = row_headings[1:]
                headings += " ; ".join(row_headings)

            if headings:
                table_headings.append(headings)
        infos["table_headings"] = "\n".join(table_headings)

        # Extract figure infos
        infos["figure_captions"] = "\n".join(f.caption for f in self.figures)

        return infos.astype(str)

    def pretty_text(self):
        text = []
        for section in self.sections:
            sec_title = section.num + " | " + section.heading.upper()
            if section.type:
                sec_title += " [" + section.type + "]"
            text.append(sec_title + "\n" + section.text)

        return "\n\n==============================\n\n".join(text)

    def to_xml_dict(self, export_text_segments: bool = True):
        data = {
            "title": self.title,
            "abstract": self.abstract,
            "sections": [s.to_xml_dict() for s in self.sections],
            "references": self.references,
            "figures": [f.to_xml_dict() for f in self.figures],
            "tables": [t.to_xml_dict() for t in self.tables],
            "formulas": self.formulas,
        }

        if export_text_segments:
            segments = {"segments": [{"@type": k, "#text": v} for k, v in self.text_segments().items()]}
            data |= {"text_segments": segments}

        return data

    @classmethod
    def from_xml_dict(cls, data):
        sections = [ParsedSection.from_xml_dict(s) for s in data["sections"]]
        figures = [ParsedFigure.from_xml_dict(f) for f in data["figures"]]
        tables = [ParsedTable.from_xml_dict(t) for t in data["tables"]]
        return cls(
            title=data["title"],
            abstract=data["abstract"],
            sections=sections,
            references=data["references"],
            figures=figures,
            tables=tables,
            formulas=data["formulas"],
        )

    def save(self, path: Path | str):
        if isinstance(path, Path):
            path = str(path.absolute())

        data = self.to_xml_dict()
        with open(path, "w") as f:
            xmltodict.unparse(data, output=f, pretty=True)

    @classmethod
    def load(cls, path: Path | str):
        if isinstance(path, Path):
            path = str(path.absolute())

        with open(path, "r") as f:
            data = xmltodict.parse(f.read())
        return cls.from_xml_dict(data)


def parse_sections(article) -> List[ParsedSection]:
    """
    Parse list of sections from a given BeautifulSoup of an article
    """
    article_text = article.find("text")
    divs = article_text.find_all("div", attrs={"xmlns": "http://www.tei-c.org/ns/1.0"})
    sections = []

    for div in divs:
        div_children = list(div.children)
        if len(div_children) == 0:
            continue

        if div_children[0].name == "head":
            heading = div_children[0].text
            n = div_children[0].get("n", "")
            div_children = div_children[1:]
        else:
            heading = ""
            n = ""

        text = []
        for i, p in enumerate(div_children):
            if p is None:
                continue
            try:
                p_text = str(p.text).strip()
            except:
                continue

            # Check if last paragraph is section title
            # if i == len(div_children) - 1 and len(p_text) < 50

            text.append(p_text)
        text = "\n".join(text)

        if heading != "" or text != "":
            ref_dict = scipdf.find_references(div)
            sections.append(
                ParsedSection(
                    heading=heading,
                    num=n,
                    text=text,
                    publication_refs=ref_dict["publication_ref"],
                    figure_refs=ref_dict["figure_ref"],
                    table_refs=ref_dict["table_ref"],
                )
            )
    return sections


def classify_sections(sections: List[ParsedSection]) -> List[ParsedSection]:
    """
    Clean up sections parsed by parse_sections and classify them into:
    - Introduction
    - Related Work (optional)
    - Methods: Materials and Methods, Methodology, Approach
    - Evaluation: Experiments, Results, Validation
    - Conclusion: Discussion, Conclusion, Future Work


    """
    df = pd.DataFrame(sections)

    # === Remove duplicated sections ===
    duplicates = df[df.duplicated(subset=["heading", "num"], keep=False)]
    if not "heading" in duplicates:
        return []

    for d in duplicates.groupby(["heading", "num"]).groups.values():
        d_texts = df.loc[d, "text"]
        longest_d = d_texts.str.len().idxmax()
        src_text = re.sub(r"[^\w]", "", d_texts.loc[longest_d])
        for i in d:
            if i == longest_d:
                continue
            i_text = re.sub(r"[^\w]", "", d_texts.loc[i])
            if src_text[:100] == i_text[:100]:
                df.drop(i, inplace=True)

    df.num = df.num.fillna("").str.strip()
    df.heading = df.heading.fillna("").str.strip()
    heading = df.heading.str.lower()

    # === Identify main sections ===
    # 0. Search for the introduction for reference
    if (intro_sections := df[df.heading.str.contains(r"Introduction", case=False)]).empty:
        intro_num = ""
        intro_title = ""
        sections_tree = {}
        current_section = ("", "Introduction")  #: (section number, section title)
    else:
        intro_num = intro_sections.iloc[0].num
        intro_title = intro_sections.iloc[0].heading
        current_section = ("", "")  #: (section number, section title)
        sections_tree = {}

    # 1. Standard case: sections are numbered (1., 2., 2.1, 2.2, 3., ...)
    if not (digit_num := df.num[df.num.isin(("1", "1."))]).empty:
        no_dot = digit_num.iloc[0] == "1"
        for i, section in df.iterrows():
            sec_num = section.num.strip()
            sec_title = section.heading.strip()
            if not sec_num:
                sections_tree.setdefault(current_section, []).append((i, ""))
            elif (sec_num.endswith(".") and sec_num[:-1].isdigit()) if not no_dot else sec_num.isdigit():
                current_section = sec_num, sec_title.lower()
                sections_tree[current_section] = [(i, "section")]
            elif sec_num.startswith(current_section[0]):
                sections_tree.setdefault(current_section, []).append((i, "subsection"))
            else:
                sections_tree.setdefault(current_section, []).append((i, ""))

    # 2. Section are numbered in roman numerals
    elif heading.str.startswith("i. ").any():
        latin_num_re = re.compile(r"^(?:ix|iv|v?i{0,3})\.")
        for i, section in df.iterrows():
            sec_title = section.heading.strip()
            if latin_num_re.match(sec_title.lower()):
                sec_num, sec_title = sec_title.split(".", 1)
                sec_num = f"{roman_to_int(sec_num)}."
                df.loc[i, "num"] = sec_num
                df.loc[i, "heading"] = sec_title.strip()
                current_section = sec_num, sec_title.lower()
                sections_tree[current_section] = [(i, "section")]
            else:
                sections_tree.setdefault(current_section, []).append((i, "subsection"))

    # 3. Main section are capitalized
    elif any(sec_head.isupper() for sec_head in df.heading):
        sec_num = 0
        for i, section in df.iterrows():
            sec_title = section.heading.strip()
            sec_title_words = [w.strip() for w in sec_title.split()]
            if sec_title and sec_title_words[0].isupper():
                sec_title = []
                for w in sec_title_words:
                    if not w.isupper():
                        break
                    sec_title.append(w)
                sec_title = " ".join(sec_title)

                current_section = f"{sec_num}.", sec_title.lower()
                sections_tree[current_section] = [(i, "section")]
                sec_num += 1
            else:
                sections_tree.setdefault(current_section, []).append((i, "subsection"))
    else:
        for i, section in df.iterrows():
            sec_num = section.num.strip()
            sec_title = section.heading.strip()
            if not sec_title:
                sections_tree.setdefault(current_section, []).append((i, ""))
            else:
                current_section = sec_num, sec_title.lower()
                sections_tree.setdefault(current_section, []).append((i, "section"))

    if sections_tree:
        # === Merge sub sections into main sections ===
        sections = []
        for (num, title), subsections in sections_tree.items():
            text = []
            for sub_id, sub_type in subsections:
                if sub_type == "subsection":
                    subsectitle = df.loc[sub_id, "num"] + " " + df.loc[sub_id, "heading"]
                    text.append("\n" + subsectitle + "\n" + "-" * len(subsectitle))
                elif sub_type == "":
                    if not text:
                        text.append("")
                    text[-1] += df.loc[sub_id, "heading"] + " " + df.loc[sub_id, "text"]
                if df.loc[sub_id, "text"]:
                    text.append(df.loc[sub_id, "text"])
            section = ParsedSection(
                heading=title,
                num=num,
                text="\n".join(text),
                publication_refs=[r for i in subsections for r in df.loc[i[0], "publication_refs"]],
                figure_refs=[r for i in subsections for r in df.loc[i[0], "figure_refs"]],
                table_refs=[r for i in subsections for r in df.loc[i[0], "table_refs"]],
            )
            sections.append(section)
    else:
        sections = ParsedSection(**df.to_dict(orient="records"))

    # === Classify sections ===
    section_titles = pd.Series([s.heading for s in sections]).str.lower()
    section_class = pd.Series(index=section_titles.index, dtype="string").fillna("")

    # 1. Introduction
    intro_idx = section_titles.str.contains(r"introduction|background").idxmax()
    section_class[intro_idx] = SectionType.INTRODUCTION.value

    # 2. Related Work
    related_work_idx = section_titles.str.contains(r"related work|previous work|literature review")
    if related_work_idx.any():
        section_class[related_work_idx] = SectionType.RELATED_WORK.value

    # 3. Conclusion
    conclusion_idx = section_titles.str.contains(r"discussion|conclusion|future work")
    if conclusion_idx.any():
        section_class[conclusion_idx] = SectionType.CONCLUSION.value

    # 4. Evaluation
    evaluation_idx = section_titles.str.contains(r"experiments|results|validation|evaluation")
    if evaluation_idx.any():
        section_class[evaluation_idx] = SectionType.EVALUATION.value

    # 5. Methods
    methods_idx = section_titles.str.contains(r"method")
    methods_idx = methods_idx & (section_class == "")
    if not methods_idx.any():
        if intro_idx + 1 < len(section_class):
            section_class[intro_idx + 1] = SectionType.METHOD.value
    else:
        section_class[methods_idx] = SectionType.METHOD.value

    # Propagate classification to subsections
    last_class = ""
    for i in section_class.index:
        if section_class[i] != "":
            last_class = section_class[i]
            if section_class[i] == SectionType.CONCLUSION.value:
                break
        else:
            section_class[i] = last_class

    for section, sec_class in zip(sections, section_class, strict=True):
        section.type = sec_class

    return sections


def parse_figures_tables(article) -> Tuple[List[ParsedFigure], List[ParsedTable]]:
    """
    Parse list of figures/tables from a given BeautifulSoup of an article
    """
    figures_list = []
    tables_list = []

    figures = article.find_all("figure")
    for figure in figures:
        figure_type = figure.attrs.get("type") or "figure"
        figure_id = figure.attrs.get("xml:id") or ""
        if (fig_head := figure.find("head")) is not None:
            label = fig_head.text
        elif (figlabel := figure.find("label")) is not None:
            label = figlabel.text
        if figure_type == "table":
            caption = figdesc.text if (figdesc := figure.find("figDesc")) is not None else ""
            data = [[cell.text for cell in row.find_all("cell")] for row in figure.table.find_all("row")]
            tables_list.append(ParsedTable(label=label, figure_id=figure_id, caption=caption, data=pd.DataFrame(data)))
        else:
            figures_list.append(ParsedFigure(label=label, type=figure_type, figure_id=figure_id, caption=figure.text))
    return figures_list, tables_list
