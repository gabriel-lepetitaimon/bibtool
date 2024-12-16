from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xmltodict
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

catpuccin = [
    "#d20f39",
    "#e64553",
    "#ea76cb",
    "#8839ef",
    "#dc8a78",
    "#dd7878",
    "#fe640b",
    "#df8e1d",
    "#40a02b",
    "#179299",
    "#04a5e5",
    "#209fb5",
    "#1e66f5",
    "#7287fd",
]


def load_xml_corpus(folder: str | Path):
    """Load a corpus of XML files containing papers exported from Zotero.

    Parameters
    ----------
    folder : str | Path
        Path to the folder containing the XML files.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the content of the XML files and the following columns:
            - ``DOI``
            - ``title``
            - ``abstract``
            - ``introduction`` (text related to the introduction)
            - ``method`` (text related to the methods)
            - ``evaluation`` (text related to the validation and results)
            - ``conclusion`` (text related to the conclusion)

    """
    from .sci_parser import SectionType

    folder = Path(folder)
    df = pd.DataFrame(columns=["DOI", "title", "abstract"] + [s.value for s in SectionType])
    df.index.rename("zoteroID", inplace=True)
    for file in folder.glob("*.xml"):
        with open(file, "rb") as f:
            xml = xmltodict.parse(f)["paper"]
            df.loc[file.stem, "abstract"] = xml["abstract"]
            for k, v in xml.items():
                if k.startswith("@"):
                    df.loc[file.stem, k[1:]] = v

            for segment in xml["text_segments"]["segments"]:
                df.loc[file.stem, segment["@type"]] = segment.get("#text", "")
    return df


def freq_graph(data, *words, column="abstract", keep_order=False, color_scale="BuPu", **labelled_words):
    labels = [l.replace("_", " ").strip() for l in list(words) + list(labelled_words.keys())]
    words = list(words) + list(labelled_words.values())

    words_freq = {
        word: data[column].str.lower().str.contains(word, regex=True).groupby(data.year).mean() for word in words
    }
    words_freq = pd.DataFrame(words_freq)
    if not keep_order:
        chronology = (words_freq / (words_freq.max() + 1e-3)) ** 3
        chronology = (
            (chronology / (chronology.sum() + 1e-3)).mul(words_freq.index, axis=0).sum().reset_index().sort_values(by=0)
        )
        revert_lookup = np.arange(len(chronology))
        revert_lookup[chronology.index] = revert_lookup
        words_freq = words_freq[chronology["index"]]
        labels = [labels[c] for c in chronology.index]

    freq_max = words_freq.max().max()

    fig = make_subplots(rows=len(words), cols=1, shared_xaxes=True, vertical_spacing=0)
    annotations = []

    color_scale = plt.cm.get_cmap(color_scale)

    for i, (word, freq) in enumerate(words_freq.items()):
        year = freq.index
        freq = np.clip(savgol_filter(freq, 5, 3), 0, 1)
        i_shifted = ((i + 1) / len(words_freq)) * 0.75 + 0.25
        color = "rgb" + str(tuple(int(_ * 255) for _ in color_scale(i_shifted)[:3]))

        fig.add_trace(
            go.Scatter(
                x=year,
                y=freq,
                mode="lines",
                name=word,
                line_color=color,
                line=dict(shape="spline", smoothing=1.3),
                fill="tonexty",  # Add fill below the curve
            ),
            row=i + 1,
            col=1,
        )

        fig.update_yaxes(tickvals=[freq.max()], tickmode="array", row=i + 1, col=1)

        annotations.append(
            dict(
                text=labels[i],
                xref="paper",
                yref=f"y{i+1}",
                x=0,
                y=freq_max,  # Center vertically in subplot
                showarrow=False,
                font=dict(size=30, color="rgba(0,0,0,0.6)"),
                textangle=0,
                yanchor="top",
            )
        )

    fig.update_layout(
        showlegend=False,
        annotations=annotations,
        width=800,
        height=100 * len(words),
        template="simple_white",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_xaxes(tickformat="linear", showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(tickformat=",.0%", range=[0, freq_max + 0.05], ticklabelposition="outside bottom", showgrid=False)

    fig.show()
