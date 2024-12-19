"""Example of abstract analysis using a LLM.

This script uses the `pipeline` method from the `transformers` library to generate responses to a series of questions about the abstracts of research papers.

The `process_article` function filters papers to keep only those who propose a segmentation or classification method for fundus vessels. It checks the language, image type, contribution, and segmented structure in the paper.
"""  # noqa: E501

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from bibtool.utils import RichProgress
from bibtool.zotero import ZoteroLibrary

LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


def analyse_abstracts(zlib: ZoteroLibrary, output_csv: str | Path) -> None:
    """Analyze abstracts from a Zotero library using a LLM model.

    The function will analyze the abstracts of the articles in the Zotero library and save the results in a CSV file.
    If the CSV file already exists, the function will load the existing analysis and continue from there.

    Parameters
    ----------
    zlib : ZoteroLibrary
        The zotero library to fetch articles from.
    output_csv : str | Path
        The output CSV file to save the analysis.
    """
    # === Fetch articles, title and abstract from Zotero ===
    articles = zlib.fetch_item_fields(["doi", "title", "abstract"]).dropna()
    articles.drop_duplicates(subset="title", inplace=True)

    # === Uses doi as index (rather than Zotero key) ===
    articles.set_index("doi", inplace=True)

    # === Load existing analysis ===
    if Path(output_csv).exists():
        articles_csv = pd.read_csv(output_csv).set_index("doi")
        articles["language"] = articles_csv["language"]
        articles["image"] = articles_csv["image"]
        articles["contribution"] = articles_csv["contribution"]
        articles["structure"] = articles_csv["structure"]
        articles["fundus_vessel_segmentation"] = articles_csv["fundus_vessel_segmentation"]
        articles["fundus_vessel_classification"] = articles_csv["fundus_vessel_classification"]

    # === Create LLM pipeline ===
    pipe, gen_args = create_llm_pipeline()

    # === Analyze abstracts ===
    try:
        with RichProgress("Analyzing abstracts", len(articles), "Done analyzing {t} seconds") as progress:
            for i, art in articles.iterrows():
                if pd.notna(articles.loc[i, "language"]):
                    progress.update(advance=1)
                    continue
                try:
                    result = process_article(art["title"], art["abstract"], pipe, gen_args)
                    for k, v in result.items():
                        articles.loc[i, k] = v
                    progress.update(advance=1)
                except Exception as e:
                    print(f"Error in {art['title']}: {e}")
                    progress.update(total=progress.total - 1)
    finally:
        pd.DataFrame(articles).to_csv(output_csv)


def process_article(title, abstract, pipe, gen_args) -> Dict:
    """Process an article abstract using a LLM model.

    This is an example conversation with the LLM model to extract information from an article abstract.

    Parameters
    ----------
    title : str
        The title of the article.

    abstract : str
        The abstract of the article.

    pipe : pipeline
        The ``transformers`` pipeline object to generate responses.

    gen_args : dict
        The generation arguments to pass to the pipeline.

    Returns
    -------
    dict
        A dictionary containing the extracted information from the article abstract.
    """

    # === Truncate abstract to 300 words to avoid CUDA memory issues ===
    abstract = " ".join([_ for _ in abstract.replace("\t", "").split(" ") if _][-300:])
    gen_args["max_new_tokens"] = 24

    # === Define conversation utility functions ===
    def SYSTEM(message):
        return {"role": "system", "content": message}

    def USER(*question):
        return {"role": "user", "content": "\n".join(question)}

    def AGENT(answer):
        return {"role": "agent", "content": answer}

    def simplify_response(response, valid_responses):
        r = response[0]["generated_text"].lower()
        for valid in valid_responses + ["other"]:
            if valid.lower() in r:
                return valid
        return r.strip('". ')

    # === Define conversation messages ===
    # Check the image type used in the paper
    IMG_TYPE = [
        "color fundus",
        "OCT or OCTA",
        "angiography or angiograms",
        "slit lamp",
        "ultrasound",
        "3D images",
        "MR",
        "CT",
    ]

    def ASK_IMAGE_TYPE(name):
        return USER(f"Which type of images is mostly used in {name}?", *IMG_TYPE, "other")

    # Check the paper's main contribution
    CONTRIBS = [
        "review other research papers",
        "novel segmentation method",
        "novel artery and vein classification method",
        "novel diagnosis method",
    ]

    def ASK_CONTRIBUTION(name):
        return USER(f"Which sentence best describes the contribution of {name}?", *CONTRIBS, "other")

    # Check segmented biological structure
    STRUCTURES = [
        "optic disc",
        "macula or fovea",
        "retinal vessels",
        "arteries and veins",
        "lesions",
        "microaneurysms or hemorrhages",
    ]

    def ASK_STRUCTURE(name):
        return USER(f"Which structure is segmented by the method proposed in {name}?", *STRUCTURES, "other")

    # === General Prompt ===
    messages = [
        SYSTEM(
            "You are an accurate AI assistant answering Multiple-Choices questions on research papers. "
            "Your answers are always one of the provided choices."
        ),
        SYSTEM("Here are some examples of conversations."),
        # ---
        # Exemple 1
        USER(
            "EXAMPLE PAPER 1: Fast and robust retinal biometric key generation using deep neural nets.",
            """For biometric identification with retina, vascular structure based features bear significance in preparing retinal digital templates. It needs a faster and robust automated system to extract the quantitative measures from huge amount of retinal images. Therefore, fast and accurate detection of existing retinal features is key for biometric identification. In this work, we propose a retinal biometric key generation framework with deep neural network. The purpose is to replace the semi-automated or automated retinal vascular feature identification methods. The approach begins with segmentation from coloured fundus images, followed by selection of some unique features like center of optic disc, macula center and distinct bifurcation points on a convolutional neural network model. This network was trained and tested with the training and testing images of DRIVE dataset and some of our previously published result sets on automated feature extraction methods.""",  # noqa: E501
        ),
        USER("Which language is mostly used in the paper?", "English", "Chinese", "German", "other"),
        AGENT("English"),
        ASK_IMAGE_TYPE("EXAMPLE PAPER 1"),
        AGENT("color fundus"),
        ASK_CONTRIBUTION("EXAMPLE PAPER 1"),
        AGENT("novel segmentation method"),
        ASK_STRUCTURE("EXAMPLE PAPER 1"),
        AGENT("retinal vessels"),
        # ---
        # Exemple 2
        USER(
            "EXAMPLE PAPER 2: Pyramidal Optical Flow Method-Based Lightweight Monocular 3D Vascular Point Cloud Reconstruction",  # noqa: E501
            """We propose a method for reconstructing a 3D point cloud of the organ model based on optical flow and take the 3D cardiovascular model reconstruction as an example. Firstly, we remove the noise points and improve the resolution. Secondly, we implement the Shi-Tomasi method to extract the feature points. Thirdly, we remove the redundancy in the feature point set by the optical flow distributions. Finally, we converted the obtained feature points from 2D to 3D through the optical flow distribution and then reconstructed a 3D point cloud of the medical organ.""",  # noqa: E501
        ),
        ASK_IMAGE_TYPE("EXAMPLE PAPER 2"),
        AGENT("other"),
        # ---
        # Exemple 3
        USER(
            "EXAMPLE PAPER 3: Multithreshold Image Segmentation Technique Using Remora Optimization Algorithm for Diabetic Retinopathy Detection from Fundus Images.",  # noqa: E501
            """One of the most common complications of diabetes mellitus is diabetic retinopathy (DR), which produces lesions on the retina. A novel framework for DR detection and classification was proposed in this study. Initially, the image pre-processing is performed, the MTRO algorithm performs the vessel segmentation. The feature extraction and classification process are done by a R-CNN. Finally, the proposed R-CNN with WGA effectively classifies the different stages of DR. The experimental images were collected from the DRIVE database, and the proposed framework exhibited superior DR detection performance. Compared to other existing methods, the proposed R-CNN with WGA provided 95.42% accuracy, 93.10% specificity, 93.20% sensitivity, and 98.28% F-score results.""",  # noqa: E501
        ),
        ASK_IMAGE_TYPE("EXAMPLE PAPER 3"),
        AGENT("color fundus"),
        ASK_CONTRIBUTION("EXAMPLE PAPER 3"),
        AGENT("novel diagnosis method"),
        # ---
        # Exemple 4
        USER(
            "EXAMPLE PAPER 4: Automated retinal vessel type classification in color fundus images.",
            """Automated retinal vessel type classification is an essential first step toward machine-based quantitative measurement of various vessel topological parameters and identifying vessel abnormalities and alternations in cardiovascular disease risk analysis. This paper presents a new and accurate automatic artery and vein classification method developed for arteriolar-to-venular width ratio (AVR) and artery and vein tortuosity measurements. This method includes illumination normalization, automatic optic disc detection and retinal vessel segmentation, feature extraction, and a partial least squares (PLS) classification. We trained the algorithm on a set of 51 color fundus images using manually marked arteries and veins. We obtained an area under the ROC curve (AUC) of 93.7% in the ROI of AVR measurement and 91.5% of AUC in the ROI of tortuosity measurement. The proposed AV classification method has the potential to assist automatic cardiovascular disease early detection and risk analysis.""",  # noqa: E501
        ),
        ASK_IMAGE_TYPE("EXAMPLE PAPER 4"),
        AGENT("color fundus"),
        ASK_CONTRIBUTION("EXAMPLE PAPER 4"),
        AGENT("novel artery and vein classification method"),
        SYSTEM("---"),
        SYSTEM("You are now ready to answer questions about the abstracts of research papers."),
        # ---
        # User's article
        USER("USER PAPER:" + title + ".", abstract),
    ]

    # === Process user's article ===
    out = {
        "language": None,
        "image": None,
        "contribution": None,
        "structure": None,
        "fundus_vessel_segmentation": False,
        "fundus_vessel_classification": False,
    }

    with torch.no_grad():
        # Check language
        m0 = messages.copy() + [
            USER("Which language is mostly used in the USER PAPER?", "English", "Chinese", "German", "other")
        ]
        r = pipe(m0, **gen_args)
        out["language"] = simplify_response(r, ["English", "Chinese", "German", "other"])
        if "English" not in out["language"]:
            return out

        # Check image type
        m1 = messages.copy() + [ASK_IMAGE_TYPE("USER PAPER")]
        r = pipe(m1, **gen_args)
        out["image"] = simplify_response(r, IMG_TYPE)
        if "color fundus" not in out["image"]:
            return out

        # Check contribution
        m2 = messages.copy() + [ASK_CONTRIBUTION("USER PAPER")]
        r = pipe(m2, **gen_args)
        out["contribution"] = simplify_response(r, CONTRIBS)
        if "artery and vein" in out["contribution"]:
            out["fundus_vessel_classification"] = True
            return out
        elif "segmentation" not in out["contribution"]:
            return out

        # Check structure
        m3 = messages.copy() + [ASK_STRUCTURE("USER PAPER")]
        r = pipe(m3, **gen_args)
        out["structure"] = simplify_response(r, STRUCTURES)
        if "vessels" in out["structure"]:
            out["fundus_vessel_segmentation"] = True
        return out


def create_llm_pipeline() -> Tuple[pipeline, Dict]:
    """Create a LLM pipeline to process article abstracts.

    Returns
    -------
    pipeline
        The pipeline object to process article abstracts.

    generation_args : dict
        The generation arguments to pass to the pipeline.
    """
    torch.random.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    generation_args = {
        "return_full_text": False,
        # "temperature": 0.0,
        "do_sample": False,
    }
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    return pipe, generation_args


if __name__ == "__main__":
    # === Script Arguments ===
    import argparse

    arg = argparse.ArgumentParser(prog="filter_abstracts", description="Filter abstracts from Zotero library using LLM")
    arg.add_argument("-O", "--output_csv", required=True, type=str, help="Output CSV file")
    arg.add_argument(
        "-Z",
        "--zotero_library",
        required=True,
        type=str,
        help="Zotero library name as [Library]:[Collection]. (Collection is optional)",
    )

    args = arg.parse_args()
    output_csv = args.output_csv
    zotero_library = args.zotero_library.split(":")
    zlib = ZoteroLibrary(*zotero_library)

    analyse_abstracts(zlib, output_csv)
