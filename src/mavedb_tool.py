"""Download and annotate MaveDB DMS data."""


import requests
import re
import numpy as np


AMINO_ACID_SINGLE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}


def download_mavedb_score(urn, directory):
    """Download and save MaveDB scoreset data from given urn ID to a specific folder.

    Parameters
    ----------
    urn: str
        The scoreset urn ID to be downloaded.
    directory: str
        Directory to save downloaded MaveDB data.
    """
    print(f"Downloading: {urn} from MaveDB.")
    download = requests.get(f"https://www.mavedb.org/scoreset/{urn}/scores/")
    if download.status_code != 200:  # If not able to download.
        raise Exception(f"Error, failed to download {urn} data!")
    with open(f"{directory}{urn}.csv", "w+") as file:
        file.write(download.content.decode("utf-8"))
    return


def clean_mavedb_scores(mavedb_data):
    """Remove rows with NaN score value and average scores for duplicate amino acid variants.

    Parameters
    ----------
    mavedb_data: pd.DataFrame
        Contains at least the score and hgvs_pro columns of a certain MaveDB scoreset.

    Return
    ------
    output_data: pd.DataFrame
        Basically the same as input with but hgvs_pro values are no more duplicated.
    """
    output_data = mavedb_data[mavedb_data["score"].notna()].copy()
    output_data = output_data.groupby("hgvs_pro", as_index=False).mean(
        numeric_only=True
    )
    return output_data


def _is_single_hgvs_pro_variant(hgvs_pro):
    """Check if the input variant is single amino acid variant.

    If the variant: i) is a deletion-insertion (e.g.: p.Cys28delinsTrpVal), ii) contains multiple
    amino acid substitutions (e.g.: p.[Cys2Ala;Asp5Glu]), iii) is a frameshift (e.g.: p.Ter337Ala,
    p.Asn234Thrfs), or iv) a stop-loss (e.g.: p.Met1?), it is not regarded as a single amino acid
    variant.

    Parameters
    ----------
    hgvs_pro: str
        The HGVS annotation for checked protein variant.

    Return
    ------
    is_single: bool
        If the variant is a single amino acid variant rather than deletion-insertion, multi-variant,
        frameshift or stop-loss.
    """
    is_delins = re.search("delins", hgvs_pro) is not None  # E.g.: p.Cys28delinsTrpVal
    is_multi = hgvs_pro[-1] == "]"  # E.g.: p.[Cys2Ala;Asp5Glu]
    is_fs = (hgvs_pro[2:5] == "Ter") or (
        re.search("fs", hgvs_pro) is not None
    )  # E.g.: p.Ter337Ala, p.Asn234Thrfs
    is_stoploss = hgvs_pro[-1] == "?"  # E.g.: p.Met1?
    is_single = not (is_delins or is_multi or is_fs or is_stoploss)
    return is_single


def _annotate_singe_hgvs_pro_variant(hgvs_pro):
    """Annotate wild-type, mutation type amino acids, position and variant type for input single amino acid variant.

    Parameters
    ----------
    hgvs_pro: str
        The HGVS annotation for ipnut single amino acid variant.

    Return
    ------
    aa1: str or np.nan
        It shows the wild-type amino acid if the variant is a missense variant.
    aa2: str or np.nan
        It shows the mutation type amino acid if the variant is a missense variant.
    position: int or np.nan
        It shows the location of substitution if the variant is a missense variant.
    mut_type: str
        It shows the type of this variant: nonsense, deletion, synonymous or missense.
    """
    aa1 = AMINO_ACID_SINGLE.get(hgvs_pro[2:5], "np.nan")
    # Nonsense variant.
    if (hgvs_pro[-1] == "*") or (hgvs_pro[-3:] == "Ter"):
        # We do not record details for nonsesne variant here.
        aa1 = "np.nan"
        aa2 = "np.nan"
        position = "np.nan"
        mut_type = "nonsense"
    # Deletion variant.
    elif hgvs_pro[-3:] == "del":
        # We do not record details for nonsesne variant here.
        aa1 = "np.nan"
        aa2 = "np.nan"
        position = "np.nan"
        mut_type = "deletion"
    # Synonymous variant.
    elif (hgvs_pro[-1] == "=") or (hgvs_pro in ["p.(=)", "_wt", "_sy"]):
        # We do not record details for synonymous variant here.
        aa1 = "np.nan"
        aa2 = "np.nan"
        position = "np.nan"
        mut_type = "synonymous"
    # Missense or synonymous (e.g.: p.Met1Met) variant.
    else:
        aa2 = AMINO_ACID_SINGLE[hgvs_pro[-3:]]
        if aa1 == aa2:  # E.g.: p.Met1Met
            aa1 = "np.nan"
            aa2 = "np.nan"
            position = "np.nan"
            mut_type = "synonymous"
        else:
            position = int(hgvs_pro[5:-3])
            mut_type = "missense"
    return aa1, aa2, position, mut_type


def annotate_mavedb_sav_data(input_data):
    """Annotate single amino acid variant for input MaveDB data.

    Parameters
    ----------
    input_data: pd.DataFrame
        Contains at least the hgvs_pro columns of a certain MaveDB scoreset.

    Return
    ------
    output_data: pd.DataFrame
        Contains extra columns of: position, aa1, aa2 and mut_type for annotated single amino acid
        variants.
    """
    output_data = input_data.copy()
    aa1_list = []
    aa2_list = []
    pos_list = []
    type_list = []
    for variant in output_data["hgvs_pro"]:
        # Check if it is single amino acid variant.
        if _is_single_hgvs_pro_variant(variant):
            aa1, aa2, position, mut_type = _annotate_singe_hgvs_pro_variant(variant)
            aa1_list.append(aa1)
            aa2_list.append(aa2)
            pos_list.append(position)
            type_list.append(mut_type)
        else:
            aa1_list.append("np.nan")
            aa2_list.append("np.nan")
            pos_list.append("np.nan")
            type_list.append("other")
    output_data["position"] = pos_list
    output_data["aa1"] = aa1_list
    output_data["aa2"] = aa2_list
    output_data["mut_type"] = type_list
    return output_data


def get_dms_wiltype_like_score(input_data):
    """Retrieve DMS score for wildtype-like variants from variants annotated MaveDB scoreset.

    The function retrieves DMS score for wildtype-like variants from annotated synonymous variants. It will
    try to return the score of variant noted as p.=, _wt, p.(=) and _sy (the first one in order). Otherwise,
    mean score for synoymous variants on each codon is returned. If no synonymous variant is available, np.nan
    will be returned.

    Parameters
    ----------
    input_data: pd.DataFrame
        Contains at least the hgvs_pro, score and mut_type columns of a varaints annotated MaveDB scoreset.

    Return
    ------
    wt_score: float or int
        Retrieved DMS score for wildtype-like variants.
    """
    syn_data = input_data.query("mut_type=='synonymous'")
    syn_hgvs = syn_data["hgvs_pro"].unique()
    if len(syn_data) == 0:
        return np.nan
    for syn_label in ["p.=", "_wt", "p.(=)", "_sy"]:  # Ordered by priority.
        if syn_label in syn_hgvs:
            # Should be one row or the same.
            wt_score = syn_data.query("hgvs_pro == @syn_label").iloc[0]["score"]
            return wt_score
    # Otherwise, synonymous variants are measured on each codon.
    wt_score = syn_data["score"].median()
    return wt_score
