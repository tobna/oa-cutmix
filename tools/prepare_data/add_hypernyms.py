#!/usr/bin/env python3
"""
add_hypernyms.py
----------------
Read an ImageNet class‑name file (e.g. `imagenet_classnames.txt`) and create a
CSV file where each line is extended with the immediate hypernym of the class
name according to WordNet.

Output format (comma‑separated):
    index,wnid,classname,hypernym
If no hypernym can be found, “N/A” is used.

Usage
-----
    python add_hypernyms.py -i imagenet_classnames.txt -o imagenet_classnames_with_hypernyms.csv
"""

import argparse
import sys
import re
from tqdm.auto import tqdm

# --------------------------------------------------------------
# NLTK / WordNet setup
# --------------------------------------------------------------
try:
    from nltk.corpus import wordnet as wn
    import nltk
except ImportError as e:
    sys.stderr.write("NLTK is required. Install it via `pip install nltk`.\n")
    raise e


# Ensure the WordNet corpus is available; download if needed
def ensure_wordnet():
    try:
        wn.ensure_loaded()
    except Exception:
        # This will trigger a download of the 'wordnet' resource
        nltk.download("wordnet")
        wn.ensure_loaded()


ensure_wordnet()


# --------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------
def parse_line(line: str):
    """Parse a line like "1: n01440764, tench".
    Returns (index, wnid, classname) or (None, None, None) on failure.
    """
    # Accept lines with or without the leading index
    m = re.match(r"(n\d+), (.*)$", line)
    if not m:
        return None, None
    wnid, classname = m.groups()
    # idx may be None if not present; keep as empty string
    return wnid, classname


def get_immediate_hypernym(wnid: str) -> str:
    """Return the immediate hypernym lemma for the given WordNet ID (wnid).
    The wnid is of the form 'n01440764' which corresponds to a noun synset.
    Returns "N/A" if the synset cannot be found or has no hypernym.
    """
    try:
        # wnid format: 'n' + 8‑digit offset
        offset = int(wnid[1:])
        syn = wn.synset_from_pos_and_offset("n", offset)
    except Exception:
        return "N/A"
    classname = syn.lemma_names()[0].replace("_", " ")
    hypernyms = syn.hypernyms()
    if not hypernyms:
        return "N/A"
    # Take the first hypernym and its first lemma name
    lemma = hypernyms[0].lemma_names()[0]
    return classname, lemma.replace("_", " ")


# --------------------------------------------------------------
# Main processing
# --------------------------------------------------------------
def process_file(in_path: str, out_path: str):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for raw_line in tqdm(fin):
            raw_line = raw_line.rstrip("\n")
            wnid, classname = parse_line(raw_line)
            if wnid is None:
                # Write the original line unchanged (or skip)
                continue
            classnames, hypernym = get_immediate_hypernym(wnid)
            # CSV line
            fout.write(f"{wnid}; {classnames}; {hypernym}\n")


# --------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Add WordNet immediate hypernyms to ImageNet class names and output CSV."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the original imagenet_classnames.txt")
    parser.add_argument(
        "-o",
        "--output",
        default="imagenet_classnames_with_hypernyms.csv",
        help="Path for the output CSV file (default: %(default)s)",
    )
    args = parser.parse_args()
    process_file(args.input, args.output)
    print(f"✅ Done – output written to: {args.output}")


if __name__ == "__main__":
    main()
