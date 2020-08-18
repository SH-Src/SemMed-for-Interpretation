"""
Script to ground cui of medical records to that of SemMed
"""
from tqdm import tqdm
import sys
import json

SEMMED_CUI = None


def load_semmed_cui(semmed_cui_path):
    # TODO: deal with | in cui
    with open(semmed_cui_path, "r", encoding="utf-8") as fin:
        semmed_cui = [l.strip() for l in fin]
    semmed_cui = [s.replace("_", "") for s in semmed_cui]
    return semmed_cui


def combine_visit_cui(medical_record_cui):
    """
    merge visits of cui into one list and delete repetition
    if nothing left, print cui not found?
    """
    combined_record_cui = []
    for visit_cui in medical_record_cui:
        combined_record_cui += visit_cui
    combined_record = list(set(combined_record_cui))
    # if combined_record == []:
    #     print("no cui is found")
    return combined_record_cui


def prune(data: list, semmed_cui_path: str) -> list:
    # TODO: add multiprocessing
    """
    prune all record cui that do not exist in SemMed cui
    `param`:
        data: list of dictionaries with the form of {"record_cui": , "hf_cui": }
        semmed_cui_path: path of semmed cui
    `return`:
        prune_data: list of dictionaries with the form of {"record_cui": , "hf_cui": }
    """
    with open(semmed_cui_path, "r", encoding="utf8") as fin:
        semmed_cui = [l.strip() for l in fin]

    prune_data = []
    for item in tqdm(data, desc="grounding"):
        prune_record_cui = []
        record_cui = item["record_cui"]
        for cui in record_cui:
            if cui in semmed_cui:
                prune_record_cui.append(cui)
        prune_hf_cui = []
        hf_cui = item["hf_cui"]
        for cui in hf_cui:
            if cui in semmed_cui:
                prune_hf_cui.append(cui)

        item["record_cui"] = prune_record_cui
        item["hf_cui"] = prune_hf_cui
        prune_data.append(item)
    return prune_data


def ground(medical_record_path, semmed_cui_path, output_path, debug=False):
    global SEMMED_CUI
    if SEMMED_CUI is None:
        SEMMED_CUI = load_semmed_cui(semmed_cui_path)

    total_cui_list = []
    with open(medical_record_path, "r") as fin:
        lines = [line for line in fin]

    if debug:
        lines = lines[0:2]

    for line in lines:
        # if line == "":
        #     continue
        total_cui = {}
        j = json.loads(line)
        medical_record = j["medical_records"]
        heart_disease = j["heart_diseases"]

        combined_record_cui = combine_visit_cui(medical_record["record_cui"])
        total_cui["record_cui"] = combined_record_cui
        total_cui["hf_cui"] = heart_disease["hf_cui"]
        total_cui_list.append(total_cui)
    grounded_cui = prune(total_cui_list, semmed_cui_path)

    with open(output_path, "w") as fout:
        for cui in grounded_cui:
            fout.write(json.dumps(cui) + '\n')

    print(f'grounded cui saved to {output_path}')
    print()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise ValueError("Provide three arguments")
    ground((sys.argv[1]), (sys.argv[2]), (sys.argv[3]))