import networkx as nx
from tqdm import tqdm
import json
import numpy as np
from multiprocessing import Pool
import random
from semmed import relations
# from .semmed import relations

cui2idx = None
idx2cui = None
relation2idx = None
idx2relation = None

semmed = None
semmed_simple = None

cui_embs = None
relation_embs = None


def load_resources(semmed_cui_path):
    global cui2idx, idx2cui, relation2idx, idx2relation
    with open(semmed_cui_path, "r", encoding="utf-8") as fin:
        idx2cui = [c.strip() for c in fin]
    cui2idx = {c: i for i, c in enumerate(idx2cui)}

    idx2relation = relations
    relation2idx = {r: i for i, r in enumerate(idx2relation)}


def load_semmed(semmed_graph_path):
    """
    load the pruned or unpruned SemMed graph file

    `return`:
        semmed: multirelational directed graph of SemMed, weight of each path
        is 1.0
        semmed_simple: basic undirected graph of SemMed, weight of each edge is
        the number of paths
    """
    global semmed, semmed_simple
    semmed = nx.read_gpickle(semmed_graph_path)
    semmed_simple = nx.Graph()
    for u, v, data in semmed.edges(data=True):
        w = 1.0 # initial weight to 1
        if semmed_simple.has_edge(u, v):
            semmed_simple[u][v]['weight'] += w
        else:
            semmed_simple.add_edge(u, v, weight=w)


##################### path finding #####################


def get_edge(src_cui: int, tgt_cui: int) -> list:
    # TODO: deal with hf cui that are not in the semmed source cui but in the semmed cui
    """
    find edges between source cui and target cui

    `param`:
        src_cui: index of the single source cui
        tgt_cui: index of the single target cui
    `return`:
        res: list of all relations with no repetition
    """
    global semmed
    rel_list = semmed[src_cui][tgt_cui]
    rel_seen = set()
    res = [r["rel"] for r in rel_list.values() if r["rel"] not in rel_seen and (rel_seen.add(r["rel"]) or True)]
    return res


def find_paths_between_single_cui_pair(src_cui: int, tgt_cui: int) -> list:
    """
    find paths for a (single_record_cui, single_hf_cui) pair

    `param`:
        src_cui: index of the single source cui
        tgt_cui: index of the single target cui
    `return`:
        pf_res: list of dictionaries with the form of {"path": p, "rel": rl}
    """
    all_path = []
    try:
        for p in nx.shortest_simple_paths(semmed_simple, source=src_cui, target=tgt_cui):
            if len(p) > 5 or len(all_path) >= 100:
                break
            if len(p) >= 2: # skip paths of length 1 (self-loop?)
                all_path.append(p)
    except nx.exception.NetworkXNoPath:
        pass

    pf_res = []
    for p in all_path:
        rl = [] # list of rel_list of all pair in this path
        for src in range(len(p) - 1): # num of pair in the path
            src_cui = p[src]
            tgt_cui = p[src + 1]

            rel_list = get_edge(src_cui, tgt_cui)
            rl.append(rel_list)
            pf_res.append({"path": p, "rel": rl})
    return pf_res


def find_paths_between_pair(pair: list) -> list:
    """
    find paths between a pair of record_cui and hf_cui

    `param`:
        pair: list of a record_cui and hf_cui pair
    `return`:
        pfr_pair: list of dictionaries with the form of
        {"record_cui": record_cui, "hf_cui": hf_cui, "pf_res": pf_res}
    """
    record_cui, hf_cui = pair
    pfr_pair = []
    for single_hf_cui in hf_cui:
        for single_record_cui in record_cui:
            pf_res = find_paths_between_single_cui_pair(single_record_cui, single_hf_cui)
            pfr_pair.append({"record_cui": record_cui, "hf_cui": hf_cui, "pf_res": pf_res})
    return pfr_pair


##################### path scoring #####################

# TODO: no pretrained embedding with TransE

def score_triple(h, t, r, flag):

def score_triples(cui_idx, rel_idx, debug=False):
    global cui_embs, relation_embs, idx2cui, idx2relation
    cui = cui_embs[cui_idx]
    relation = []
    flag = []
    for i in range(len(rel_idx))
    embs = []
    l_flag = []


def score_paths_between_pair(pfr_pair):
    """
    score paths between a pair of record_cui and hf_cui
    """
    # between a pair of cui, there are several paths
    pair_scores = []
    for pf_res in pfr_pair:
        pair_paths = pf_res["pf_res"]
        if pair_paths is not None:
            path_scores = []
            for path in pair_paths


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################

#TODO: add multiprocessing

def find_paths(grounded_path, semmed_cui_path, semmed_graph_path, output_path, num_processes=1, random_state=0):
    print(f'generating paths for {grounded_path}...')
    random.seed(random_state)
    np.random.seed(random_state)

    global cui2idx, idx2cui, relation2idx, idx2relation, semmed, semmed_simple
    if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
        load_resources(semmed_cui_path)
    if any(x is None for x in [semmed, semmed_simple]):
        load_semmed(semmed_graph_path)

    with open(grounded_path, "r") as fin:
        data = [json.loads(line) for line in fin]
    data = [[item["record_cui"], item["hf_cui"]] for item in data]
    nrow = len(data)

    # with Pool(num_processes) as p, open(output_path, "w") as fout:
        # for pfr_pair in tqdm(p.imap(find_paths_between_pair, data), total=nrow):
    with open(output_path, "w") as fout:
        for row in tqdm(range(nrow)):
            pfr_pair = find_paths_between_pair(data[row])
            fout.write(json.dumps(pfr_pair) + "\n")

    print(f'paths saved to {output_path}')
    print()


def score_paths(raw_paths_path, cui_emb_path, rel_emb_path, semmed_cui_path, output_path, num_processes=1, method="triple_cls"):
    print(f"scoring paths for {raw_paths_path}...")
    global cui2idx, idx2cui, relation2idx, idx2relation, cui_embs, relation_embs
    if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
        load_resources(semmed_cui_path)
    if cui_embs is None:
        cui_embs = np.load(cui_emb_path)
    if relation_embs is None:
        relation_embs = np.load(rel_emb_path)

    if method != "triple_cls":
        raise NotImplementedError()

    with open(raw_paths_path, "r") as fin:
        data = [json.loads(line) for line in fin]
    nrow = len(data)

    with open(output_path, "w") as fout:
        for row in tqdm(range(nrow)):
            path_scores = score_paths_between_pair(data[row])
            fout.write(json.dumps(path_scores) + "\n")

    print(f"path scores saved to {output_path}")
    print()


if __name__ == "__main__":
    # find_paths("E:\python\SemMed\data\hfdata\grounded\dev.grounded.jsonl", "E:\python\SemMed\data\semmed\cui_list.txt",
    # "E:\python\SemMed\data\semmed\semmed.unpruned.graph", "E:\python\SemMed\data\hfdata\paths\dev.paths.raw.jsonl")