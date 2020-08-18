import numpy as np
from scipy import spatial
import networkx as nx
from tqdm import tqdm
import pickle
import json
import random
import os
from semmed import relations
__all__ = ['find_paths']
concept2id = None
id2concept = None
relation2id = None
id2relation = None

semmed = None
semmed_simple = None

def load_cui_vocab(cui_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cui_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}
    id2relation = relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_semmed(semmed_graph_path):
    global semmed, semmed_simple
    semmed = nx.read_gpickle(semmed_graph_path)
    semmed_simple = nx.Graph()
    for u, v, data in semmed.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if semmed_simple.has_edge(u, v):
            semmed_simple[u][v]['weight'] += w
        else:
            semmed_simple.add_edge(u, v, weight=w)

def get_edge(src_concept, tgt_concept):
    global semmed
    rel_list = semmed[src_concept][tgt_concept]  # list of dicts
    seen = set()
    res = [r['rel'] for r in rel_list.values() if r['rel'] not in seen and (seen.add(r['rel']) or True)]  # get unique values from rel_list
    return res


def find_paths_qa_concept_pair(source: str, target: str):
    global semmed, semmed_simple, concept2id, id2concept, relation2id, id2relation
    s = concept2id[source]
    t = concept2id[target]

    if s not in semmed_simple.nodes() or t not in semmed_simple.nodes():
        return
    all_path = []
    try:
        for p in nx.shortest_simple_paths(semmed_simple, source=s, target=t):
            if len(p) > 5 or len(all_path) >= 100:  # top 100 paths
                break
            if len(p) >= 2:  # skip paths of length 1
                all_path.append(p)
    except nx.exception.NetworkXNoPath:
        pass

    pf_res = []
    for p in all_path:
        # print([id2concept[i] for i in p])
        rl = []
        for src in range(len(p) - 1):
            src_concept = p[src]
            tgt_concept = p[src + 1]

            rel_list = get_edge(src_concept, tgt_concept)
            rl.append(rel_list)

        pf_res.append({"path": p, "rel": rl})
    return pf_res


def find_paths_qa_pair(qa_pair):
    acs, qcs = qa_pair
    pfr_qa = []
    for ac in acs:
        for qc in qcs:
            pf_res = find_paths_qa_concept_pair(qc, ac)
            pfr_qa.append({"hf_cui": ac, "record_cui_list": qc, "pf_res": pf_res})
    return pfr_qa


def find_paths(grounded_path, cui_vocab_path, semmed_graph_path, output_path):

    print(f'generating paths for {grounded_path}...')
    global concept2id, id2concept, relation2id, id2relation, semmed_simple, semmed
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_cui_vocab(cui_vocab_path)
    if semmed is None or semmed_simple is None:
        load_semmed(semmed_graph_path)

    with open(grounded_path, 'r') as fin:
        data = [json.loads(line) for line in fin]
    data = [[item["heart_diseases"]["hf_cui"], item["medical_records"]["record_cui_list"]] for item in data]
    with open(output_path, 'w') as fout:
        for i in tqdm(range(0, len(data))):
            pfr_qa = find_paths_qa_pair(data[i])
            fout.write(json.dumps(pfr_qa) + '\n')

    print(f'paths saved to {output_path}')
    print()


if __name__ == "__main__":
    find_paths("../data/hfdata/grounded/dev_ground.jsonl", "../data/semmed/cui_vocab.txt", "../data/semmed/database_pruned.graph", "../data/hfdata/paths/dev_paths.jsonl")
