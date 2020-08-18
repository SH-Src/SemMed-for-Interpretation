import networkx as nx
import json
from tqdm import tqdm

# try:
#     from .utils import check_file
# except ImportError:
#     from utils import check_file

__all__ = ["construct_graph", "relations"]


# (33 pos, 31 neg): no neg_compared_with, neg_prep
relations = ['administered_to', 'affects', 'associated_with', 'augments', 'causes', 'coexists_with', 'compared_with', 'complicates',
             'converts_to', 'diagnoses', 'disrupts', 'higher_than', 'inhibits', 'isa', 'interacts_with', 'location_of', 'lower_than',
             'manifestation_of', 'measurement_of', 'measures', 'method_of', 'occurs_in', 'part_of', 'precedes', 'predisposes', 'prep',
             'prevents', 'process_of', 'produces', 'same_as', 'stimulates', 'treats', 'uses']


def load_merge_relation():
    # TODO: merge relation
    """
    `return`: relation_mapping: {"":}
    """
def separate_semmed_cui(semmed_cui: str) -> list:
    """
    separate semmed cui with | by perserving the replace the numbers after |
    `param`:
        semmed_cui: single or multiple semmed_cui separated by |
    `return`:
        sep_cui_list: list of all separated semmed_cui
    """
    sep_cui_list = []
    sep = semmed_cui.split("|")
    first_cui = sep[0]
    sep_cui_list.append(first_cui)
    ncui = len(sep)
    for i in range(ncui - 1):
        last_digs = sep[i + 1]
        len_digs = len(last_digs)
        if len_digs < 8: # there exists some strange cui with over 7 digs
            sep_cui = first_cui[:8 - len(last_digs)] + last_digs
            sep_cui_list.append(sep_cui)
    return sep_cui_list

def extract_semmed_cui(semmed_csv_path, output_csv_path, semmed_cui_path):
    # TODO: deal with some error cui and its influence on graph constructing
    """
    read the original SemMed csv file to extract all cui and store
    """
    print('extracting cui list from SemMed...')
    semmed_cui_vocab = []
    cui_seen = set()
    nrow = sum(1 for _ in open(semmed_csv_path, "r", encoding="utf-8"))
    with open(semmed_csv_path, "r", encoding="utf-8") as fin, open(output_csv_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split(',')
            if ls == ['']:
                continue
            subj = ls[4]
            obj = ls[8]
            if len(subj) == 8 and len(obj) == 8 and subj.startswith("C") and obj.startswith("C"):
                fout.write(line+"\n")
                for i in [subj, obj]:
                    if i not in cui_seen:
                        semmed_cui_vocab.append(i)
                        cui_seen.add(i)

    with open(semmed_cui_path, "w", encoding="utf-8") as fout:
        for semmed_cui in semmed_cui_vocab:
            fout.write(semmed_cui + "\n")

    print(f'extracted cui saved to {semmed_cui_path}')
    print()

def construct_graph(semmed_csv_path, semmed_cui_path, output_path, prune=True):
    # TODO: 1. prune 2. deal with the case that subj == obj 3. cui with | 4. cui2idx?
    """
    construct the SemMed graph file
    """
    print("generating SemMed graph file...")
    with open(semmed_cui_path, "r", encoding="utf-8") as fin:
        idx2cui = [c.strip() for c in fin]
    cui2idx = {c: i for i, c in enumerate(idx2cui)}

    idx2relation = relations
    relation2idx = {r: i for i, r in enumerate(idx2relation)}

    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(semmed_csv_path, "r", encoding="utf-8"))
    with open(semmed_csv_path, "r", encoding="utf-8") as fin:
        attrs = set()
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split(',')
            if ls == ['']:
                continue
            rel = relation2idx[ls[3].lower()]
            subj = cui2idx[ls[4]]
            obj = cui2idx[ls[8]]
            if subj == obj:
                continue
            if (subj, obj, rel) not in attrs:
                graph.add_edge(subj, obj, rel=rel)
                attrs.add((subj, obj, rel))
                graph.add_edge(obj, subj, rel=rel+len(relation2idx))
                attrs.add((obj, subj, rel+len(relation2idx)))
    nx.write_gpickle(graph, output_path)

    print(f"graph file saved to {output_path}")
    print()

def construct_subgraph(semmed_csv_path, semmed_cui_path, output_graph_path):
    print("generating subgraph of SemMed using newly extracted cui list...")

    with open(semmed_cui_path, "r", encoding="utf-8") as fin:
        idx2cui = [c.strip().split('	')[0].strip() for c in fin]
        print(idx2cui)
    cui2idx = {c: i for i, c in enumerate(idx2cui)}

    idx2relation = relations
    relation2idx = {r: i for i, r in enumerate(idx2relation)}

    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(semmed_csv_path, "r", encoding="utf-8"))

    with open(semmed_csv_path, "r", encoding="utf-8") as fin:
        attrs = set()
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split(',')
            if ls[3].lower() not in relations:
                continue
            if ls[4] not in idx2cui or ls[8] not in idx2cui:
                continue
            if ls[4] == ls[8]: # delete self-loop, not useful for our task
                continue

            sent = ls[1]
            rel = relation2idx[ls[3].lower()]

            if ls[4].startswith("C") and ls[8].startswith("C"):
                if len(ls[4]) == 8 and len(ls[8]) == 8:
                    subj = cui2idx[ls[4]]
                    obj = cui2idx[ls[8]]
                    if (subj, obj, rel) not in attrs:
                        graph.add_edge(subj, obj, rel=rel, sent=sent)
                        attrs.add((subj, obj, rel))
                        graph.add_edge(obj, subj, rel=rel+len(relation2idx), sent=sent)
                        attrs.add((obj, subj, rel+len(relation2idx)))
                elif len(ls[4]) != 8 and len(ls[8]) == 8:
                    cui_list = separate_semmed_cui(ls[4])
                    subj_list = [cui2idx[s] for s in cui_list]
                    obj = cui2idx[ls[8]]
                    for subj in subj_list:
                        if (subj, obj, rel) not in attrs:
                            graph.add_edge(subj, obj, rel=rel, sent=sent)
                            attrs.add((subj, obj, rel))
                            graph.add_edge(obj, subj, rel=rel + len(relation2idx), sent=sent)
                            attrs.add((obj, subj, rel + len(relation2idx)))
                elif len(ls[4]) == 8 and len(ls[8]) != 8:
                    cui_list = separate_semmed_cui(ls[8])
                    obj_list = [cui2idx[o] for o in cui_list]
                    subj = cui2idx[ls[4]]
                    for obj in obj_list:
                        if (subj, obj, rel) not in attrs:
                            graph.add_edge(subj, obj, rel=rel, sent=sent)
                            attrs.add((subj, obj, rel))
                            graph.add_edge(obj, subj, rel=rel + len(relation2idx), sent=sent)
                            attrs.add((obj, subj, rel + len(relation2idx)))
                else:
                    cui_list1 = separate_semmed_cui(ls[4])
                    subj_list = [cui2idx[s] for s in cui_list1]
                    cui_list2 = separate_semmed_cui(ls[8])
                    obj_list = [cui2idx[o] for o in cui_list2]
                    for subj in subj_list:
                        for obj in obj_list:
                            if (subj, obj, rel) not in attrs:
                                graph.add_edge(subj, obj, rel=rel, sent=sent)
                                attrs.add((subj, obj, rel))
                                graph.add_edge(obj, subj, rel=rel + len(relation2idx), sent=sent)
                                attrs.add((obj, subj, rel + len(relation2idx)))
    nx.write_gpickle(graph, output_graph_path)
    print(len(attrs))
    print(f"graph file saved to {output_graph_path}")
    #print(f"txt file saved to {output_txt_path}")
    print()

if __name__ == "__main__":
    #extract_semmed_cui("../data/semmed/database.csv", "../data/semmed/database_pruned.csv", "../data/semmed/cui_vocab.txt")
    construct_subgraph("../data/semmed/database.csv", "../data/semmed/entity2id.txt", "../data/semmed/database.graph")