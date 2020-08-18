import os
import argparse
from multiprocessing import cpu_count
from utils.convert_hfdata import convert_to_cui
# from utils.tokenization_utils import tokenize_medical_records, make_cui_list
# from utils.semmed import extract_semmed_cui, construct_graph
# from utils.embedding import glove2npy, load_pretrained_embeddings
#from utils.grounding import ground
#from utils.paths import find_paths
# , score_paths, prune_paths, find_relational_paths_from_paths, generate_path_and_graph_from_adj
from utils.graph import generate_adj_data_from_grounded_concepts
# from utils.triples import generate_triples_from_adj

input_paths = {
    'hfdata': {
        'train': './data/hfdata/hf_training_new.pickle',
        'dev': './data/hfdata/hf_validation_new.pickle',
        'test': './data/hfdata/hf_testing_new.pickle',
        'code2idx': './data/hfdata/hf_code2idx_new.pickle',
    },
    'semmed': {
        'csv': './data/semmed/database.csv',
    },
    'snomed': {
        'snomedct': './data/snomed/SNOMEDCT.txt',
        'icd2snomed_1to1': './data/snomed/ICD9CM_SNOMED_MAP_1TO1.txt',
        'icd2snomed_1toM': './data/snomed/ICD9CM_SNOMED_MAP_1TOM.txt',
    },
    'glove': {
        'txt': './data/glove/glove.6B.50d.txt',
    },
    'numberbatch': {
        'txt': './data/transe/numberbatch-en.txt',
    },
    'transe': {
        'ent': './data/transe/glove.transe.sgd.ent.npy',
        'rel': './data/transe/glove.transe.sgd.rel.npy',
    },
}

output_paths = {
    'semmed': {
        'vocab': './data/semmed/entity2id.txt',
        'unpruned-graph': './data/semmed/semmed.unpruned.graph',
        'pruned-graph': './data/semmed/database.graph',
    },
    'glove': {
        'npy': './data/glove/glove.6B.50d.npy',
        'vocab': './data/glove/glove.vocab',
    },
    'numberbatch': {
        'npy': './data/transe/nb.npy',
        'vocab': './data/transe/nb.vocab',
        'concept_npy': './data/transe/concept.nb.npy'
    },
    'hfdata': {
        'converted': {
            'train': './data/hfdata/converted/train.jsonl',
            'dev': './data/hfdata/converted/dev.jsonl',
            'test': './data/hfdata/converted/test.jsonl',
            'vocab': './data/hfdata/converted/vocab.json', # haven't change to cui-list (hfdata) don't know where it will be used
        },
        'tokenized': {
            'train': './data/hfdata/tokenized/train.tokenized.txt',
            'dev': './data/hfdata/tokenized/dev.tokenized.txt',
            'test': './data/hfdata/tokenized/test.tokenized.txt',
        },
        'grounded': {
            'train': './data/hfdata/grounded/train_ground.jsonl',
            'dev': './data/hfdata/grounded/dev_ground.jsonl',
            'test': './data/hfdata/grounded/test_ground.jsonl',
        },
        'paths': {
            'raw-train': './data/hfdata/paths/train.paths.raw.jsonl',
            'raw-dev': './data/hfdata/paths/dev.paths.raw.jsonl',
            'raw-test': './data/hfdata/paths/test.paths.raw.jsonl',
            'scores-train': './data/hfdata/paths/train.paths.scores.jsonl',
            'scores-dev': './data/hfdata/paths/dev.paths.scores.jsonl',
            'scores-test': './data/hfdata/paths/test.paths.scores.jsonl',
            'pruned-train': './data/hfdata/paths/train.paths.pruned.jsonl',
            'pruned-dev': './data/hfdata/paths/dev.paths.pruned.jsonl',
            'pruned-test': './data/hfdata/paths/test.paths.pruned.jsonl',
            'adj-train': './data/hfdata/paths/train.paths.adj.jsonl',
            'adj-dev': './data/hfdata/paths/dev.paths.adj.jsonl',
            'adj-test': './data/hfdata/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': './data/hfdata/graph/train.graph.jsonl',
            'dev': './data/hfdata/graph/dev.graph.jsonl',
            'test': './data/hfdata/graph/test.graph.jsonl',
            'adj-train': './data/hfdata/graph/train.graph.adj.pk',
            'adj-dev': './data/hfdata/graph/dev.graph.adj.pk',
            'adj-test': './data/hfdata/graph/test.graph.adj.pk',
            'nxg-from-adj-train': './data/hfdata/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': './data/hfdata/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': './data/hfdata/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': './data/hfdata/triples/train.triples.pk',
            'dev': './data/hfdata/triples/dev.triples.pk',
            'test': './data/hfdata/triples/test.triples.pk',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common', 'hfdata'], choices=['common', 'hfdata',  'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            # {'func': glove2npy, 'args': (input_paths['glove']['txt'], output_paths['glove']['npy'], output_paths['glove']['vocab'])},
            # {'func': glove2npy, 'args': (input_paths['numberbatch']['txt'], output_paths['numberbatch']['npy'], output_paths['numberbatch']['vocab'], True)},
            # {'func': extract_semmed_cui, 'args': (input_paths['semmed']['csv'], output_paths['semmed']['cui-list'])},
            # {'func': load_pretrained_embeddings,
            #  'args': (output_paths['numberbatch']['npy'], output_paths['numberbatch']['vocab'], output_paths['semmed']['vocab'], False, output_paths['numberbatch']['concept_npy'])},
            # {'func': construct_graph, 'args': (input_paths['semmed']['csv'], output_paths['semmed']['cui-list'], output_paths['semmed']['unpruned-graph'], False)},
            # {'func': construct_graph, 'args': (output_paths['semmed']['csv'], output_paths['semmed']['vocab'],
                                            #    output_paths['semmed']['pruned-graph'], True)},
        ],
        'hfdata': [
            # {'func': convert_to_cui, 'args': (input_paths['hfdata']['train'], output_paths['hfdata']['converted']['train'], input_paths['hfdata']['code2idx'],
            #                                   input_paths['snomed']['snomedct'], input_paths['snomed']['icd2snomed_1to1'], input_paths['snomed']['icd2snomed_1toM'])},
            # {'func': convert_to_cui, 'args': (input_paths['hfdata']['dev'], output_paths['hfdata']['converted']['dev'], input_paths['hfdata']['code2idx'],
            #                                   input_paths['snomed']['snomedct'], input_paths['snomed']['icd2snomed_1to1'], input_paths['snomed']['icd2snomed_1toM'])},
            # {'func': convert_to_cui, 'args': (input_paths['hfdata']['test'], output_paths['hfdata']['converted']['test'], input_paths['hfdata']['code2idx'],
            #                                   input_paths['snomed']['snomedct'], input_paths['snomed']['icd2snomed_1to1'], input_paths['snomed']['icd2snomed_1toM'])},
            # {'func': tokenize_medical_records, 'args': (output_paths['hfdata']['converted']['train'], output_paths['hfdata']['tokenized']['train'])},
            # {'func': tokenize_medical_records, 'args': (output_paths['hfdata']['converted']['dev'], output_paths['hfdata']['tokenized']['dev'])},
            # {'func': tokenize_medical_records, 'args': (output_paths['hfdata']['converted']['test'], output_paths['hfdata']['tokenized']['test'])},
            # {'func': make_word_vocab, 'args': (output_paths['hfdata']['statement']['train'], output_paths['hfdata']['statement']['vocab'])},
            # {'func': ground, 'args': (output_paths['hfdata']['converted']['train'], output_paths['semmed']['cui-list'], output_paths['hfdata']['grounded']['train'])},
            # {'func': ground, 'args': (output_paths['hfdata']['converted']['dev'], output_paths['semmed']['cui-list'], output_paths['hfdata']['grounded']['dev'])},
            # {'func': ground, 'args': (output_paths['hfdata']['converted']['test'], output_paths['semmed']['cui-list'], output_paths['hfdata']['grounded']['test'])},
            # {'func': find_paths, 'args': (output_paths['hfdata']['grounded']['train'], output_paths['semmed']['cui-list'],
            #                               output_paths['semmed']['pruned-graph'], output_paths['hfdata']['paths']['raw-train'], args.nprocs, args.seed)},
            #{'func': find_paths, 'args': (output_paths['hfdata']['grounded']['dev'], output_paths['semmed']['cui-list'],
            #                              output_paths['semmed']['unpruned-graph'], output_paths['hfdata']['paths']['raw-dev'], args.nprocs, args.seed)},
            # {'func': find_paths, 'args': (output_paths['hfdata']['grounded']['test'], output_paths['semmed']['cui-list'],
            #                               output_paths['semmed']['pruned-graph'], output_paths['hfdata']['paths']['raw-test'], args.nprocs, args.seed)},
            # {'func': score_paths, 'args': (output_paths['hfdata']['paths']['raw-train'], input_paths['transe']['ent'], input_paths['transe']['rel'],
            #                                output_paths['semmed']['vocab'], output_paths['hfdata']['paths']['scores-train'], args.nprocs)},
            # {'func': score_paths, 'args': (output_paths['hfdata']['paths']['raw-dev'], input_paths['transe']['ent'], input_paths['transe']['rel'],
            #                                output_paths['semmed']['vocab'], output_paths['hfdata']['paths']['scores-dev'], args.nprocs)},
            # {'func': score_paths, 'args': (output_paths['hfdata']['paths']['raw-test'], input_paths['transe']['ent'], input_paths['transe']['rel'],
            #                                output_paths['semmed']['vocab'], output_paths['hfdata']['paths']['scores-test'], args.nprocs)},
            # {'func': prune_paths, 'args': (output_paths['hfdata']['paths']['raw-train'], output_paths['hfdata']['paths']['scores-train'],
            #                                output_paths['hfdata']['paths']['pruned-train'], args.path_prune_threshold)},
            # {'func': prune_paths, 'args': (output_paths['hfdata']['paths']['raw-dev'], output_paths['hfdata']['paths']['scores-dev'],
            #                                output_paths['hfdata']['paths']['pruned-dev'], args.path_prune_threshold)},
            # {'func': prune_paths, 'args': (output_paths['hfdata']['paths']['raw-test'], output_paths['hfdata']['paths']['scores-test'],
            #                                output_paths['hfdata']['paths']['pruned-test'], args.path_prune_threshold)},
            # {'func': generate_graph, 'args': (output_paths['hfdata']['grounded']['train'], output_paths['hfdata']['paths']['pruned-train'],
            #                                   output_paths['semmed']['vocab'], output_paths['semmed']['pruned-graph'],
            #                                   output_paths['hfdata']['graph']['train'])},
            # {'func': generate_graph, 'args': (output_paths['hfdata']['grounded']['dev'], output_paths['hfdata']['paths']['pruned-dev'],
            #                                   output_paths['semmed']['vocab'], output_paths['semmed']['pruned-graph'],
            #                                   output_paths['hfdata']['graph']['dev'])},
            # {'func': generate_graph, 'args': (output_paths['hfdata']['grounded']['test'], output_paths['hfdata']['paths']['pruned-test'],
            #                                   output_paths['semmed']['vocab'], output_paths['semmed']['pruned-graph'],
            #                                   output_paths['hfdata']['graph']['test'])},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['hfdata']['grounded']['train'], output_paths['semmed']['pruned-graph'],
                                                                        output_paths['semmed']['vocab'], output_paths['hfdata']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['hfdata']['grounded']['dev'], output_paths['semmed']['pruned-graph'],
                                                                        output_paths['semmed']['vocab'], output_paths['hfdata']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['hfdata']['grounded']['test'], output_paths['semmed']['pruned-graph'],
                                                                        output_paths['semmed']['vocab'], output_paths['hfdata']['graph']['adj-test'], args.nprocs)},
            # {'func': generate_triples_from_adj, 'args': (output_paths['hfdata']['graph']['adj-train'], output_paths['hfdata']['grounded']['train'],
            #                                              output_paths['semmed']['vocab'], output_paths['hfdata']['triple']['train'])},
            # {'func': generate_triples_from_adj, 'args': (output_paths['hfdata']['graph']['adj-dev'], output_paths['hfdata']['grounded']['dev'],
            #                                              output_paths['semmed']['vocab'], output_paths['hfdata']['triple']['dev'])},
            # {'func': generate_triples_from_adj, 'args': (output_paths['hfdata']['graph']['adj-test'], output_paths['hfdata']['grounded']['test'],
            #                                              output_paths['semmed']['vocab'], output_paths['hfdata']['triple']['test'])},
            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['hfdata']['graph']['adj-train'], output_paths['semmed']['pruned-graph'], output_paths['hfdata']['paths']['adj-train'], output_paths['hfdata']['graph']['nxg-from-adj-train'], args.nprocs)},
            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['hfdata']['graph']['adj-dev'], output_paths['semmed']['pruned-graph'], output_paths['hfdata']['paths']['adj-dev'], output_paths['hfdata']['graph']['nxg-from-adj-dev'], args.nprocs)},
            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['hfdata']['graph']['adj-test'], output_paths['semmed']['pruned-graph'], output_paths['hfdata']['paths']['adj-test'], output_paths['hfdata']['graph']['nxg-from-adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
