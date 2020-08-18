import json
from tqdm import tqdm


def ground(semmed_vocab_path, hf_mapped_path, output_path):
    with open(semmed_vocab_path, "r", encoding="utf-8") as fin:
        semmed_vocab = [w.strip().split("	")[0] for w in fin]
    with open(hf_mapped_path, "r", encoding="utf-8") as fin1, open(output_path, "w", encoding="utf-8") as fout:
        lines = [line for line in fin1]
        for line in tqdm(lines, total=len(lines)):
            j = json.loads(line)
            outj = {}
            record_cui = j["medical_records"]["record_cui"]
            records = []
            for visit in record_cui[:]:
                for cui in visit[:]:
                    if cui not in semmed_vocab:
                        visit.remove(cui)
                if len(visit) != 0:
                    records.append(visit)

            if len(records) != 0:
                j["medical_records"]["record_cui"] = records
                j["medical_records"]["record_cui_list"] = [i for li in records for i in li]
                outj["record_cui"] = [i for li in records for i in li]
                hf_cui_list = j["heart_diseases"]["hf_cui"]
                for hf_cui in hf_cui_list[:]:
                    if hf_cui not in semmed_vocab:
                        hf_cui_list.remove(hf_cui)
                j["heart_diseases"]["hf_cui"] = hf_cui_list
                outj["hf_cui"] = hf_cui_list
                fout.write(json.dumps(j) + "\n")


    print(f'grounded cui saved to {output_path}')
    print()



if __name__ == "__main__":
    ground("../data/semmed/entity2id.txt", "../data/hfdata/converted/dev.jsonl", "../data/hfdata/grounded/dev_ground.jsonl")
    ground("../data/semmed/entity2id.txt", "../data/hfdata/converted/train.jsonl",
           "../data/hfdata/grounded/train_ground.jsonl")
    ground("../data/semmed/entity2id.txt", "../data/hfdata/converted/test.jsonl",
           "../data/hfdata/grounded/test_ground.jsonl")