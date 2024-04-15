from argparse import ArgumentParser
import os
import json



def load_jsonl(in_path, subdir):
    data = []
    fname = os.path.join(in_path, subdir, 'test.jsonl')
    with open(fname, 'r') as f:
        for line in f:
            if line.strip() != '':
                data.append(json.loads(line))
    return data


def load_ner_types(in_path, subdir):
    fname = os.path.join(in_path, subdir, 'entity_types.txt')
    with open(fname, 'r') as f:
        return [each.strip() for each in f.read().split('\n') if each.strip() != '']


class Evaluator:
    def __init__(self, in_path):
        self.eval_data = load_jsonl(in_path, 'ref')
        self.pred_data = load_jsonl(in_path, 'res')
        self.ner_types = load_ner_types(in_path, 'ref')
        self.num_types = len(self.ner_types)
        self.ner_maps = {ner: (i + 1) for i, ner in enumerate(self.ner_types)}


    def validate(self):
        for pred_example in self.pred_data:
            for i, (s, e, t) in enumerate(pred_example['entities'], 1):
                if t not in self.ner_maps:
                    raise ValueError('{t} is unknown type name from line {i}'.format(t=t, i=i))
        if len(self.eval_data) != len(self.pred_data):
            raise ValueError('number of lines in ground truth is not equal to number of lines in solution passed: ' + 
                             '{} != {}'.format(len(self.eval_data), len(self.pred_data)))
        eval_data_ids = set([each['id'] for each in self.eval_data])
        pred_data_ids = set([each['id'] for each in self.pred_data])
        if eval_data_ids != pred_data_ids:
            raise ValueError(
                "ids don't match. Difference: {}".format(eval_data_ids.symmetric_difference(pred_data_ids))
            )

    def evaluate(self):
        tp, fn, fp = 0, 0, 0
        sub_tp, sub_fn, sub_fp = [0] * self.num_types, [0] * self.num_types, [0] * self.num_types
        id_to_ind = {each['id']: ind for ind, each in enumerate(self.pred_data)}
        for gold_example in self.eval_data:
            pred_example = self.pred_data[id_to_ind[gold_example['id']]]
            try:
                gold_ners = set([(s, e, self.ner_maps[t]) for s, e, t in gold_example['entities']])
            except:
                TypeError
            pred_ners = set([(s, e, self.ner_maps[t]) for s, e, t in pred_example['entities']])
            tp += len(gold_ners & pred_ners)
            fn += len(gold_ners - pred_ners)
            fp += len(pred_ners - gold_ners)
            for i in range(self.num_types):
                sub_gm = set((s, e) for s, e, t in gold_ners if t == i+1)
                sub_pm = set((s, e) for s, e, t in pred_ners if t == i+1)
                sub_tp[i] += len(sub_gm & sub_pm)
                sub_fn[i] += len(sub_gm - sub_pm)
                sub_fp[i] += len(sub_pm - sub_gm)
        m_r = 0 if tp == 0 else float(tp) / (tp+fn)
        m_p = 0 if tp == 0 else float(tp) / (tp+fp)
        m_f1 = 0 if m_p == 0 else 2.0*m_r*m_p / (m_r+m_p)
        print("Mention F1: {:.2f}%".format(m_f1 * 100))
        print("Mention recall: {:.2f}%".format(m_r * 100))
        print("Mention precision: {:.2f}%".format(m_p * 100))
        f1_scores_list = []
        counts_list = []
        for i in range(self.num_types):
            sub_r = 0. if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fn[i])
            sub_p = 0. if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fp[i])
            sub_f1 = 0. if sub_p == 0 else 2.0 * sub_r * sub_p / (sub_r + sub_p)
            f1_scores_list.append(sub_f1)
            counts_list.append(sub_tp[i] + sub_fn[i])
        summary_dict = {}
        summary_dict["Mention F1"] = m_f1
        summary_dict["Mention recall"] = m_r
        summary_dict["Mention precision"] = m_p
        filtered_f1s = [
            f1
            for i, (f1, count) in enumerate(zip(f1_scores_list, counts_list))
            if count > 0
        ]
        summary_dict["Mention F1"] = m_f1
        summary_dict["Mention recall"] = m_r
        summary_dict["Mention precision"] = m_p
        filtered_f1s = [
            f1
            for i, (f1, count) in enumerate(zip(f1_scores_list, counts_list))
            if count > 0
        ]

        summary_dict["Macro F1"] = sum(filtered_f1s) / float(len(filtered_f1s))
        print("Macro F1: {:.2f}%".format(summary_dict["Macro F1"] * 100))

        return summary_dict, summary_dict["Macro F1"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = Evaluator(args.in_path)
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    evaluator.validate()
    _, f1 = evaluator.evaluate()
    with open(os.path.join(args.out_path, 'scores.txt'), 'w') as f:
        f.write('f1_score: {:.5f}\n'.format(f1))



if __name__ == '__main__':
    main()
