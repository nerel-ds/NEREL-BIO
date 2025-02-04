import itertools
import logging
import os
import random
from argparse import ArgumentParser
from typing import List, Tuple, Dict

import pandas as pd
from matplotlib import pyplot as plt

from bionnel.utils.entity import Entity
from bionnel.utils.nestedness_utils import entities_groupby_nested, create_nestedness_lists
import networkx as nx


def create_nestedness_nodes_edges(nested_entities: List[List[Entity]]):
    graphs: List[Dict] = []
    for entities in nested_entities:
        node_names: List[str] = [e.entity_str for e in entities]
        edges: List[Tuple[int, int]] = []
        for i, j in itertools.combinations(range(len(entities)), 2):
            ent_1 = entities[i]
            ent_2 = entities[j]
            s1, s2 = ent_1.entity_str, ent_2.entity_str
            if s1 == s2:
                continue
            if s1 in s2:
                edges.append((i, j))
            if s2 in s1:
                edges.append((j, i))
        graphs.append(
            {"nodes": node_names,
             "edges": edges
             }
        )
    return graphs


def split_node_labels(n_label, max_line_length=16, min_line_length=4):
    label_length = len(n_label)
    accum_length = 0
    lines = []
    words = n_label.split()
    curr_s = f"{words[0]}"
    for w in words[1:]:
        curr_len = len(curr_s)
        w_len = len(w)
        if curr_len + w_len > max_line_length and label_length - accum_length > min_line_length:
            lines.append(curr_s)
            curr_s = w
            accum_length += len(curr_s)
        else:
            curr_s += f" {w}"
    if len(curr_s) > 0:
        lines.append(curr_s)

    return '\n'.join(lines)


def draw_graph(graph_dict, save_path):
    node_labels = graph_dict["nodes"]
    edges = graph_dict["edges"]
    nx_graph = nx.from_edgelist(edges)
    node_labels = [split_node_labels(x, max_line_length=10, min_line_length=4) for x in node_labels]
    node_labels = {i: x for i, x in enumerate(node_labels)}

    try:
        pos = nx.planar_layout(nx_graph, scale=0.05)
    except Exception as e:
        pos = nx.spring_layout(nx_graph, scale=0.1)
    # plt.figure(figsize=(4.5, 4.5))
    plt.figure(figsize=(5, 5))
    nx.draw(nx_graph, pos=pos, node_size=250, alpha=0.8, font_size=20,
            font_weight='bold')

    pos_node_labels = {}
    # print("pos.values()", tuple(pos.values()))

    y_off = 0.075  # offset on the y axis
    max_pos = max(v[1] for v in pos.values())
    min_pos = min(v[1] for v in pos.values())
    # print(max_pos, min_pos)
    delta_pos = max_pos - min_pos

    for k, v in pos.items():
        offset = y_off * delta_pos

        pos_node_labels[k] = (v[0], v[1] - offset)
    nx.draw_networkx_labels(nx_graph, pos_node_labels, node_labels, font_size=8.5)

    x0, x1 = plt.xlim()
    y0, y1 = plt.ylim()
    plt.xlim(x0 * 1.5, x1 * 1.5)
    plt.ylim(y0 * 1.1, y1 * 1.1)

    plt.savefig(save_path, format="PDF")
    plt.clf()


def main(args):
    input_tsv = args.input_tsv
    graph_count = args.graph_count
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    df = pd.read_csv(input_tsv, sep='\t')
    document_id2entities = entities_groupby_nested(df=df)
    nested_entities: List[List[Entity]] = create_nestedness_lists(document_id2entities=document_id2entities)

    graphs = create_nestedness_nodes_edges(nested_entities=nested_entities)
    graphs = list(filter(lambda d: len(d["edges"]) > 0, graphs))
    # print(graphs)
    graph_dicts = random.sample(graphs, k=graph_count)

    for g_dict in graph_dicts:
        node_names = g_dict["nodes"]
        longest_s = max(node_names, key=lambda x: len(x))
        longest_s = longest_s.replace(' ', '_')
        save_path = os.path.join(output_dir, f"{longest_s}.pdf")

        draw_graph(g_dict, save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()

    parser.add_argument('--input_tsv', type=str,
                        default="/home/c204/University/NLP/PROJECTS/bionne_2025_organizers/bionne-l_tsv/ru/bionnel_ru_dev.tsv")
    parser.add_argument('--graph_count', type=int, default=5)
    parser.add_argument('--output_dir', type=str,
                        default="./graphs_debug/")

    arguments = parser.parse_args()
    main(arguments)
