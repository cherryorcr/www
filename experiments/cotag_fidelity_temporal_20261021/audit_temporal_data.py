"""Time and graph-availability statistics independent of model outcomes."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
import torch

from temporal_protocol import arrival_stages, BOUNDS

SOURCE = Path('/home/pc/research/TAGSAM/data/ogbn_arxiv_full_minilm.pt')
OUTPUT = Path('/home/pc/research/TAG_condensation/results/cotag_fidelity_temporal_20261021/temporal_data_audit.json')


def main():
    data = torch.load(SOURCE, map_location='cpu', weights_only=False)
    year = torch.as_tensor(data['node_year']).flatten()
    arrival = arrival_stages(year)
    original_ids = data['original_node_ids'].long()
    pool = ~torch.isin(original_ids%10, torch.tensor([1, 2, 3]))
    names = list(map(str, data['all_labels']))
    mapping = {name: index for index, name in enumerate(names)}
    y = torch.tensor([mapping[str(label)] for label in data['label_list']])
    edges = data['edge_index'].long()
    ep = arrival[edges]
    pool_edge = pool[edges[0]] & pool[edges[1]]
    rows, previous = [], None
    for stage in range(5):
        current = arrival == stage
        counts = torch.bincount(y[current & pool], minlength=len(names)).numpy()
        p = counts/counts.sum()
        current_edges = pool_edge & (ep == stage).all(0)
        cross_edges = pool_edge & (ep <= stage).all(0) & (ep == stage).any(0) & (ep < stage).any(0)
        available_incident = int(current_edges.sum()+cross_edges.sum())
        rows.append(dict(stage=stage, year_end=BOUNDS[stage], nodes=int(current.sum()),
            construction_nodes=int((current & pool).sum()), classes_with_training_support=int((counts > 0).sum()),
            class_counts=counts.tolist(), stored_directed_current_pool_edges=int(current_edges.sum()),
            stored_directed_cross_history_pool_edges=int(cross_edges.sum()),
            fraction_current_incident_edges_lost_by_delta_only=(float(cross_edges.sum())/available_incident if available_incident else 0.),
            js_divergence_from_previous=None if previous is None else float(jensenshannon(previous, p, base=2)**2)))
        previous = p
    output = dict(source=str(SOURCE), feature_protocol=data.get('feature_protocol'),
        total_nodes=len(year), total_stored_directed_edges=edges.shape[1],
        class_names=names, source_year_field='node_year', stage_boundaries=BOUNDS, stages=rows,
        edge_assumption='observed final edges restricted to endpoints available by the stage',
        label_split='original id modulo 1/3/2 for selector-dev/report-dev/historical-test; others train',
        use='data characterization only, independent of downstream score selection')
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k not in ('class_names', 'stages')}, indent=2))
    print(json.dumps([{k:v for k,v in row.items() if k != 'class_counts'} for row in rows], indent=2))


if __name__ == '__main__':
    main()
