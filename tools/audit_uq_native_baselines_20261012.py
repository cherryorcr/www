"""Numerical checks for pinned baseline cores, independent of result quality."""
import argparse
import json

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph

from uq_high_fidelity_baselines_20261012 import native_fuzzy_centers
from uq_tagsam_native_selector_20261012 import native_language_model, native_subset, predecessor_csr, pmi_matrix, greedy_groups


def run(language_model=False, device="cpu"):
    rng = np.random.RandomState(41)
    values = rng.normal(size=(36, 6)).astype(np.float32)
    values[:18] += 3
    _, fuzzy = native_fuzzy_centers(values, 3, 71, restarts=3, max_iters=40)
    edges = torch.tensor([[0, 1, 2, 2, 3, 5, 6, 7, 8, 9], [1, 2, 0, 3, 4, 4, 5, 6, 6, 9]])
    adjacency = predecessor_csr(edges, 11)
    for center in range(11):
        for hop in range(1, 5):
            # The fixture contains node id 10, so pass the explicit graph size
            # instead of letting PyG infer it from the largest source column.
            subset = k_hop_subgraph(torch.tensor([center]), hop, edges,
                                    num_nodes=11)[0]
            if len(subset) >= 4 or hop == 4:
                break
        assert np.array_equal(native_subset(center, adjacency), subset[:4].numpy())
    matrix = np.array([[0., 2., 1., .5], [1., 0., 4., .3], [.4, 2., 0., 2.], [1., 1., 1., 0.]])
    selected = greedy_groups(matrix)
    assert len(selected) == 3 and len(set(selected)) == 3
    result = dict(passed=True, fuzzy_native_center_error=fuzzy["native_center_error"],
                  fuzzy_membership_error=fuzzy["row_sum_error"], directed_neighborhood_cases=11)
    if language_model:
        model, native = native_language_model(device)
        sentences = ["Graph models learn useful features.", "Text provides semantic information.",
                     "We compress the graph and text.", "Memory supports continual learning."]
        # batch=1 in upstream avoids its missing final partial-batch defect.
        expected = native["get_npmi_matrix"](model, sentences, batch_size=1)[1]
        actual, diagnostic = pmi_matrix(model, sentences, native["get_probabilities"], batch=5)
        error = float(np.max(np.abs(expected - actual)))
        assert error < 1e-3, (error, expected, actual)
        assert greedy_groups(expected) == greedy_groups(actual)
        result.update(pmi_max_error=error, pmi_selected_groups=greedy_groups(actual),
                      repaired_tail_pairs=diagnostic["tail_batch_pairs"],
                      native_word_probabilities=True, gpt_device=device)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-model", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(run(args.language_model, args.device), indent=2))
