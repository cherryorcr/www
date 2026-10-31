import math
import os

import numpy as np
import spacy
import torch
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

from model import TextEncoder, GPT2


def get_probabilities(model, articles):
    article_splits = [article.split(" ") for article in articles]
    payload = model.get_probabilities(articles, topk=20)
    res = [[] for i in range(len(articles))]
    for t, article in enumerate(articles):
        context = ""
        idx = 0
        chain = False
        next_word = ""
        article_words = article_splits[t]
        # print(article, article_words)
        word_probability = 1.0
        gt_count = 0
        idx += 1
        found_words = []
        for i, word in enumerate(payload["context_strings"][t][:-1]):
            context = context + " " + word
            probability = payload['real_probs'][t][i]  # [1]
            next_word_fragment = payload["context_strings"][t][i + 1]

            next_word += next_word_fragment
            # print(next_word, article_words[gt_count])
            if next_word == article_words[gt_count]:
                chain = False
                gt_count += 1
            else:
                chain = True

            word_probability *= probability
            assert word_probability <= 1.0, print(word_probability, context)
            if chain == False:
                # print("Word Probability: ", word_probability, next_word)
                res[t].append(word_probability)
                word_probability = 1.0
                next_word = ""
            # print(gt_count, len(article_words))
            if gt_count == len(article_words):
                break
    return res


def get_npmi_matrix(model, sentences, method=1, batch_size=1):
    temp = np.zeros((len(sentences), len(sentences)))
    temp2 = np.zeros((len(sentences), len(sentences)))
    batch_indices = {}
    batch = []
    batchCount = 0
    batchSize = batch_size
    c = 0
    p = []
    for i in range(len(sentences)):
        result = get_probabilities(model, [sentences[i]])
        try:
            p.append(sum([math.log(i) for i in result[0]]))
        except:
            print("Math domain error surprise", i)
            return temp, temp2, p
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                temp[i][j] = -1
                temp2[i][j] = -1
                continue
            article = sentences[i] + " " + sentences[j]
            batch_indices[str(i) + "-" + str(j) + "-" + str(len(sentences[i].split()))] = batchCount
            batch.append(article)
            batchCount += 1

            if batchCount == batchSize or (i == len(sentences) - 1 and j == len(sentences) - 1):
                c += 1
                result = get_probabilities(model, batch)
                for key in batch_indices.keys():
                    idx_i, idx_j, idx_l = [int(idx) for idx in key.split("-")]
                    try:
                        pxy = sum([math.log(q) for q in result[batch_indices[key]][idx_l:]])
                        py = p[idx_j]
                        px = p[idx_i]

                        temp[idx_i][idx_j] = (pxy - py) / (-1 * (pxy + px))
                        temp2[idx_i][idx_j] = (pxy - py)
                    except ZeroDivisionError:
                        # rint("Zero division error ", idx_i, idx_j)
                        temp[idx_i][idx_j] = -1
                        temp2[idx_i][idx_j] = -1
                    except:
                        print("Math Domain Error", i, j)
                batchCount = 0
                batch = []
                batch_indices = {}
    return temp, temp2, p


def remove_unicode(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


@torch.no_grad()
def select_text(dataset, args):
    select_idx = np.random.permutation(len(dataset))[:args.syn_size]
    cache_file = os.path.join(str(args.buffer_save_dir), args.name, f"syn_data_0.pt")

    if os.path.exists(cache_file):
        cache_data = torch.load(cache_file)
        node_embeds = cache_data['node_f']
        text_embeds = cache_data['text_embeds']
        selected_text = cache_data['selected_text']
        return node_embeds, text_embeds, selected_text

    selected_text = []
    feature_aggregate =[]
    text_list = dataset.text_list
    nlp = spacy.load("en_core_web_sm")
    model = GPT2(device=args.device)
    ratio = float(args.syn_ratio_summary) / 100

    for center_node in tqdm(select_idx):
        center_node = torch.tensor([center_node])
        hop = 0

        subset = []
        while len(subset) < args.syn_num_summary:
            hop += 1
            subset, _, _, _ = k_hop_subgraph(
                center_node, num_hops=hop, edge_index=dataset.edge_index, relabel_nodes=False
            )
            if hop > 3:
                break
        if len(subset) > args.syn_num_summary:
            subset = subset[:args.syn_num_summary]
        subset = np.array(subset.tolist())

        sub_texts = [text_list[i] for i in subset]

        num_sentences = 0
        sentences_list = []
        for sub_text in sub_texts:
            doc = nlp(str(sub_text))
            sentences = [remove_unicode(sentence.text) for sentence in doc.sents]
            num_sentences += len(sentences)
            sentences_list.append(sentences)

        sentence_ranges = []
        start_idx = 0
        combine = 5
        for i in range(len(sub_texts)):
            sentences = sentences_list[i]
            merged_sentences = [' '.join(sentences[i:i + combine]) for i in range(0, len(sentences), combine)]

            sentence_ranges.append((start_idx, start_idx + len(merged_sentences)))
            start_idx += len(merged_sentences)

        all_sentences = []
        for i in range(len(sub_texts)):
            sentences = sentences_list[i]
            merged_sentences = [' '.join(sentences[i:i + combine]) for i in range(0, len(sentences), combine)]
            all_sentences.extend(merged_sentences)

        normalised, matrix, surprise = get_npmi_matrix(model, all_sentences, batch_size=5)
        matrix[matrix < 0] = 0

        # n = matrix.shape[0]
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         average = (matrix[i, j] + matrix[j, i]) / 2
        #         matrix[i, j] = average
        #         matrix[j, i] = average

        relevance = [sum(matrix[idx]) for idx in range(len(all_sentences))]
        penalty = [0 for _ in range(len(all_sentences))]
        selected = []

        num_selected_sentences = math.ceil(len(all_sentences) * ratio)
        for k in range(num_selected_sentences):
            maxIdx = -1
            maxVal = -float('inf')

            for i in range(len(all_sentences)):
                temp = 1 * relevance[i] + (-1) * penalty[i]
                if temp > maxVal and i not in selected:
                    maxIdx = i
                    maxVal = temp

            for i in range(len(all_sentences)):
                penalty[i] += matrix[i][maxIdx]

            selected.append(maxIdx)

        summary = " ".join(all_sentences[i] for i in sorted(selected))
        selected_text.append(summary)

        weighted_features = []
        for (start, end), idx in zip(sentence_ranges, subset):
            doc_selected = [i for i in selected if start <= i < end]
            extraction_ratio = len(doc_selected) / (end - start)
            sub_feature = dataset.node_f[idx]
            weighted_features.append(sub_feature * extraction_ratio)

        sub_feature = torch.stack(weighted_features).sum(dim=0)

        feature_aggregate.append(sub_feature)

    del nlp
    del model
    torch.cuda.empty_cache()

    text_encoder = TextEncoder(args).to(args.device)
    text_embeds = []

    batch_size = 128
    for i in range(0, len(selected_text), batch_size):
        batch = selected_text[i:i + batch_size]
        text_embed = text_encoder(batch)
        text_embeds.append(text_embed)

    text_embeds = torch.cat(text_embeds, dim=0)
    node_embeds = torch.stack(feature_aggregate)

    return node_embeds, text_embeds, selected_text
