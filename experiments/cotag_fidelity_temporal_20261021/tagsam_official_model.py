import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, adjs):
        edge_index, _, size = adjs[0]
        x = self.conv1(x, edge_index)[:size[1]]
        x = F.leaky_relu(x)

        edge_index, _, size = adjs[1]
        x = self.conv2(x, edge_index)[:size[1]]
        return x


class GraphEncoder(nn.Module):
    def __init__(self, args):
        super(GraphEncoder, self).__init__()
        self.args = args

        if args.graph_encoder == 'gcn':
            self.model = GCN(args.gnn_input_dim, args.gnn_hidden_dim, args.gnn_output_dim)
        else:
            raise ValueError('Invalid graph encoder')

    def forward(self, x, adjs):
        return self.model(x, adjs)


class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.args = args

        if args.text_encoder == "bert":
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.target_token_idx = 0
        else:
            raise ValueError('Invalid text encoder')

        self.requires_grad_(False)

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.args.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, self.target_token_idx, :]


class TextProjection(nn.Module):
    def __init__(self, args):
        super(TextProjection, self).__init__()
        self.args = args
        dropout_rate = 0.1

        self.projection = nn.Linear(args.lm_output_dim, args.gnn_output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(args.gnn_output_dim, args.gnn_output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(args.gnn_output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIP(nn.Module):
    def __init__(self, args):
        super(CLIP, self).__init__()
        self.args = args

        self.text_encoder = TextProjection(args)
        self.graph_encoder = GraphEncoder(args)

    def forward(self, node_f, adjs, texts, is_eval=False):
        text_emb = self.encode_text(texts)
        graph_emb = self.encode_graph(node_f, adjs)

        logits = np.exp(np.log(1 / 0.07)) * graph_emb @ text_emb.t()

        if is_eval:
            return logits
        else:
            ground_truth = torch.arange(len(logits)).type_as(logits).long()
            loss = (F.cross_entropy(logits, ground_truth) + F.cross_entropy(logits.t(), ground_truth)) / 2
            return loss, logits

    def encode_graph(self, node_f, edge_index):
        graph_emb = self.graph_encoder(node_f, edge_index)
        graph_emb = graph_emb / (graph_emb.norm(dim=1, keepdim=True) + 1e-10)
        return graph_emb

    def encode_text(self, texts):
        text_emb = self.text_encoder(texts)
        text_emb = text_emb / (text_emb.norm(dim=1, keepdim=True) + 1e-10)
        return text_emb


class wBCELoss(nn.Module):
    def __init__(self):
        super(wBCELoss, self).__init__()

    def forward(self, logits, gt_matrix):
        gt_matrix = gt_matrix.to(logits.device)
        probs1 = torch.sigmoid(logits)
        probs2 = torch.sigmoid(gt_matrix)

        loss_matrix = - probs2 * torch.log(probs1 + 1e-6) - (1 - probs2) * torch.log(1 - probs1 + 1e-6)

        pos_mask = (probs2 > 0.5).detach()
        neg_mask = ~pos_mask

        loss_pos = torch.where(pos_mask, loss_matrix, torch.tensor(0.0, device=probs1.device)).sum()
        loss_neg = torch.where(neg_mask, loss_matrix, torch.tensor(0.0, device=probs1.device)).sum()

        loss_pos /= (pos_mask.sum() + 1e-6)
        loss_neg /= (neg_mask.sum() + 1e-6)

        return (loss_pos + loss_neg) / 2


class GPT2:
    def __init__(self, device="cpu", location=""):
        if location == "":
            self.enc = GPT2Tokenizer.from_pretrained("gpt2-large")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        else:
            self.enc = GPT2Tokenizer.from_pretrained(location)
            self.model = GPT2LMHeadModel.from_pretrained(location)
        self.device = torch.device(device)
        self.model.eval()
        self.start_tok = "<|endoftext|>"
        self.model.to(self.device)

    def pad(self, context, max_length=1024):
        for i in range(len(context)):
            if len(context[i]) > max_length:
                context[i] = context[i][:max_length]
        max_len = max([len(sentence) for sentence in context])
        # print("Maximum Length: ", max_len)
        for i in range(len(context)):
            # print(len(context[i]), max_len - len(context[i]))
            for j in range(max_len - len(context[i])):
                context[i].append(context[i][0])

        return context

    def get_probabilities(self, in_text, topk=40):
        with torch.no_grad():
            context = [self.start_tok + " " + in_text[i] for i in range(len(in_text))]
            context = [self.enc.encode(context[i]) for i in range(len(context))]
            context = self.pad(context)
            context = torch.tensor(context, device=self.device, dtype=torch.long)
            output = self.model(context)
            logits = output.logits
            yhat = torch.softmax(logits[:, :-1], dim=-1)
            y = context[:, 1:]
            real_topk_probs = [yhat[t][np.arange(0, y[t].shape[0], 1), y[t]].data.cpu().numpy().tolist() for t in
                               range(yhat.shape[0])]
            real_topk_probs = [list(map(lambda x: round(x, 15), real_topk_probs[t])) for t in
                               range(len(real_topk_probs))]

            real_topk = [list(real_topk_probs[t]) for t in range(len(real_topk_probs))]

            context_strings = [[self.enc.decoder[s.item()] for s in context[t]] for t in range(len(context))]
            context_strings = [[self.postprocess(s) for s in context_strings[t]] for t in range(len(context_strings))]
            del context, logits, y, yhat,
            torch.cuda.empty_cache()
        """ 
        pred_topk = [[list(zip([self.enc.decoder[p] for p in sorted_preds[t][i][:topk]],
            list(map(lambda x: round(x, 5),yhat[t][i][sorted_preds[t][i][
                :topk]].data.cpu().numpy().tolist()))))
                    for i in range(y[t].shape[0])] for t in range(y.shape[0])]
        pred_topk = [[[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk[t]] for t in range(len(pred_topk))]
        """
        payload = {'context_strings': context_strings,
                   'real_probs': real_topk}  # , 'pred_topk': pred_topk}

        # del context, logits, y, yhat,
        # torch.cuda.empty_cache()
        # code.interact(local=locals())
        return payload

    def postprocess(self, token):
        with_space = False
        with_break = False
        # print(token, token[0], token[1:]),
        if token[0] == 'Ġ':
            with_space = True
            token = token[1:]
        elif token.startswith('â'):
            token = ' '
        elif token.startswith('Ċ'):
            token = ' '
            with_break = True

        if len(token) > 0 and token[0] == "Â":
            token = token[1:]
        token = '-' if token.startswith('â') else token
        token = '“' if token.startswith('ľ') else token
        token = '”' if token.startswith('Ŀ') else token
        token = "'" if token.startswith('Ļ') else token
        # if with_space:
        #    token = '\u0120' + token
        # if with_break:
        #    token = '\u010A' + token
        # print(token)
        return token