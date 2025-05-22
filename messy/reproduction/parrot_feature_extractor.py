import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertModel
import torch
from queue import Queue
from transformers import BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertModel
import torch
from collections import defaultdict
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm
from networks.transformers import TransformerDecoderLayer, TransformerDecoder, TokenEmbedding, PositionalEncoding

BOS, EOS, PAD, MASK = '[BOS]', '[EOS]', '[PAD]', '[MASK]'

class ParrotFeatureExtractor(BertForSequenceClassification):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        num_decoder_layers = config.num_decoder_layers
        nhead = config.nhead
        tgt_vocab_size = config.tgt_vocab_size
        dim_feedforward = config.dim_feedforward
        dropout = config.dropout
        d_model = config.d_model
        device = None
        dtype = None

        if hasattr(config, 'output_attention'):
            self.output_attention = config.output_attention
        else:
            self.output_attention = False

        self.bert = BertModel(config)
        activation = F.relu
        layer_norm_eps = 1e-5
        factory_kwargs = {'device': device, 'dtype': dtype}

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first=True,
            norm_first=False,
            **factory_kwargs,
            output_attention=self.output_attention)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size=d_model)
        self.positional_encoding = PositionalEncoding(emb_size=d_model, dropout=dropout)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            output_attention=self.output_attention)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self,inputs, beam = [1,3,1,5,1]):
        start_symbol = self.config.condition_label_mapping[1][BOS]
        step2translate = defaultdict(list)
        succ_translate = []
        translate_quene = Queue()
        max_len = 6

        memory_key_padding_mask = (inputs["attention_mask"]==0)
        memory = self.encode(inputs)

        ys = torch.ones(memory.size(0),1).fill_(start_symbol).type(torch.long).cuda().transpose(0, 1)
        cumul_score = torch.ones(memory.size(0)).type(torch.float).cuda().view(-1, 1).transpose(0, 1)
        step_number = 0
        translate_quene.put((ys, cumul_score, step_number))
        while (not translate_quene.empty()):
            ys, cumul_score, step_number = translate_quene.get()
            if ys.size(0) >= max_len:
                succ_translate.append((ys, cumul_score, step_number))
                continue
            ys = ys.transpose(0, 1)
            cumul_score = cumul_score.transpose(0, 1)
            tgt_mask = (torch.triu(torch.ones((ys.size(1), ys.size(1)), device=self.device)) == 0).transpose(0, 1)
            out, _ = self.decode(ys,memory,tgt_mask,memory_key_padding_mask=memory_key_padding_mask)
            pred = self.generator(out[:, -1])
            prob = torch.softmax(pred, dim=1)
            next_scores, next_words = prob.topk(beam[step_number])
            step_number += 1
            for i in range(next_words.size(1)):
                _ys = torch.cat([ys, next_words[:, i].unsqueeze(1)], dim=1)

                _cumul_score = cumul_score * next_scores[:, i].unsqueeze(1)
                step2translate[step_number].append((_ys, _cumul_score, step_number))
            thread_number = 1
            for i in range(step_number):
                thread_number *= beam[i]
            if len(step2translate[step_number]) == thread_number:
                put_list = step2translate[step_number]
                _ys_cat = torch.cat([x[0].unsqueeze(0) for x in put_list],dim=0)
                _ys_cat = _ys_cat.transpose(1, 2)
                _cumul_score_cat = torch.cat([x[1] for x in put_list],dim=1)
                _cumul_score_cat = _cumul_score_cat.transpose(0, 1)
                _ys_cat_sorted = torch.zeros_like(_ys_cat)
                _cumul_score_cat_sorted = torch.zeros_like(_cumul_score_cat)
                for j in range(_cumul_score_cat.size(1)):
                    dim_cumul_score_sorted, _idx = _cumul_score_cat[:, j].topk(thread_number)
                    _ys_cat_sorted[:, :, j] = _ys_cat[_idx, :, j]
                    _cumul_score_cat_sorted[:, j] = dim_cumul_score_sorted

                for n in range(thread_number):
                    translate_quene.put((_ys_cat_sorted[n],_cumul_score_cat_sorted[n].unsqueeze(0),step_number))
        _tgt_tokens = torch.cat([x[0].unsqueeze(0) for x in succ_translate],dim=0)
        _cumul_scores = torch.cat([x[1] for x in succ_translate])
        tgt_tokens = torch.zeros_like(_tgt_tokens)
        cumul_scores = torch.zeros_like(_cumul_scores)
        for j in range(_cumul_scores.size(1)):
            dim_cumul_scores_sorted, _idx = _cumul_score_cat[:, j].topk(thread_number)
            tgt_tokens[:, :, j] = _ys_cat[_idx, :, j]
            cumul_scores[:, j] = dim_cumul_scores_sorted
        tgt_tokens = tgt_tokens[:, 1:]
        tgt_tokens = tgt_tokens.permute(2,0,1)
        return tgt_tokens, cumul_scores

    def encode(self, inputs):
        return self.bert(**inputs)[0]

    def decode(self, tgt, memory, tgt_mask, memory_key_padding_mask):
        decoder_output, attention_weightes = self.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        return decoder_output, attention_weightes