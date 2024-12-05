
import math
import os
import json
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5TokenizerFast, T5ForConditionalGeneration, BartTokenizerFast, BartForConditionalGeneration
from tokenizers import AddedToken
import config


class SAMode:
    ems = [
        '',
        'SHARE_Q_ENCODER',
        'T5_EMBEDDING',
        'EMPTY_EMBEDDING'
    ]
    tms = [
        '',
        'SA_SA+T',
        'SA_T',
        'SA+T',
        'ITERATE'
    ]
    ums = [
        '',
        'EMBEDDING+ATTENTION',
        'ATTENTION'
    ]
    def __init__(self, mode_str):
        embed_mode, train_mode, update_mode = map(int, mode_str)
        self.use_sa = embed_mode != 0
        self.embed_mode = self.ems[embed_mode]
        self.train_mode = self.tms[train_mode]
        self.update_mode = self.ums[update_mode]
        print(self.use_sa, self.embed_mode, self.train_mode, self.update_mode)
        

class SequentialSchemaAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, max_q_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.d_k = math.sqrt(hidden_dim)
        self.k_embedding = nn.Embedding(73, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1)
        self.wq = nn.Linear(embed_dim, hidden_dim)
        self.wk = nn.Linear(embed_dim, hidden_dim)
        self.attn = nn.Linear(max_q_len, 1)

    def forward(self, question, schema=None):
        # indices = torch.arange(73).cuda()
        # schema = self.k_embedding(indices).unsqueeze(0)
        # question: [batch, question_max_seq_len, emb_dim]
        # schema: [1, schema_len, emb_dim]
        q = self.wq(question)  # [batch, question_max_seq_len, hidden_dim]
        k = self.wk(schema).squeeze(0)  # [schema_len, hidden_dim]
        x = torch.matmul(q, k.T) / self.d_k  # [batch, question_max_seq_len, schema_len]
        x = F.softmax(x, dim=1)
        x = self.attn(x.permute(0, 2, 1))  # [batch, 1, schema_len]
        x = F.sigmoid(x.squeeze(2))  # [batch, schema_len]

        emb = schema.repeat((x.shape[0], 1, 1)) * x.unsqueeze(-1)  # [batch, schema_len, emb_dim]
        
        return emb, x
    



class PLMPlain(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.sa_mode = SAMode(self.opt.sa_mode)
        self.current_training_period = 'NONE'
        if self.opt.eval_model_suffix:
            model_path = f'{self.opt.save_path}/checkpoint-{self.opt.eval_model_suffix}'
        else:
            model_path = self.opt.model_name_or_path
    
        if 't5' in model_path.lower():
            self.tokenizer = T5TokenizerFast.from_pretrained(opt.model_name_or_path, add_prefix_space=True)
        elif 'bart' in model_path.lower():
            self.tokenizer = BartTokenizerFast.from_pretrained(opt.model_name_or_path, add_prefix_space=True)

        aql_tokens = [AddedToken(' ' + s) for s in ['FOR', 'IN', 'INBOUND', 'OUTBOUND', 'FILTER', '==', 'AND', 'CONTAINS', 'COLLECT', 'WITH', 'COUNT', 'INTO', 'SORT', 'DESC', 'ASC', 'LIMIT', 'RETURN', 'LET']]
        mmnat_tokens = [AddedToken(' ' + s) for s in ['SCAN', 'YIELD', 'FILTER', 'AND', 'CONTAINS', '==', 'EXPAND', 'IN', 'OUT', 'EDGE', 'GROUP', 'COUNT', 'AS', '*', 'SORT', 'ASC', 'DESC', 'LIMIT', 'RETURN']]
        myrial_tokens = [AddedToken(' ' + s) for s in ['scan', '=', 'asc', 'desc', 'emit', 'RESULT']]
        ecql_tokens = [AddedToken(' ' + s) for s in ['@>', 'JSONB', 'ARRAY', 'ELEMENTS', 'LENGTH', '{', '}', 'MAX']]
        pir_tokens = [AddedToken(' ' + s) for s in ['COLLECTION', 'TableScan', 'OrderBy']]
        self.tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")] + aql_tokens + mmnat_tokens + myrial_tokens + ecql_tokens + pir_tokens)

        if 't5' in model_path.lower():
            self.plm = T5ForConditionalGeneration.from_pretrained(model_path)
        elif 'bart' in model_path.lower():
            self.plm = BartForConditionalGeneration.from_pretrained(model_path)
        self.plm.resize_token_embeddings(len(self.tokenizer))

        self.aql_schema_filepath = config.SCHEMA_PATH
        self.prepare_schema()

    def prepare_schema(self):
        with open(self.aql_schema_filepath, 'r', encoding='utf-8') as fj:
            indices = json.load(fj)['indices']

        self.n_schema_items = len(indices)  # n == 73
        self.schema_names = [indices[j]['item'] for j in range(self.n_schema_items)]

        tokenized_schema_names = self.tokenizer(
            self.schema_names, 
            padding="max_length",
            return_tensors="pt",
            max_length=8,
            truncation=True
        )  # [n_schema_items, schema_len]

        self.schema_token_ids = tokenized_schema_names['input_ids'].cuda()  # [n_schema_items, schema_len]
        self.schema_padding_mask = tokenized_schema_names['attention_mask'].cuda()  # [n_schema_items, schema_len]

    def parameters_requiring_grad(self):
        if self.sa_mode.use_sa:
            if self.current_training_period == 'SCHEMA':
                params = self.sa.parameters()
            elif self.current_training_period == 'TRANSLATION':
                params = [p for p in self.plm.parameters() if p.requires_grad]
                params += list(self.sa.parameters())
        else:
            params = [p for p in self.plm.parameters() if p.requires_grad]

        return params
    
    def forward_output(self, batch, mode, params):
        question, question_padded_schema, schema_occurrence, schema_occurrence_mask, aql, pir, myrial, am, ecql = batch

        if self.opt.ql == 'AQL':
            output_token_length = 256
            output_ql = aql
        elif self.opt.ql == 'ECQL':
            output_token_length = 256
            output_ql = ecql

        if mode == 'train':
            with self.tokenizer.as_target_tokenizer():
                tokenized_outputs = self.tokenizer(
                    output_ql, 
                    padding='max_length', 
                    return_tensors='pt',
                    max_length=output_token_length,
                    truncation=True
                )
            decoder_labels = tokenized_outputs['input_ids'].cuda()
            decoder_labels[decoder_labels == self.tokenizer.pad_token_id] = -100
            decoder_attention_mask = tokenized_outputs['attention_mask'].cuda()

            params.update({
                'labels': decoder_labels,
                'decoder_attention_mask': decoder_attention_mask,
                'return_dict': True
            })

            model_outputs = self.plm(**params)
            loss = model_outputs['loss']
        else:
            params.update({
                'max_length': output_token_length,
                'decoder_start_token_id': self.plm.config.decoder_start_token_id,
                'num_beams': self.opt.num_beams,
                'num_return_sequences': self.opt.num_return_sequences
            })
            
            model_outputs = self.plm.generate(**params)
            loss = None

        return loss, model_outputs
    
    def preprocess_batch(self, batch):
        question, question_padded_schema, schema_occurrence, schema_occurrence_mask, aql, pir, myrial, am, ecql = batch

        tokenizer_input = question_padded_schema

        return tokenizer_input, 512
    
    def forward_translation_params(self, batch):
        tokenizer_input, token_len = self.preprocess_batch(batch)

        tokenized_question = self.tokenizer(
            tokenizer_input,
            padding='max_length',
            return_tensors='pt',
            max_length=token_len,
            truncation=True
        )
        question_token_ids = tokenized_question['input_ids'].cuda()
        question_attention_mask = tokenized_question['attention_mask'].cuda()

        params = {
            'input_ids': question_token_ids,
            'attention_mask': question_attention_mask
        }
        return params

    def forward_translation(self, batch, mode):
        params = self.forward_translation_params(batch)
        
        loss, model_outputs = self.forward_output(batch, mode, params)
        return loss, model_outputs

    def save_model(self, fold, train_step):
        print(f'Save at {train_step}.')
        os.makedirs(self.opt.save_path, exist_ok=True)
        d = f'{self.opt.save_path}/checkpoint-{fold}-{train_step}'
        self.plm.save_pretrained(save_directory=d)

    def save_sa_model(self, sa_train_step):
        d = f'{self.opt.save_path}/SA-{sa_train_step}'
        os.makedirs(d, exist_ok=True)
        torch.save(self.sa.state_dict(), f'{d}/schema_attention.pt')

    def load_best_sa_model(self):
        last_model_suffix = None
        for dirname in os.listdir(self.opt.save_path):
            if dirname.startswith('SA-'):
                suffix = int(dirname.split('-')[-1])
                if last_model_suffix is None or last_model_suffix < suffix:
                    last_model_suffix = suffix

        checkpoint_path = f'{self.opt.save_path}/SA-{last_model_suffix}/schema_attention.pt'
        print(f'Loading SA model from {checkpoint_path}')
        self.sa.load_state_dict(torch.load(checkpoint_path))

    class EvalWrapper:
        def __init__(self, model):
            self.model = model

        def __enter__(self):
            self.model.plm.eval()
            if self.model.sa_mode.use_sa:
                self.model.sa.eval()
                self.model.s_emb.eval()
            
            return self.model

        def __exit__(self, exc_type, exc_value, traceback):
            self.model.plm.train()
            if self.model.sa_mode.use_sa:
                self.model.sa.train()
                self.model.s_emb.train()
            return False

    def eval_mode(self):
        return self.EvalWrapper(self)


class PLMSequentialAttention(PLMPlain):
    def __init__(self, opt):
        super().__init__(opt)

        if self.sa_mode.use_sa:
            self.sa = SequentialSchemaAttention(768, 64, 64)
            self.q_encoder = self.plm.encoder
            if self.sa_mode.embed_mode == 'SHARE_Q_ENCODER':
                self.s_emb = self.plm.shared
        else:
            self.sa = None
            self.s_emb = self.plm.shared
            self.q_encoder = self.plm.encoder

    def predict_schema_occurrence(self, schema_probability, threshold):
        # [n_schema_items]
        predicted_schema_occurrence = []
        for j in range(self.n_schema_items):
            if schema_probability[j] >= threshold:
                predicted_schema_occurrence.append(self.schema_names[j])
        return predicted_schema_occurrence
    
    def schema_language_embedding(self):
        schema_embedding = self.s_emb(self.schema_token_ids)  # [n_schema_items, schema_len, embed_dim]
        masked_schema_embedding = schema_embedding * self.schema_padding_mask.unsqueeze(2)  # [n_schema_items, schema_len, embed_dim]
        mean_schema_embedding = masked_schema_embedding.mean(dim=1)  # [n_schema_items, embed_dim]
        
        return mean_schema_embedding
    
    def forward_schema_embedding(self, question_embedding, question_attention_mask):
        weighted_schema_embedding, schema_probability = self.sa(
            question_embedding * question_attention_mask.unsqueeze(2),
            self.schema_language_embedding()
        )  # [batch, n_schema_items, emb_dim], [batch, n_schema_items]
        return weighted_schema_embedding, schema_probability

    def forward_schema_attention(self, batch, calc_loss):
        question, question_padded_schema, schema_occurrence, schema_occurrence_mask, aql, pir, myrial, am, ecql = batch

        tokenized_question = self.tokenizer(
            question, 
            padding='max_length',
            return_tensors='pt',
            max_length=64,
            truncation=True
        )

        question_token_ids = tokenized_question['input_ids'].cuda()
        question_attention_mask = tokenized_question['attention_mask'].cuda()
        question_embedding = self.q_encoder.embed_tokens(question_token_ids)  # [batch, max_seq_len, emb_dim]

        weighted_schema_embedding, schema_probability = self.forward_schema_embedding(question_embedding, question_attention_mask)

        if calc_loss:
            schema_occurrence_mask = torch.tensor(schema_occurrence_mask, dtype=torch.float32).cuda()
            schema_loss = F.binary_cross_entropy(schema_probability, schema_occurrence_mask)
        else:
            schema_loss = None

        return schema_loss, schema_probability, question_attention_mask, question_embedding, weighted_schema_embedding

    def forward_translation_params(self, batch):
        question, question_padded_schema, schema_occurrence, schema_occurrence_mask, aql, pir, myrial, am, ecql = batch
        b = len(question)

        # We can assume that this model does use SA
        schema_loss, schema_probability, question_attention_mask, question_embedding, weighted_schema_embedding = self.forward_schema_attention(batch, False)
        schema_attention_mask = torch.ones((1, self.n_schema_items)).cuda()
        input_embedding = torch.concat([question_embedding, weighted_schema_embedding], dim=1)  # [batch, max_seq_len + n_schema_items, emb_dim]
        input_attention_mask = torch.concat([schema_attention_mask.repeat((b, 1)), question_attention_mask], dim=1)  # [batch, max_seq_len + n_schema_items]

        params = {
            'inputs_embeds': input_embedding,
            'attention_mask': input_attention_mask
        }

        return params


class PLMSA(PLMSequentialAttention):
    def __init__(self, opt):
        super().__init__(opt)

    def forward_schema_embedding(self, question_embedding, question_attention_mask):
        schema_occurrence_mask = torch.tensor(schema_occurrence_mask, dtype=torch.float32).cuda()
        b = schema_occurrence_mask.shape[0]
        schema_embedding = self.schema_language_embedding().unsqueeze(0).repeat((b, 1, 1))
        weighted_schema_embedding = schema_embedding * schema_occurrence_mask.unsqueeze(2)
        schema_probability = schema_occurrence_mask
        return weighted_schema_embedding, schema_probability

    def forward_schema_attention(self, batch, calc_loss):
        return super().forward_schema_attention(batch, False)


class PLMSAWithSchemaItemNames(PLMPlain):
    def __init__(self, opt):
        super().__init__(opt)

    def preprocess_batch(self, batch):
        question, question_padded_schema, schema_occurrence, schema_occurrence_mask, aql, pir, myrial, am, ecql = batch

        question_padded_gt_schema = [q + ' | ' + ', '.join(s) for q, s in zip(question, schema_occurrence)]

        return question_padded_gt_schema, 256


class PLMPlainNoSchema(PLMPlain):
    def __init__(self, opt):
        super().__init__(opt)

    def preprocess_batch(self, batch):
        question, question_padded_schema, schema_occurrence, schema_occurrence_mask, aql, pir, myrial, am, ecql = batch

        tokenizer_input = question

        return tokenizer_input, 64