import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import argparse
import logging

from transformers import BertForPreTraining, BertTokenizer, BertConfig, BertModel
from torch.nn import CrossEntropyLoss

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0, metric = 'l2'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        # print('ContrastiveLoss, Metric:', self.metric)

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        #elf.check_type_forward((x0, x1, y))

        # euclidian distance
        if self.metric == 'l2':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
            dist = torch.sqrt(dist_sq)
        elif self.metric == 'cos':
            prod = torch.sum(x0 * x1, -1)
            dist = 1 - prod /  torch.sqrt(torch.sum(x0**2, 1) * torch.sum(x1**2, 1))
            dist_sq = dist ** 2
            #print(x0, x1, torch.sum(torch.pow(x0-x1, 2), 1) / x0.shape[-1], dist, dist_sq)
        else:
            logging.info("Error Loss Metric!!")
            return 0
        #dist = torch.sum( - x0 * x1 / np.sqrt(x0.shape[-1]), 1).exp()
        #dist_sq = dist ** 2

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, dist_sq, dist

def label_smoothed_nll_loss(contrastive_scores, contrastive_labels, eps=0.0):
    '''
        contrasive_scores: bsz x seqlen x seqlen
        contrasive_labels: bsz x seqlen; masked positions with 0., otherwise 1.
    '''
    bsz, seqlen, _ = contrastive_scores.size()
    logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
    gold = torch.arange(seqlen).view(-1,)
    gold = gold.expand(bsz, seqlen).contiguous().view(-1)
    if contrastive_scores.is_cuda:
        gold = gold.cuda(contrastive_scores.get_device())
    loss =  -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
    loss = loss.view(bsz, seqlen) * contrastive_labels
    loss = torch.sum(loss) / contrastive_labels.sum()

    _, pred = torch.max(logprobs, -1)
    correct_num = torch.eq(gold, pred).float().view(bsz, seqlen)
    correct_num = torch.sum(correct_num * contrastive_labels)
    total_num = contrastive_labels.sum()
    return loss, correct_num, total_num

class BERTContrastivePretraining(nn.Module):
    def __init__(self, model_name, sim='cosine', temperature=0.01, use_contrastive_loss='True'):
        super(BERTContrastivePretraining, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.model = BertForPreTraining.from_pretrained(model_name)
        self.bert = self.model.bert
        self.cls = self.model.cls
        self.config = BertConfig.from_pretrained(model_name)
        embed_dim = self.config.hidden_size
        self.embed_dim = embed_dim
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)
        assert sim in ['dot_product', 'cosine']
        self.sim = sim
        if self.sim == 'dot_product':
            logging.info('use dot product similarity')
        else:
            logging.info('use cosine similarity')
        self.temperature = temperature

        if use_contrastive_loss == 'True':
            use_contrastive_loss = True
        elif use_contrastive_loss == 'False':
            use_contrastive_loss = False
        else:
            raise Exception('Wrong contrastive loss setting!')

        self.use_contrastive_loss = use_contrastive_loss
        if self.use_contrastive_loss:
            print ('Initializing teacher BERT.')
            self.teacher_bert = BertModel.from_pretrained(model_name)
            for param in self.teacher_bert.parameters():
                param.requires_grad = False
            logging.info('Teacher BERT initialized.')
        else:
            logging.info('Train BERT with vanilla MLM loss.')

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)

        # save model
        self.bert.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)

    def compute_teacher_representations(self, input_ids, token_type_ids, attention_mask):
        bsz, seqlen = input_ids.size()
        outputs = self.teacher_bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        rep, pooled_output = outputs[0], outputs[1]
        # rep: bsz x seqlen x embed_dim
        rep = rep.view(bsz, seqlen, self.embed_dim)
        logits, sen_relation_scores = self.cls(rep, pooled_output) # bsz x seqlen x vocab_size
        return rep, pooled_output, logits, sen_relation_scores

    def compute_representations(self, input_ids, token_type_ids, attention_mask):
        bsz, seqlen = input_ids.size()
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        rep, pooled_output = outputs[0], outputs[1]
        # rep: bsz x seqlen x embed_dim
        rep = rep.view(bsz, seqlen, self.embed_dim)
        logits, sen_relation_scores = self.cls(rep, pooled_output) # bsz x seqlen x vocab_size
        return rep, pooled_output, logits, sen_relation_scores

    def compute_mlm_loss(self, truth, msk, logits):
        truth = truth.transpose(0,1)
        msk = msk.transpose(0,1)
        msk_token_num = torch.sum(msk).float().item()
        # center
        y_mlm = logits.transpose(0,1).masked_select(msk.unsqueeze(-1).to(torch.bool))
        y_mlm = y_mlm.view(-1, self.vocab_size)
        gold = truth.masked_select(msk.to(torch.bool))
        log_probs_mlm = torch.log_softmax(y_mlm, -1)
        mlm_loss = F.nll_loss(log_probs_mlm, gold, reduction='mean')
        _, pred_mlm = log_probs_mlm.max(-1)
        mlm_correct_num = torch.eq(pred_mlm, gold).float().sum().item()
        return mlm_loss, mlm_correct_num
    
    def get_random_index_second_language(self, batch_size, index):
        for _ in range(10):
            random_document_index = random.randint(0, batch_size - 1)
            if random_document_index != index:
                return random_document_index
        return random.randint(0, batch_size - 1)
    
    def compute_contrastive_loss(self, pooled_output_cls_lang1, pooled_output_cls_lang2, device, distmetric = 'l2'):
        batch_size, hidden_size = pooled_output_cls_lang1.size()
        feature_lang1 = []
        feature_lang2 = []
        agreement = []
        for index_lang1, hidden_rep_lang1 in enumerate(pooled_output_cls_lang1):
            feature_lang1.append(hidden_rep_lang1)
            feature_lang2.append(pooled_output_cls_lang2[index_lang1])
            agreement.append(1)
            #choose 5 negative representation from the same batch that is different from the current index of language 1
            for i in range(5):
                index_lang2 = self.get_random_index_second_language(batch_size, index_lang1)
                feature_lang1.append(hidden_rep_lang1)
                feature_lang2.append(pooled_output_cls_lang2[index_lang2])
                agreement.append(0)
        agreement = torch.FloatTensor(agreement).cuda(device)
        feature_lang1 = torch.stack(feature_lang1).cuda(device)
        feature_lang2 = torch.stack(feature_lang2).cuda(device)
        criterion = ContrastiveLoss(margin = 1.0, metric = distmetric)
        loss, dist_sq, dist = criterion(feature_lang1, feature_lang2, agreement)
        
        return loss




    def forward(self, truth_lang1, inp_lang1, seg_lang1, msk_lang1, attn_msk_lang1, labels_lang1, contrastive_labels_lang1, nxt_snt_flag_lang1, 
                truth_lang2, inp_lang2, seg_lang2, msk_lang2, attn_msk_lang2, labels_lang2, contrastive_labels_lang2, nxt_snt_flag_lang2 ):
        '''
           truth: bsz x seqlen
           inp: bsz x seqlen
           seg: bsz x seqlen
           msk: bsz x seqlen
           attn_msk: bsz x seqlen
           labels: bsz x seqlen; masked positions are filled with -100
           contrastive_labels: bsz x seqlen; masked position with 0., otherwise 1.
        '''
        if truth_lang1.is_cuda:
            is_cuda = True
            device = truth_lang1.get_device()
        else:
            is_cuda = False

        bsz, seqlen = truth_lang1.size()
        masked_rep_lang1, pooled_output_cls_lang1, logits_lang1, sen_relation_scores_lang1 = \
        self.compute_representations(input_ids=inp_lang1, token_type_ids=seg_lang1, attention_mask=attn_msk_lang1)

        # compute masked language model loss
        # --------------------------------------------------------------------------------------- #
        mlm_loss, mlm_correct_num = self.compute_mlm_loss(truth_lang1, msk_lang1, logits_lang1)
        # --------------------------------------------------------------------------------------- #

        # compute contrastive loss
        if self.use_contrastive_loss:
            truth_rep_lang2, pooled_output_cls_lang2, truth_logits_lang2, _ =  self.compute_teacher_representations(input_ids=truth_lang2, token_type_ids=seg_lang2, attention_mask=attn_msk_lang2)
            
            contrastive_loss = self.compute_contrastive_loss(pooled_output_cls_lang1, pooled_output_cls_lang2, device)

            ''' 
                mask_rep, truth_rep : hidden_size
                rep, left_rep, right_rep: bsz x seqlen x embed_dim
            '''
            # if self.sim == 'dot_product':
            #     contrastive_scores = torch.matmul(masked_rep, truth_rep.transpose(1,2))
                
            # elif self.sim == 'cosine': # 'cosine'
            #     masked_rep = masked_rep / masked_rep.norm(dim=2, keepdim=True)
            #     truth_rep = truth_rep / truth_rep.norm(dim=2, keepdim=True)
            #     contrastive_scores = torch.matmul(masked_rep, truth_rep.transpose(1,2)) / self.temperature # bsz x seqlen x seqlen
            # else:
            #     raise Exception('Wrong similarity mode!!!')

            # assert contrastive_scores.size() == torch.Size([bsz, seqlen, seqlen])
            # contrastive_loss, correct_contrastive_num, total_contrastive_num = \
            # label_smoothed_nll_loss(contrastive_scores, contrastive_labels)
        else:
            correct_contrastive_num, total_contrastive_num = 0., 1.

        if is_cuda:
            nxt_snt_flag = nxt_snt_flag_lang1.type(torch.LongTensor).cuda(device)
        else:
            nxt_snt_flag = nxt_snt_flag_lang1.type(torch.LongTensor)
        next_sentence_loss = self.loss_fct(sen_relation_scores_lang1.view(-1, 2), nxt_snt_flag.view(-1))
        if self.use_contrastive_loss:
            tot_loss = mlm_loss + next_sentence_loss + contrastive_loss
        else:
            tot_loss = mlm_loss + next_sentence_loss

        next_sentence_logprob = torch.log_softmax(sen_relation_scores_lang1, -1)
        next_sentence_predictions = torch.max(next_sentence_logprob, dim = -1)[-1]
        nxt_snt_correct_num = torch.eq(next_sentence_predictions, nxt_snt_flag.view(-1)).float().sum().item()
        tot_tokens = msk_lang1.float().sum().item()
        if is_cuda:
            return tot_loss, torch.Tensor([mlm_correct_num]).cuda(device), torch.Tensor([tot_tokens]).cuda(device), \
            torch.Tensor([nxt_snt_correct_num]).cuda(device)
        else:
            return tot_loss, torch.Tensor([mlm_correct_num]), torch.Tensor([tot_tokens]), torch.Tensor([nxt_snt_correct_num])
