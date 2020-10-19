"""Simple bert fine tuning classifier"""

from collections import defaultdict
import os
import pickle

import scipy
from sklearn.model_selection import KFold

from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import torch

from tqdm import tqdm

CUDA = (torch.cuda.device_count() > 0)



def prep_data(texts, labels, tokenizer):
    """ prepare data for dataloader: get text, ids, and labels into dict"""
    out = defaultdict(list)
    for text, label in zip(texts, labels):
        encoded_sent = tokenizer.encode(text, add_special_tokens=True)
        out['input_raw'].append(text)
        out['input_ids'].append(encoded_sent)
        out['labels'].append(label)
    out['input_ids'] = pad_sequences(
        out['input_ids'],
        maxlen=128,
        dtype='long',
        value=0,
        truncating='post',
        padding='post')
    return out

def build_dataloader(*args, sampler='random', batch_size=32):
    data = (torch.tensor(x) for x in args)
    data = TensorDataset(*data)

    sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

def select_indices(idx, l):
    out = []
    idx = set(idx)
    for i, j in enumerate(l):
        if i in idx:
            out.append(j)
    return out



def train_bert(dataloader, epochs, learning_rate):
    """Use a dataloader to run training."""
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False)
    if CUDA:
        model = model.cuda()
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    for epoch in range(epochs):
        model.train()
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if CUDA: 
                batch = (x.cuda() for x in batch)
            input_ids, labels = batch
            model.zero_grad()
            loss, logits = model(input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

    return model

def inference_bert(model, dataloader):
    """Use a dataloader to run inference."""
    model.eval()
    out = []
    for step, batch in enumerate(dataloader):
        if CUDA: 
            batch = (x.cuda() for x in batch)
        input_ids, labels = batch
        with torch.no_grad():
            loss, logits = model(input_ids, labels=labels)            
        preds = scipy.special.softmax(logits.cpu().numpy(), axis=1)
        out += preds.tolist()
    return out


def fit_predict(text, Y, epochs=2, learning_rate=2e-5, working_dir='~/Desktop/TEST'):
    """ Train bert to predict Y from the text and get Y_hat with cross validation.
    """
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

    if os.path.exists(working_dir + "/data.cache.pkl"):
        data = pickle.load(open(working_dir + "/data.cache.pkl", 'rb'))
    else:
        data = prep_data(text, Y, tokenizer)
        pickle.dump(data, open(working_dir + "/data.cache.pkl", 'wb'))

    all_preds = [0 for _ in range(len(data['input_ids']))]
    kf = KFold(n_splits=3)
    for train_index, test_index in tqdm(kf.split(data['input_ids'])):
        train_dataloader = build_dataloader(
            select_indices(train_index, data['input_ids']),
            select_indices(train_index, data['labels']))
        test_dataloader = build_dataloader(
            select_indices(test_index, data['input_ids']),
            select_indices(test_index, data['labels']),
            sampler='test')

        model = train_bert(train_dataloader, epochs, learning_rate)
        preds = inference_bert(model, test_dataloader)
    
        for pred, idx in zip(preds, test_index):
            all_preds[idx] = pred

    return all_preds


