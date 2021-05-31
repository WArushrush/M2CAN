
from model.config import *
import json

f = open("word2idx.json", 'r')
word2idx = json.load(f)
f.close()
f = open("idx2word.json", 'r')
idx2word = json.load(f)
f.close()


def process_text_batch(text_batch, bs):
    len_batch = [len(txt) for txt in text_batch]
    max_len = max(len_batch)
    text_batch = [txt + [padding_idx] * (max_len - len(txt)) for txt in text_batch]
    pos = [idx for idx in range(bs)]
    temp = sorted(pos, key=lambda x: len_batch[x], reverse=True)
    return [text_batch[temp[idx]] for idx in range(bs)], [len_batch[temp[idx]] for idx in range(bs)]


def text_transform(string):
    string = string.split()
    res = []
    for word in string:
        try:
            temp = word2idx[word]
            res.append(temp)
        except:
            res.append(UNK_token)
    return res


def text_transform2(strings_batch, bert_model):
    text = ['[CLS] ' for _ in range(len(strings_batch))]
    for _ in range(len(strings_batch)):
        cur_len = 0
        strings = strings_batch[_]
        for idx in range(len(strings)):
            string = strings[idx]
            cur_len += 1
            if cur_len > 3:
                break
            text[_] += string + ' [SEP] '
    max_len = max([len(txt) for txt in text])
    text_dict = [tokenizer.encode_plus(txt, max_length=max_len, pad_to_max_length=True, add_special_tokens=True, return_attention_mask=True, truncation=True) for txt in text]
    input_ids = [torch.tensor(txt_dict['input_ids'], requires_grad=False).unsqueeze(0).to(device) for txt_dict in text_dict]
    token_type_ids = [torch.tensor(txt_dict['token_type_ids'], requires_grad=False).unsqueeze(0).to(device) for txt_dict in text_dict]
    attention_mask = [torch.tensor(txt_dict['attention_mask'], requires_grad=False).unsqueeze(0).to(device) for txt_dict in text_dict]

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    res = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    return res[1]


def text_augmentation_transform2(strings_batch, bert_model):
    text = ['[CLS] ' for _ in range(len(strings_batch))]
    for _ in range(len(strings_batch)):
        cur_len = 0
        strings = strings_batch[_]
        for idx in range(len(strings)):
            string = strings[idx]
            temp = string.split()
            temp2 = []
            for ttt in range(len(temp)):
                rd = random.random()
                if rd > 0.1:
                    temp2.append(temp[ttt])
            string = ' '.join(temp2)
            cur_len += 1
            if cur_len > 3:
                break
            text[_] += string + ' [SEP] '
    max_len = max([len(txt) for txt in text])
    text_dict = [tokenizer.encode_plus(txt, max_length=max_len, pad_to_max_length=True, add_special_tokens=True, return_attention_mask=True, truncation=True) for txt in text]
    input_ids = [torch.tensor(txt_dict['input_ids'], requires_grad=False).unsqueeze(0).to(device) for txt_dict in text_dict]
    token_type_ids = [torch.tensor(txt_dict['token_type_ids'], requires_grad=False).unsqueeze(0).to(device) for txt_dict in text_dict]
    attention_mask = [torch.tensor(txt_dict['attention_mask'], requires_grad=False).unsqueeze(0).to(device) for txt_dict in text_dict]

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    res = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    return res[1]
