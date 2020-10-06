#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : data_utils.py
from __future__ import print_function

import os

import logging
logger = logging.getLogger(__name__)

class Example(object):
    def __init__(self, guid, text_a, label, meta, att):
        self.guid = guid
        self.text_a = text_a
        self.text_b = None
        self.label = label
        self.att = att
        self.aux_label = []

        for no in range(att):
            if str(no) in meta:
                self.aux_label.append("1")
            else:
                self.aux_label.append("0")

class Data(object):
    def __init__(self, data_path, att=2):

        sep = "="*20
        dataset = [[], [], []]
        split = 0
        labels = set()

        guid = 0
        with open(data_path) as f:
            for line in f:
                line = line.lstrip()
                if line.strip() == sep:
                    split += 1
                    guid = 0
                    continue
                label, text, meta = line.split("\t")
                dataset[split].append(Example(guid, text, label, meta.strip(), att))
                guid += 1

                labels.add(label)

        self.label_list = list(labels)
        self.dataset = dataset

    def get_labels(self):
        return self.label_list

    def get_train_examples(self):
        return self.dataset[0]

    def get_dev_examples(self):
        return self.dataset[1]

    def get_test_examples(self):
        return self.dataset[2]

class AG_data(Data):
    @classmethod
    def get_ag_data(cls, data_dir):
        data_path = os.path.join(data_dir, "ag_data.txt")
        return cls(data_path, att=5)

class Blog_data(Data):
    @classmethod
    def get_blog_data(cls, data_dir):
        data_path = os.path.join(data_dir, "blog_data.txt")
        return cls(data_path, att=2)

class TP_data(Data):
    @classmethod
    def get_tp_data(cls, data_dir):
        data_path = os.path.join(data_dir, "tp_us.txt")
        return cls(data_path, att=2)

class TPUK_data(Data):
    @classmethod
    def get_tp_data(cls, data_dir):
        data_path = os.path.join(data_dir, "tp_uk.txt")
        return cls(data_path, att=2)

def get_processors(data_dir):
    get_data = {"ag": lambda : AG_data.get_ag_data(data_dir),
                "bl": lambda : Blog_data.get_blog_data(data_dir),
                "tp": lambda : TP_data.get_tp_data(data_dir),
                "tpuk": lambda : TPUK_data.get_tp_data(data_dir),
                }

    return get_data

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None, aux_label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.aux_label = aux_label

def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    label_list=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    label_map = {label: i for i, label in enumerate(label_list)}
    aux_label_map = {"0": 0, "1": 1}

    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        label = label_map[example.label]
        aux_label = [aux_label_map[l] for l in example.aux_label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))
            logger.info("auxilary label: %s " % (" ".join(example.aux_label)))

        features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    label=label, aux_label=aux_label
                )
            )

    return features
