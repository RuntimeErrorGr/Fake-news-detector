import os
import re

import torch
import pandas as pd

from yaml import safe_load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader, default_collate

from constants import DATA_CONFIG

class TextDataset(Dataset):

    def __init__(self, df, text_column="text", label_column="label", model_type="lstm", **kwargs):
        super(TextDataset, self).__init__()
        self.df = df
        self.text_column = text_column
        self.label_column = label_column
        self.model_type = model_type
        self.tokenizer = kwargs.get("tokenizer", None)
        self.vocab = kwargs.get("vocab", None)
        self.tokenize = self._bert_tokenize if model_type == "bert" else self._nonbert_tokenize

    def _bert_tokenize(self, text):

        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt"
        )
        inputs = {k:v.squeeze() for k, v in inputs.items()}
        return inputs
    
    def _nonbert_tokenize(self, text):
        return self.vocab(self.tokenizer(text))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        text, label = row[self.text_column], row[self.label_column]
        return self.tokenize(text), label, row["id"]
    

def load_config(config_file):

    with open(config_file, mode="r", encoding="UTF-8") as fin:
        config = safe_load(fin)

    return config

def text_preprocess(text):

    text = re.sub("[^a-zA-Z0-9\$]+", " ", text)
    text = re.sub(" +", " ", text)
    return text.lower()

def preprocess(df: pd.DataFrame, enable_preproc=True):

    subset = ["text", "label"] if "label" in df.columns else ["text"]
    df.dropna(axis=0, how="any", subset=subset, inplace=True)
    if "label" in df.columns:
        df.drop_duplicates(subset=["text"], keep="first", inplace=True)
    if enable_preproc:
        df["text"] = df["text"].apply(text_preprocess)
    df.drop(df[df.text.apply(lambda x: re.search("[a-zA-Z]+", x) is None)].index, inplace=True)

def collate_fn(batch, max_len):

    texts, labels, lengths, ids = [], [], [], []
    for text, label, sample_id in batch:
        lengths.append(len(text[:max_len]))
        texts.append(torch.tensor(text[:max_len]))
        labels.append(torch.tensor(label,dtype=torch.float32))
        ids.append(sample_id)

    return (
        torch.nn.utils.rnn.pad_sequence(texts, batch_first=True),
        torch.tensor(labels),
        torch.tensor(lengths),
        torch.tensor(ids)
    )

def isot_clean(txt):

    txt = re.sub("^([A-Za-z\. ,\/]+)?(\(Reuters\))? - ","", txt)
    txt = re.sub("(https?://\S+)|(\((pic\.twitter\.com?)|(bit\.ly)\S+\))|(\(?(@|#)[a-zA-Z0-9_]+\)?)", "", txt)
    txt = re.sub("((Video)|(Featured)|(Via)|(Read more)|(READ MORE)).*","",txt)
    
    return txt

def get_dataloaders(dataset_id="fake_news", model="lstm", random_state=42, **kwargs):

    config = load_config(DATA_CONFIG)
    input_path = config["datasets"][dataset_id].get("input_path", None)
    max_len = config["datasets"][dataset_id]["max_len"]
    max_tokens = config["datasets"][dataset_id]["max_tokens"]
    val_size = kwargs.get("val_size", 0.05)
    test_size = kwargs.get("test_size", 0.2)

    if dataset_id == "fake_news":
        train_df = pd.read_csv(os.path.join(input_path, "train.csv"))
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
        test_df = pd.read_csv(os.path.join(input_path, "test.csv"))
        test_df["label"] = [-1] * len(test_df)
    elif dataset_id == "isot":
        fake_df = pd.read_csv(os.path.join(input_path, "Fake.csv"))
        true_df = pd.read_csv(os.path.join(input_path, "True.csv"))
        fake_df["label"] = [1] * len(fake_df)
        true_df["label"] = [0] * len(true_df)
        true_df["id"] = list(range(len(true_df)))
        fake_df["id"] = list(range(len(true_df), len(true_df)+len(fake_df)))
        full_df = (
            pd.concat([true_df, fake_df], ignore_index=True)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )
        full_df.text = full_df.text.apply(lambda x: isot_clean(x))
        train_df, test_df = train_test_split(full_df, test_size=test_size, random_state=random_state)
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
    elif dataset_id == "liar":
        hf_dset = load_dataset(dataset_id)

        train_df = hf_dset["train"].rename_column("statement", "text").to_pandas()
        val_df = hf_dset["validation"].rename_column("statement", "text").to_pandas()
        test_df = hf_dset["test"].rename_column("statement", "text").to_pandas()

        train_df["id"] = train_df["id"].apply(lambda x: int(x.split(".")[0]))
        val_df["id"] = val_df["id"].apply(lambda x: int(x.split(".")[0]))
        test_df["id"] = test_df["id"].apply(lambda x: int(x.split(".")[0]))

        binarize = kwargs.get("binarize", True)
        if binarize:
            train_df.label = train_df.label.apply(lambda x: 0 if x == 3 else 1)
            val_df.label = val_df.label.apply(lambda x: 0 if x == 3 else 1)
            test_df.label = test_df.label.apply(lambda x: 0 if x == 3 else 1)
        
        
    preprocess(train_df, kwargs.get("enable_preproc", True))
    preprocess(val_df, kwargs.get("enable_preproc", True))
    preprocess(test_df, kwargs.get("enable_preproc", True))

    vocab = None
    if model != "bert":
        tokenizer = get_tokenizer("basic_english")
        stop_words = stopwords.words("english")
        wnl = WordNetLemmatizer()
        enable_lemmatize = kwargs.get("lemmatize", False)
        lemmatize = (lambda x: wnl.lemmatize(x)) if enable_lemmatize else (lambda x: x)
        vocab = build_vocab_from_iterator(
            iter(train_df["text"].apply(
                lambda x: [
                    lemmatize(token) for token in tokenizer(x) if token not in stop_words
                ]
            )),
            specials=["<unk>"],
            special_first=False
        )
        vocab.set_default_index(vocab["<unk>"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            kwargs.get("model_id", "bert-base-uncased")
        )
        tokenizer.model_max_length = max_tokens

    train_ds = TextDataset(df=train_df, model_type=model, tokenizer=tokenizer, vocab=vocab)
    val_ds = TextDataset(df=val_df, model_type=model, tokenizer=tokenizer, vocab=vocab)
    test_ds = TextDataset(df=test_df, model_type=model, tokenizer=tokenizer, vocab=vocab)

    collate_function = (lambda x: collate_fn(x, max_len)) if model != "bert" else (lambda x: default_collate(x))

    train_loader = DataLoader(
        train_ds,
        collate_fn=collate_function,
        **config["dataloaders"]["train"]
    )
    val_loader = DataLoader(
        val_ds,
        collate_fn=collate_function,
        **config["dataloaders"]["validation"]
    )
    test_loader = DataLoader(
        test_ds,
        collate_fn=collate_function,
        **config["dataloaders"]["test"]
    )

    return (train_loader, val_loader, test_loader), (tokenizer, vocab)


