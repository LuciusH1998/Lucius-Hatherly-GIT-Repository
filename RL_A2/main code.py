# %% [markdown]
# # Assignment 2

# %% [markdown]
# This notebook is intended to produce the plots and figures for the report on Problem 1 of the practical. You should not run this notebook in Google Colab until you have finished constructing the correct solutions for transformer_solution.py and gru_solution.py
# 
# This notebook provides some limited commentary on several HuggingFace Features and toolage. You will use HuggingFace Datasets to load the Yelp Polarity dataset for sentiment analysis. The notebook will define a Bert tokenizer, collate functions, and then train and evaluate several models using the HuggingFace utilities mentioned above. Remember, the most crucial part here is running the experiments for the report.

# %%
%matplotlib inline
%load_ext autoreload
%autoreload 2

# %% [markdown]
# ### Mount your Google Drive

# %%
# If you run this notebook locally or on a cluster (i.e. not on Google Colab)
# you can delete this cell which is specific to Google Colab. You may also
# change the paths for data/logs in Arguments below.


from google.colab import drive
drive.mount('/content/gdrive')

!pip install -qqq datasets transformers textattack --upgrade

# %% [markdown]
# ### Link your assignment folder & install requirements
# Enter the path to the assignment folder in your Google Drive
# If you run this notebook locally or on a cluster (i.e. not on Google Colab)
# you can delete this cell which is specific to Google Colab. 

# %%
import sys
import os
import shutil
import warnings

folder = "" #@param {type:"string"}
!ln -Ts "$folder" /content/assignment 2> /dev/null
!cp gdrive/MyDrive/Assignment2/transformer_solution.py .
!cp gdrive/MyDrive/Assignment2/gru_solution.py .

# Add the assignment folder to Python path
if '/content/assignment' not in sys.path:
  sys.path.insert(0, '/content/assignment')

# Check if CUDA is available
import torch
if not torch.cuda.is_available():
  warnings.warn('CUDA is not available.')

# %% [markdown]
# ### Running on GPU
# For this assignment, it will be necessary to run your experiments on GPU. To make sure the notebook is running on GPU, you can change the notebook settings with
# * (EN) `Edit > Notebook Settings`
# * (FR) `Modifier > Paramètres du notebook`
# 

# %%
import matplotlib.pyplot as plt
import urllib.request
import time
import os
import json
import random
from typing import List, Dict, Union, Optional, Tuple

from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW

from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoModel, AutoTokenizer
from tokenizers import Tokenizer

from transformer_solution import Transformer, MultiHeadedAttention
from gru_solution import EncoderDecoder

# %%
def set_seed(seed: int = 0, device: torch.device = None):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return rng

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--> Device selected: {device}")
rng = set_seed(seed=0, device=device)

# %%
dataset_name = "yelp_polarity"
dataset_train_original = load_dataset(dataset_name, split="train", cache_dir="assignment/data")
dataset_test_original = load_dataset(dataset_name, split="test", cache_dir="assignment/data").shuffle(generator=rng)
dataset_test = dataset_test_original.select(range(1000))
dataset_train = concatenate_datasets([
    dataset_train_original,
    dataset_test_original.select(range(1000, len(dataset_test_original)))
])
print(f"{len(dataset_train)=}, {len(dataset_test)=}")

# %% [markdown]
# ### 🔍 Quick look at the data
# Lets have quick look at a few samples in our test set.

# %%
n_samples_to_see = 3
for i in range(n_samples_to_see):
  print("-"*30)
  print("title:", dataset_test[i]["text"])
  print("label:", dataset_test[i]["label"])

# %% [markdown]
# ### 1️. Tokenize the `text`
# Tokenize the `text`portion of each sample (i.e. parsing the text to smaller chunks). Tokenization can happen in many ways; traditionally, this was done based on the white spaces. With transformer-based models, tokenization is performed based on the frequency of occurrence of "chunk of text". This frequency can be learned in many different ways. However the most common one is the [**wordpiece**](https://arxiv.org/pdf/1609.08144v2.pdf) model. 
# > The wordpiece model is generated using a data-driven approach to maximize the language-model likelihood
# of the training data, given an evolving word definition. Given a training corpus and a number of desired
# tokens $D$, the optimization problem is to select $D$ wordpieces such that the resulting corpus is minimal in the
# number of wordpieces when segmented according to the chosen wordpiece model.
# 
# Under this model:
# 1. Not all things can be converted to tokens depending on the model. For example, most models have been pretrained without any knowledge of emojis. So their token will be `[UNK]`, which stands for unknown.
# 2. Some words will be mapped to multiple tokens!
# 3. Depending on the kind of model, your tokens may or may not respect capitalization

# %%
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
input_sample = "Welcome to IFT6135. We now teach you 🤗(HUGGING FACE) Library :DDD."
tokenizer.tokenize(input_sample)

# %% [markdown]
# ### 2. Encoding
# Once we have tokenized the text, we then need to convert these chuncks to numbers so we can feed them to our model. This conversion is basically a look-up in a dictionary **from `str` $\to$ `int`**. The tokenizer object can also perform this work. While it does so it will also add the *special* tokens needed by the model to the encodings. 

# %%
input_sample = "Welcome to IFT6135. We now teach you 🤗(HUGGING FACE) Library :DDD." #@param {type: "string"}

print("--> Token Encodings:\n",tokenizer.encode(input_sample))
print("-."*15)
print("--> Token Encodings Decoded:\n",tokenizer.decode(tokenizer.encode(input_sample)))

# %% [markdown]
# ### 3️. Truncate/Pad samples
# Since all the sample in the batch will not have the same sequence length, we would need to truncate the longer sequences (i.e. the ones that exeed a predefined maximum length) and pad the shorter ones so we that we can equal length for all the samples in the batch. Once this is achieved, we would need to convert the result to `torch.Tensor`s and return. These tensors will then be retrieved from the [dataloader](https://https//pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

# %%
class Collate:
    def __init__(self, model_name: str, max_len: int) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_len = max_len
        self.text_column = "text"
        self.label_column = "label"

    def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        texts = list(map(lambda batch_instance: batch_instance[self.text_column], batch))
        tokenized_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        labels = list(map(lambda batch_instance: int(batch_instance[self.label_column]), batch))
        labels = torch.LongTensor(labels)
        return dict(tokenized_inputs, **{"labels": labels})


# %% [markdown]
# #### 🧑‍🍳 Setting up the collate function

# %%
model_name = "bert-base-uncased"
sample_max_length = 256         #@param {type: "integer"}
collate = Collate(model_name=model_name, max_len=sample_max_length)

# %% [markdown]
# ### 4. Models

# %%
class ReviewClassifierPretrained(nn.Module):
    def __init__(
        self,
        backbone: str = "bert-base-uncased",
        backbone_hidden_size: int = 768,
        num_classes: int = 2,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_hidden_size = backbone_hidden_size
        self.num_classes = num_classes
        self.device = device
        self.back_bone = AutoModel.from_pretrained(
            self.backbone,
            output_attentions=False,
            output_hidden_states=False,
        )
        for parameter in self.back_bone.parameters():
            parameter.requires_grad= False
        self.classifier = torch.nn.Linear(self.backbone_hidden_size, self.num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():
            self.back_bone.eval()
            back_bone_output = self.back_bone(input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        hidden_states = back_bone_output[0]
        pooled_output = hidden_states[:, 0]  # getting the [CLS] token
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(
            logits.view(-1, self.num_classes),
            labels.view(-1).to(self.device),
        )
        return loss, logits


class ReviewClassifierRNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        vocabulary_size: int = 30522,
        encoder_only: bool = False,
        dropout: float = 0.5,
        embed_dim: int = 256,
        with_attn: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_only = encoder_only
        self.device = device
        self.back_bone = EncoderDecoder(
            vocabulary_size=vocabulary_size,
            dropout=dropout,
            encoder_only=encoder_only,
            with_attn=with_attn,
        )
        self.classifier = torch.nn.Linear(embed_dim, self.num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pooled_output, _ = self.back_bone(input_ids.to(self.device), attention_mask.to(self.device))
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(
            logits.view(-1, self.num_classes),
            labels.view(-1).to(self.device),
        )
        return loss, logits


class ReviewClassifierTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        vocabulary_size: int = 30522,
        sequence_length: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        block: str="prenorm",
        embed_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.back_bone = Transformer(
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            block=block,
            dropout=dropout,
        )
        self.classifier = torch.nn.Linear(embed_dim, self.num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pooled_output = self.back_bone(input_ids.to(self.device), attention_mask.to(self.device))
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(
            logits.view(-1, self.num_classes),
            labels.view(-1).to(self.device),
        )
        return loss, logits


# %% [markdown]
# ### 5. Trainer

# %%
def train_one_epoch(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    logging_frequency: int,
    logs: dict,
    device: torch.device,
    epoch_idx: int = 0,
):
    model.train()
    epoch_loss = 0
    logging_loss = 0
    epoch_start_time = time.time()
    logfreq_start_time = time.time()
    num_train_batch = len(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        loss, _ = model(**batch)
        loss.backward()
        optimizer.step()
        current_lr = optimizer.param_groups[0]["lr"]
        # print(f"{step=}, {current_lr=}")

        epoch_loss += loss.item()
        logging_loss += loss.item()

        if (step + 1) % logging_frequency == 0 or (step + 1) == num_train_batch:
            logfreq_time = time.time() - logfreq_start_time
            logs["train_time_accum"].append(logfreq_time + logs["train_time_accum"][-1])
            logs["train_loss_logfreq"].append(logging_loss / logging_frequency)
            (
                eval_acc,
                tp_rate,
                fp_rate,
                tn_rate,
                fn_rate,
                f1,
                eval_loss,
                eval_time,
            ) = evaluate(model=model, test_dataloader=test_dataloader, device=device)
            logs["eval_acc_logfreq"].append(eval_acc)
            logs["eval_tp_rate_logfreq"].append(tp_rate)
            logs["eval_fp_rate_logfreq"].append(fp_rate)
            logs["eval_tn_rate_logfreq"].append(tn_rate)
            logs["eval_fn_rate_logfreq"].append(fn_rate)
            logs["eval_f1_score_logfreq"].append(f1)
            logs["eval_loss_logfreq"].append(eval_loss)
            logs["eval_time_accum"].append(eval_time + logs["eval_time_accum"][-1])
            print(
                f"Epoch {epoch_idx+1} step {step+1}/{num_train_batch}: "
                f"train time {logs["train_time_accum"][-1]:.1f} seconds, "
                f"train loss {logs["train_loss_logfreq"][-1]:.3f}, "
                f"eval time {eval_time:.1f}, eval loss {eval_loss:.3f}, eval acc {eval_acc:.3f}, "
                f"eval tp rate {tp_rate:.3f}, eval fp rate {fp_rate:.3f}, eval tn rate {tn_rate:.3f}, eval fn rate {fn_rate:.3f}, eval f1 score {f1:.3f}, "
                f"lr {current_lr:.3e}")

            logging_loss = 0
            logfreq_start_time = time.time()

            if (step + 1) == num_train_batch:
                logs["eval_loss_epoch"].append(eval_loss)
                logs["eval_time_epoch"].append(eval_time)
                logs["eval_acc_epoch"].append(eval_acc)

    train_loss_epoch = epoch_loss / num_train_batch
    train_time_epoch = time.time() - epoch_start_time
    logs["train_loss_epoch"].append(train_loss_epoch)
    logs["train_time_epoch"].append(train_time_epoch)
    print(f"Epoch {epoch_idx+1}: train time epoch: {train_time_epoch:.1f}, train loss epoch mean: {train_loss_epoch:.3f}")


# %%
def evaluate(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
):
    model.eval()
    eval_loss = 0
    y_pred = []
    y_true = []
    start_time = time.time()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            loss, logits = model(**batch)
            eval_loss += loss.item()
            predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
            y_pred.extend(predictions)
            y_true.extend(batch["labels"].cpu().numpy())
    y_pred, y_true = np.array(y_pred, dtype=np.float32), np.array(y_true, dtype=np.float32)
    # print(f"{sum(y_true==1)=}, {sum(y_true==0)=}")
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    tp_rate = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    fp_rate = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0)
    tn_rate = np.sum((y_true == 0) & (y_pred == 0)) / np.sum(y_true == 0)
    fn_rate = np.sum((y_true == 1) & (y_pred == 0)) / np.sum(y_true == 1)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    eval_loss = eval_loss / len(test_dataloader)
    model.train()
    return (
        accuracy,
        tp_rate,
        fp_rate,
        tn_rate,
        fn_rate,
        f1,
        eval_loss,
        time.time() - start_time
    )


# %%
def reset_logs(model):
    logs = dict()
    logs["parameters"] = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of parameters: {logs['parameters']}")

    logs["train_time_accum"] = [0]
    logs["train_loss_logfreq"] = []
    logs["train_loss_epoch"] = []
    logs["train_time_epoch"] = []

    logs["eval_time_accum"] = [0]
    logs["eval_acc_logfreq"] = []
    logs["eval_tp_rate_logfreq"] = []
    logs["eval_fp_rate_logfreq"] = []
    logs["eval_tn_rate_logfreq"] = []
    logs["eval_fn_rate_logfreq"] = []
    logs["eval_f1_score_logfreq"] = []
    logs["eval_loss_logfreq"] = []
    logs["eval_loss_epoch"] = []
    logs["eval_time_epoch"] = []
    logs["eval_acc_epoch"] = []
    return logs


def save_logs(dictionary, log_dir, exp_id):
    log_dir = os.path.join(log_dir, exp_id)
    os.makedirs(log_dir, exist_ok=True)
    # Log arguments
    with open(os.path.join(log_dir, "args.json"), "w") as f:
      json.dump(dictionary, f, indent=2)

def save_model(model, log_dir, exp_id):
  log_dir = os.path.join(log_dir, exp_id)
  os.makedirs(log_dir, exist_ok=True)
  # Save model
  torch.save(model.state_dict(), f"assignment/models/model_{exp_id}.pt")


# %% [markdown]
# ### 6. Problem 2
# Feel free to modify this code however it is convenient for you to produce a report except for the model parameters.

# %%
train_dataloader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate
)
test_dataloader = DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate
)

# %%
batch_size = 512            #@param {type: "integer"}
logging_frequency = 25      #@param {type: "integer"}
learning_rate = 1e-4        #@param {type: "number"}

sample_max_length = 256     #@param {type: "integer"}
experimental_setting = 1    #@param {type: "integer"}
num_epochs = 5              #@param {type: "integer"}

# %%
if experimental_setting == 1:
    print("GRU: no dropout, encoder only")
    model = ReviewClassifierRNN(
        dropout=0.0,
        encoder_only=True,
        device=device,
    )
elif experimental_setting == 2:
    print("GRU: dropout, encoder only")
    model = ReviewClassifierRNN(
        dropout=0.3,
        encoder_only=True,
        device=device,
    )
elif experimental_setting == 3:
    print("GRU: dropout, encoder-decoder, no attn")
    model = ReviewClassifierRNN(
        dropout=0.3,
        encoder_only=False,
        with_attn=False,
        device=device,
    )
elif experimental_setting == 4:
    print("GRU: dropout, encoder-decoder, with attn")
    model = ReviewClassifierRNN(
        dropout=0.3,
        encoder_only=False,
        with_attn=True,
        device=device,
    )
elif experimental_setting == 5:
    print("Transformer: 2 layers, prenorm")
    model = ReviewClassifierTransformer(
        sequence_length=sample_max_length,
        num_heads=4,
        num_layers=2,
        block="prenorm",
        dropout=0.3,
        device=device,
    )
elif experimental_setting == 6:
    print("Transformer: 4 layers, prenorm")
    model = ReviewClassifierTransformer(
        sequence_length=sample_max_length,
        num_heads=4,
        num_layers=4,
        block="prenorm",
        dropout=0.3,
        device=device,
    )
elif experimental_setting == 7:
    print("Transformer: 2 layers, postnorm")
    model = ReviewClassifierTransformer(
        sequence_length=sample_max_length,
        num_heads=4,
        num_layers=2,
        block="postnorm",
        dropout=0.3,
        device=device,
    )
elif experimental_setting == 8:
    print("Pretrained BERT")
    model = ReviewClassifierPretrained(
        backbone=model_name,
        device=device,
    )


# %%
# setting up the optimizer
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
model.to(device)

# %%
logs = reset_logs(model)
for epoch in range(num_epochs):
    train_one_epoch(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        logging_frequency=logging_frequency,
        logs=logs,
        device=device,
        epoch_idx=epoch,
    )


# %%
save_logs(logs, "assignment/log", str(experimental_setting))
save_model(model, "assignment/models", str(experimental_setting))


