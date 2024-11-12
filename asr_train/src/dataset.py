import json
import yaml
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset, DatasetDict, load_dataset, Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

class CommonVoice:
    def __init__(self, cfg_path):
       
        with open(cfg_path, 'r') as ymlfile:
           self._cfg = yaml.safe_load(ymlfile)
    
        self._split_ratio = self._cfg['split_ratio']
        self._sampling_rate = self._cfg['sampling_rate']  
   
    def transform(self, df, split=True):
        ds = load_dataset("audiofolder", data_files=list(df['filename']))
        ds = ds["train"]
        ds = ds.cast_column("audio", Audio(sampling_rate=self._sampling_rate))
        ds = ds.add_column("text", df["processed_text"])
        ds = ds.add_column("file", df["filename"])
        if split:
            split_ds = ds.train_test_split(test_size=self._split_ratio)
        return split_ds

    @staticmethod
    def convert_to_uppercase(batch):
        batch['text'] = batch['text'].upper()
        return batch
    
    @staticmethod
    def add_start_end_token(batch):
        batch['text'] = "<s>"+batch['text']+"</s>"
        return batch
    
    @staticmethod
    def prepare_dataset(batch, processor):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
