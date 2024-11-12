import yaml
import numpy as np

from evaluate import load
from transformers import Wav2Vec2ForCTC, Trainer
from transformers import TrainingArguments

class ModelTrainer:
    def __init__(self, cfg_path, ds, processor, data_collator):
        self.ds = ds
        self.processor = processor
        self.data_collator = data_collator
        self.model = self.load_model()
        self.wer_metric = load("wer")
        
        with open(cfg_path, 'r') as ymlfile:
           self._cfg = yaml.safe_load(ymlfile)
        
        config = self._cfg['training_args']
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            group_by_length=config['group_by_length'],
            per_device_train_batch_size=config['per_device_train_batch_size'],
            evaluation_strategy=config['evaluation_strategy'],
            max_steps=config['max_steps'],
            fp16=config['fp16'],
            gradient_checkpointing=config['gradient_checkpointing'],
            save_steps=config['save_steps'],
            eval_steps=config['eval_steps'],
            logging_steps=config['logging_steps'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            warmup_steps=config['warmup_steps'],
            save_total_limit=config['save_total_limit'],
            load_best_model_at_end=config['load_best_model_at_end'],
            metric_for_best_model=config['metric_for_best_model'],
            greater_is_better=config['greater_is_better']  
        )
        self._training_args = training_args
       
    def compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    
    def load_model(self):
        self.model =  Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        self.model.freeze_feature_extractor()
        return self.model

    def run(self):
        trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator,
            args=self._training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self.ds["train"],
            eval_dataset=self.ds["test"],
            tokenizer=self.processor.feature_extractor,
        )
        trainer.train()
        return trainer



