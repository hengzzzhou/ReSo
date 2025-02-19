import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments, Trainer
)

from model.modeling_qwen2_rm import Qwen2ForProcessRewardModel


class ScoreModelCustomizedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels'],
            return_dict=True
        )
        if return_outputs:
            return outputs.loss, outputs.logits
        return outputs.loss


class ScoreModelTrainingPipeline(object):
    def __init__(self, arguments):
        self.arguments = arguments
        self.prepare_tokenizer()
        self.prepare_model()
        self.prepare_dataset()
        self.prepare_parameter()
        self.start_training()

    def _preprocess_example(self, example):
        conversation = self.tokenizer.apply_chat_template([
            {'role': 'user', 'content': example['instruction']},
            {'role': 'assistant', 'content': example['answer']},
        ], tokenize=False)
        inputs = self.tokenizer(conversation)
        label_position = len(inputs['input_ids'])
        inputs['input_ids'].append(self.target_token)
        inputs['input_ids'].extend([self.tokenizer.pad_token_id] * self.arguments.context_length)
        inputs['input_ids'] = inputs['input_ids'][:self.arguments.context_length]
        inputs['attention_mask'].append(0)
        inputs['attention_mask'].extend([0] * self.arguments.context_length)
        inputs['attention_mask'] = inputs['attention_mask'][:self.arguments.context_length]
        inputs['labels'] = [-100] * len(inputs['input_ids'])
        if label_position < len(inputs['input_ids']):
            inputs['labels'][label_position] = example['label']
        return inputs

    def prepare_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.arguments.base_model)
        self.target_token = self.tokenizer.encode('<extra_0>')[0]
        self.collator = DataCollatorWithPadding(self.tokenizer)
        print(self.tokenizer)

    def prepare_model(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.arguments.lora_rank,
            target_modules=['q_proj', 'v_proj', 'score.0', 'score.2'],
            lora_alpha=self.arguments.lora_alpha,
            lora_dropout=self.arguments.lora_dropout
        )
        model = get_peft_model(
            model=Qwen2ForProcessRewardModel.from_pretrained(
                self.arguments.base_model,
                torch_dtype=torch.bfloat16
            ),
            peft_config=lora_config
        )
        self.model = model
        print(self.model)

    def prepare_dataset(self):
        dataset = load_dataset('json', data_files={
            'train': self.arguments.data_train,
            'test': self.arguments.data_test
        })
        dataset = dataset.map(self._preprocess_example)
        dataset['train'] = dataset['train'].remove_columns(['instruction', 'answer', 'label'])
        dataset['test'] = dataset['test'].remove_columns(['instruction', 'answer', 'label'])
        self.dataset = dataset
        print(self.dataset)

    def prepare_parameter(self):
        parameters = TrainingArguments(
            run_name=self.arguments.run_name,
            output_dir=self.arguments.save_path,
            logging_dir=self.arguments.logging_path,
            per_device_train_batch_size=self.arguments.batch_size_train,
            per_device_eval_batch_size=self.arguments.batch_size_eval,
            gradient_accumulation_steps=self.arguments.grad_accumulation,
            dataloader_num_workers=self.arguments.num_workers,
            num_train_epochs=self.arguments.num_epochs,
            learning_rate=self.arguments.learning_rate,
            weight_decay=self.arguments.weight_decay,
            logging_strategy='steps',
            logging_steps=self.arguments.logging_steps,
            eval_strategy='steps',
            eval_steps=self.arguments.eval_steps,
            save_strategy='epoch',
            ddp_find_unused_parameters=False,
            bf16=True
        )
        self.parameters = parameters
        print(self.parameters)

    def start_training(self):
        trainer = ScoreModelCustomizedTrainer(
            model=self.model,
            args=self.parameters,
            data_collator=self.collator,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test']
        )
        trainer.train()
        trainer.save_model()
        trainer.save_state()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='/path/to/your/model')
    parser.add_argument('--data_train', type=str, default='./dataset/score_model_simple_train.jsonl')
    parser.add_argument('--data_test', type=str, default='./dataset/score_model_simple_test.jsonl')
    parser.add_argument('--run_name', type=str, default='sft-qwen2.5-math-prm-7b-score-model-simple-bs128')
    parser.add_argument('--save_path', type=str, default='/path/to/your/checkpoint')
    parser.add_argument('--logging_path', type=str, default='/path/to/your/log')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--context_length', type=int, default=4096)
    parser.add_argument('--batch_size_train', type=int, default=2)
    parser.add_argument('--batch_size_eval', type=int, default=16)
    parser.add_argument('--grad_accumulation', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    arguments = parser.parse_args()
    ScoreModelTrainingPipeline(arguments)


if __name__ == '__main__':
    main()
