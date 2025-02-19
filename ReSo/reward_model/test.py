import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

from model.modeling_qwen2_rm import Qwen2ForProcessRewardModel


class ScoreModelTestingPipeline(object):
    def __init__(self, arguments):
        self.arguments = arguments
        self.prepare_tokenizer()
        self.prepare_model()
        self.prepare_dataset()
        self.start_testing()

    def _preprocess_example(self, example):
        conversation = self.tokenizer.apply_chat_template([
            {'role': 'user', 'content': example['instruction']},
            {'role': 'assistant', 'content': example['answer']},
        ], tokenize=False)
        inputs = self.tokenizer(conversation)
        inputs['input_ids'].append(self.target_token)
        inputs['attention_mask'].append(0)
        return inputs

    def prepare_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.arguments.base_model)
        self.target_token = self.tokenizer.encode('<extra_0>')[0]
        print(self.tokenizer)

    def prepare_model(self):
        model = Qwen2ForProcessRewardModel.from_pretrained(
            self.arguments.base_model,
            device_map=self.arguments.device,
            torch_dtype=torch.bfloat16
        ).eval()
        if self.arguments.lora_model:
            model.load_adapter(self.arguments.lora_model)
        self.model = model
        print(self.model)

    def prepare_dataset(self):
        dataset = load_dataset('json', data_files={'test': self.arguments.data_path})
        dataset = dataset.map(self._preprocess_example)
        self.dataset = dataset['test']

    def start_testing(self):
        probability_list = []
        prediction_list = []
        label_list = []
        with torch.no_grad():
            for example in tqdm(self.dataset):
                input_ids = torch.tensor([example['input_ids']]).to(self.model.device)
                logits = self.model(input_ids, return_dict=True).logits
                probability = F.softmax(logits, dim=-1)[0, -1, 1].item()
                prediction = int(probability >= 0.5)
                probability_list.append(probability)
                prediction_list.append(prediction)
                label_list.append(example['label'])
        metrics = {
            'log_loss': log_loss(label_list, probability_list),
            'acc_score': accuracy_score(label_list, prediction_list),
            'auc_score': roc_auc_score(label_list, probability_list)
        }
        print(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='/path/to/base/model')
    parser.add_argument('--lora_model', type=str, default='/path/to/lora/model')
    parser.add_argument('--data_path', type=str, default='./dataset/score_model_simple_test.jsonl')
    parser.add_argument('--device', type=str, default='cuda:0')
    arguments = parser.parse_args()
    ScoreModelTestingPipeline(arguments)


if __name__ == '__main__':
    main()
