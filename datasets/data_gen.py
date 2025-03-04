import argparse
from collections import defaultdict, deque
import json
import random
import re
import math
import os
from typing import List, Dict, Any, Tuple
import time

class GenDataset:
    def __init__(self, data_dir: str):
        with open(data_dir, 'r') as file:
            self.original_data = json.load(file)
        self.generated_data = []

    def _generate_random_dag(self, num_nodes: int) -> Dict[int, List[int]]:
        dag = {i: [] for i in range(num_nodes)}
        node_indices = list(range(num_nodes))
        random.shuffle(node_indices)

        for i in range(1, num_nodes):
            parent = random.choice(node_indices[:i])
            child = node_indices[i]
            dag[parent].append(child)

        extra_edges = random.randint(0, num_nodes - 1)
        for _ in range(extra_edges):
            parent_idx = random.randint(0, num_nodes - 2)
            child_idx = random.randint(parent_idx + 1, num_nodes - 1)
            parent = node_indices[parent_idx]
            child = node_indices[child_idx]
            if child not in dag[parent]:
                dag[parent].append(child)

        return dag

    def is_same_magnitude(self, combined_value, num1):
        if combined_value <= 0 or num1 <= 0:
            return False
        magnitude_combined = math.floor(math.log10(combined_value))
        magnitude_num1 = math.floor(math.log10(num1))
        return abs(magnitude_combined - magnitude_num1) <= 1

    def _compute_multi_relation(self, parent_answers: List[float], child_vals: List[float]) -> Tuple[float, str]:
        combined_value = sum(parent_answers)
        num1 = child_vals if child_vals else 0.0
        x = num1 - combined_value
        desc = f"The sum of the answers of the parent nodes plus ({x:.2f}) gives the unknown number of the question stem for this node."
        bool_ok = self.is_same_magnitude(abs(combined_value), num1)
        return num1, x, bool_ok

    def _generate_final_integration(self, answers: List[float]) -> Tuple[str, float]:
        str_mut = '*'.join([f'Answer[{i}]' for i in range(len(answers))])
        final_question = (
            f"Please calculate the value of {str_mut}. Conclude the answer by stating 'The answer is therefore \\boxed{{[ANSWER]}}.'"
        )
        final_answer = math.prod(answers)
        return final_question, final_answer

    def generate_complex_questions(self, N: int, complex: int, save_path: str):
        all_generated_data = []
        Q_ID_all = []
        index = 0

        while True:
            index += 1
            print(index)
            n = complex

            dag = self._generate_random_dag(n)

            if len(self.original_data) < n or index > N:
                print("Not enough data to sample. Breaking loop.")
                break

            sampled_items = random.sample(self.original_data, n)
            Q_ID = [sampled_items[i]['question_id'] for i in range(len(sampled_items))]

            if sorted(Q_ID) in [sorted(q) for q in Q_ID_all]:
                continue

            Q_ID_all.append(Q_ID)
            source = ' '.join([item['type'] for item in sampled_items])

            node_info = {
                i: {
                    "source": sampled_items[i]['type'],
                    "question_id": sampled_items[i]['question_id'],
                    "question_vals": sampled_items[i]['q_vals'],
                    "answer_val": float(sampled_items[i]['answer_number']),
                    "problem": sampled_items[i]['problem'],
                    "problem_UNK": sampled_items[i]['problem_UNK'],
                    "in_edges": [],
                    "out_edges": []
                }
                for i in range(n)
            }

            for parent in dag:
                for child in dag[parent]:
                    node_info[child]["in_edges"].append(parent)
                    node_info[parent]["out_edges"].append(child)

            edge_descriptions = []
            for child in range(n):
                parents = node_info[child]["in_edges"]
                if not parents:
                    node_info[child]['problem_UNK'] = re.sub(
                        r'UNK\(a constant can be caculuted by other answers\)',
                        f"{node_info[child]['question_vals']}",
                        node_info[child]['problem_UNK']
                    ) + f". The answer is recorded as Answer[{child}]"
                    edge_descriptions.append(" ")
                    continue

                parent_answers = [node_info[p]["answer_val"] for p in parents]
                new_val, x, bool_ok = self._compute_multi_relation(parent_answers, node_info[child]["question_vals"])
                node_info[child]["question_vals"] = [new_val]

                combined_desc = f"a constant calculated by adding the sum of Answer{parents} to the number ({x:.2f}). "
                edge_descriptions.append(combined_desc)
                node_info[child]['problem_UNK'] = re.sub(
                    r'UNK\(a constant can be caculuted by other answers\)',
                    f"UNK_{child}({combined_desc})",
                    node_info[child]['problem_UNK']
                ) + f". The answer is recorded as Answer[{child}]"

            all_answers = [node_info[i]["answer_val"] for i in range(n)]
            all_problem_texts = [f"{node_info[i]['problem_UNK']}\n" for i in range(n)]

            in_degree = defaultdict(int)
            for node, edges in dag.items():
                for neighbor in edges:
                    in_degree[neighbor] += 1

            queue = deque([node for node in dag if in_degree[node] == 0])
            topological_order = []
            while queue:
                current = queue.popleft()
                topological_order.append(current)
                for neighbor in dag[current]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            sorted_problems = [all_problem_texts[node] for node in topological_order]
            gt_subtask = [all_answers[node] for node in topological_order]
            all_problem_texts_combined_sort = "\n".join(sorted_problems)
            all_problem_texts_combined = "\n".join(all_problem_texts)
            final_question_core_text, final_answer_val = self._generate_final_integration(all_answers)
            gt_subtask.append(final_answer_val)

            final_question_text = (
                "The following is a complex question composed of multiple sub-questions:\n\n"
                f"{all_problem_texts_combined}\n"
                f"Please use the answers to the above questions to perform the following calculations:\n{final_question_core_text}"
            )

            final_question_text_sort = (
                "The following is a complex question composed of multiple sub-questions:\n\n"
                f"{all_problem_texts_combined_sort}\n"
                f"Please use the answers to the above questions to perform the following calculations:\n{final_question_core_text}"
            )
            sorted_problems.append(final_question_core_text)

            complex_question_item = {
                'source': source,
                'Q_ID': Q_ID,
                "complexity": n,
                "dag": dag,
                "node_info": node_info,
                "edge_descriptions": edge_descriptions,
                "problem_text": final_question_text,
                "problem_text_sort": final_question_text_sort,
                "answer_number": f"{final_answer_val}",
                "gt_plan": f"{sorted_problems}",
                "gt_subtask": gt_subtask,
                "unit": "",
            }

            all_generated_data.append(complex_question_item)

        with open(save_path, 'w', encoding='utf-8') as out_f:
            json.dump(all_generated_data, out_f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate complex math questions dataset")
    parser.add_argument("-n", "--num_questions", type=int, required=True, help="Number of questions to generate")
    parser.add_argument("-c", "--complexity", type=int, required=True, help="Complexity level of questions")
    parser.add_argument("-o", "--output", type=str, help="Output file path (optional)")

    args = parser.parse_args()

    sub_data = "datasets/sub_question/math_test.json"
    gen = GenDataset(sub_data)

    default_filename = f"datasets/mixed/mix_math_test_{args.complexity}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
    output_file = args.output if args.output else default_filename

    gen.generate_complex_questions(args.num_questions, args.complexity, output_file)