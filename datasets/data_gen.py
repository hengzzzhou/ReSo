from collections import defaultdict, deque
import json
import random
import re
import math
import os
from typing import List, Dict, Any, Tuple
import time
from urllib import response
from openai import OpenAI

class GenDataset:
    def __init__(self, data_dir: str):
        """
        初始化时，加载多个子数据集并进行基础处理。
        :param data_dir: 存放各个子数据集(如thermo.json, calculus.json等)的目录路径
        """
        self.data_dir = data_dir
        # original_data 将存放所有子数据集中混合的题目
        
        
        #self.original_data = self._load_all_subdatasets()
        
        with open("/cpfs01/shared/mabasic/zhouheng/Dataset/MATH/test_mix/math_test_gpt4o.json",'r') as file:
            data=json.load(file)
        self.original_data = data
        self.generated_data = []

    def _generate_random_dag(self, num_nodes: int) -> Dict[int, List[int]]:
        """
        与之前类似的随机生成有向无环图的方法。
        """
        dag = {i: [] for i in range(num_nodes)}
        node_indices = list(range(num_nodes))
        random.shuffle(node_indices)

        # 保证最少的连通
        for i in range(1, num_nodes):
            parent = random.choice(node_indices[:i])
            child = node_indices[i]
            dag[parent].append(child)

        # 额外随机加边
        extra_edges = random.randint(0, num_nodes - 1)
        for _ in range(extra_edges):
            parent_idx = random.randint(0, num_nodes - 2)
            child_idx = random.randint(parent_idx + 1, num_nodes - 1)
            parent = node_indices[parent_idx]
            child = node_indices[child_idx]
            if child not in dag[parent]:
                dag[parent].append(child)

        return dag
    def is_same_magnitude(self,combined_value, num1):
        # 防止取对数时遇到非正数
        if combined_value <= 0 or num1 <= 0:
            return False
        
        # 计算数量级
        magnitude_combined = math.floor(math.log10(combined_value))
        magnitude_num1 = math.floor(math.log10(num1))
        
        # 判断数量级差异
        return abs(magnitude_combined - magnitude_num1) <= 1
    def _compute_multi_relation(self, parent_answers: List[float], child_vals: List[float]) -> Tuple[float, str]:
        """
        根据父节点的答案生成一个单一的数值，替换子节点的 num1。
        每个子问题只替换一个参数，用所有父节点的答案通过简单运算得到的值。
        """
        # 用所有父节点答案进行计算 (例如：求和)
        combined_value = sum(parent_answers)  # 可换为其他运算，如乘积
        num1 = child_vals if child_vals else 0.0  # 如果没有参数，则默认为 0
        x = num1 - combined_value
        # 生成自然语言描述
        desc = f"The sum of the answers of the parent nodes plus ({x:.2f}) gives the unknown number of the question stem for this node."
        bool_ok=self.is_same_magnitude(abs(combined_value),num1)
        # 返回替换后的参数和描述
        return num1, x, bool_ok


    def _generate_final_integration(self, answers: List[float]) -> Tuple[str, float]:
        """
        在最后一步进行并联整合运算。例子里做积分或简单加法。
        """

        str_mut='*'.join([f'Answer[{i}]' for i in range(len(answers))])
        final_question = (
            f"""Please calculate the value of {str_mut}. Conclude the answer by stating 'The answer is therefore \\boxed{{[ANSWER]}}.'"""
        )
        final_answer = math.prod(answers)
        return final_question, final_answer

    
    def generate_complex_questions(self, N: int, complex: int, save_path: str):

        all_generated_data = []  # 存储所有生成的问题
        Q_ID_all=[]
        index=0
        # skip_indices = [1, 2, 4, 5, 7, 8, 11, 12, 14, 15, 16, 22, 23, 28, 31, 32, 33, 42, 43, 46, 48, 49, 51, 52, 58, 60, 63, 65, 67, 68]
        while True:
            index+=1
            print(index)
            #n=random.randint(2, 6)
            n=7
            # parsed_data = []
            # for item in self.original_data:
            #     ds_label, qid, q_vals, ans_val = self._parse_item(item)
            #     problem_text = item.get("problem_text", "")
            #     parsed_data.append((ds_label, qid, q_vals, ans_val, problem_text))

            # 生成随机DAG
            dag = self._generate_random_dag(n)

            # 随机选取 n 条
            if len(self.original_data) < n or index>100:
                print("Not enough data to sample. Breaking loop.")
                print(index)
                break
            sampled_items = random.sample(self.original_data, n)
            
            Q_ID = [sampled_items[i]['question_id'] for i in range(len(sampled_items))]
            
            if sorted(Q_ID) not in [sorted(q) for q in Q_ID_all]:
                Q_ID_all.append(Q_ID)
            else:
                continue
            source=''
            # 构建 node_info
            node_info = {}
            for i in range(n):
                
                node_info[i] = {
                    "source": sampled_items[i]['type'],
                    "question_id": sampled_items[i]['question_id'],
                    "question_vals": sampled_items[i]['q_vals'],
                    "answer_val": float(sampled_items[i]['answer_number']),
                    "problem": sampled_items[i]['problem'],
                    "problem_UNK": sampled_items[i]['problem_UNK'],
                    "in_edges": [],
                    "out_edges": []
                }
                source=source+sampled_items[i]['type']+' '

            # 建立 in_edges / out_edges
            for parent in dag:
                for child in dag[parent]:
                    node_info[child]["in_edges"].append(parent)
                    node_info[parent]["out_edges"].append(child)

            # 更新题干和边描述
            edge_descriptions = []
            for child in range(n):
                parents = node_info[child]["in_edges"]
                if not parents:
                    node_info[child]['problem_UNK'] = re.sub(r'UNK\(a constant can be caculuted by other answers\)', f"{node_info[child]['question_vals']}", node_info[child]['problem_UNK'])+f". The answer is recorded as Answer[{child}]"
                    edge_descriptions.append(" ")
                    continue
                parent_answers = [node_info[p]["answer_val"] for p in parents]
                new_val, x ,bool_ok= self._compute_multi_relation(parent_answers, node_info[child]["question_vals"])
                # if not bool_ok:
                #     break
                node_info[child]["question_vals"] = [new_val]

                combined_desc = f"a constant calculated by adding the sum of Answer{parents} to the number ({x:.2f}). "
                edge_descriptions.append(combined_desc)
                node_info[child]['problem_UNK'] = re.sub(r'UNK\(a constant can be caculuted by other answers\)', f"UNK_{child}({combined_desc})", node_info[child]['problem_UNK'])+f". The answer is recorded as Answer[{child}]"
            # if not bool_ok:
            #     continue
            # 汇总题干
            all_answers = [node_info[i]["answer_val"] for i in range(n)]
            all_problem_texts = [
                # f"Question {i} (QID={node_info[i]['question_id']} from {node_info[i]['dataset_label']}):\n"
                f"{node_info[i]['problem_UNK']}\n"
                for i in range(n)
            ]
            in_degree = defaultdict(int)
            for node, edges in dag.items():
                for neighbor in edges:
                    in_degree[neighbor] += 1

            # 初始化队列：找到入度为 0 的节点
            queue = deque([node for node in dag if in_degree[node] == 0])

            # 拓扑排序
            topological_order = []
            while queue:
                current = queue.popleft()
                topological_order.append(current)
                for neighbor in dag[current]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            # 根据拓扑顺序重新排列问题列表
            sorted_problems = [all_problem_texts[node] for node in topological_order]
            gt_subtask = [all_answers[node] for node in topological_order]
            all_problem_texts_combined_sort="\n".join(sorted_problems)
            all_problem_texts_combined = "\n".join(all_problem_texts)
            edge_texts_combined = "\n".join(edge_descriptions)
            all_answers = [node_info[i]["answer_val"] for i in range(n)]
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
                'source':source,
                'Q_ID':Q_ID,
                "complexity": n,
                "dag": dag,
                "node_info": node_info,
                "edge_descriptions": edge_descriptions,
                "problem_text": final_question_text,
                "problem_text_sort": final_question_text_sort,
                "answer_number": f"{final_answer_val}",
                "gt_plan":f"{sorted_problems}",
                "gt_subtask":gt_subtask,
                "unit": "",
            }

            all_generated_data.append(complex_question_item)
            # self.original_data = [item for item in self.original_data if item not in sampled_items]

        # 写入结果
        with open(save_path, 'w', encoding='utf-8') as out_f:
            json.dump(all_generated_data, out_f, ensure_ascii=False, indent=2)




    

if __name__ == "__main__":
    data_directory = "/cpfs01/shared/mabasic/zhouheng/Dataset/MATH/test"
    gen = GenDataset(data_directory)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    gen.generate_complex_questions(81,3, f"/cpfs01/shared/mabasic/zhouheng/Dataset/MATH/test_mix/mix_math_test_7_{current_time}.json")

