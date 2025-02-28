import asyncio
import copy
import math
import os
import random
import json
import time
import logging

import torch
import torch.nn.functional as F
import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer

# Import custom modules
from ReSo.model.modeling_qwen2_rm import Qwen2ForProcessRewardModel
from datasets.get_answer import parse_math_answer
from ReSo.agent_graph.embedding_utils import cosine_similarity, get_embedding_async
from ReSo.agent_graph.reward_utils import reward_model
from ReSo.agent_graph.logging_setup import setup_logging

# Ensure logging is set up
setup_logging()

class AgentGraph:
    """
    AgentGraph manages the agent pool, selects candidate agents based on generated prompts,
    and performs multi-agent inference using an MCTS-based approach.
    """
    def __init__(self, agent_pool, similarity_weight=1.0, reputation_weight=1.0, cost_weight=1.0,
                 threshold=0.5, exploration_const=1.0, top_k=10, mode="train", total_visit=0):
        self.agent_pool = agent_pool
        self.similarity_weight = similarity_weight
        self.reputation_weight = reputation_weight
        self.cost_weight = cost_weight
        self.threshold = threshold
        self.exploration_const = exploration_const
        self.top_k = top_k
        self.mode = mode
        self.total_visit = total_visit

        self.system_prompt = (
            "You are an AI assistant specialized in generating structured prompts for domain-specific experts in a multi-agent system. \n\n"
            "**Task:**  \n"
            "Given a subquestion, analyze its domain, required expertise, and problem complexity. Then, generate a structured prompt that precisely describes the expert’s role in solving the problem. The generated prompt will be used for vector-based similarity matching to select the most appropriate agent from an agent pool.\n\n"
            "**Prompt Format:**  \n"
            "\"You are a [Expert Type], highly skilled in [Specific Knowledge Areas]. Your task is to analyze the problem by first reviewing previously solved variables and solutions from other agents in the multi-agent system. Apply domain-specific knowledge to reason rigorously and provide a well-structured, logically sound answer. If calculations are required, show all steps. If problem decomposition is needed, outline a systematic approach. Ensure consistency with previous solutions in the multi-agent system and resolve any discrepancies when necessary. Your role is to assist in solving complex reasoning problems with precision and alignment with the broader system.\"\n\n"
            "**Instructions for Prompt Generation:**\n"
            "1. **Expert Type Selection**: Identify the most relevant expert type (e.g., MechanicsExpert, AlgebraExpert, ThermodynamicsExpert).\n"
            "2. **Specific Knowledge Areas**: Define the precise knowledge fields required to solve the problem.\n"
            "3. **Problem Scope & Complexity**: Determine whether the problem requires deep theoretical knowledge, numerical computation, or practical modeling.\n\n"
            "**Output:**  \n"
            "Provide only the generated prompt without additional explanations."
        )
        self.embedding_cache = {}
        oai_api_key = os.getenv('OAI_API_KEY')
        oai_base_url = os.getenv('OAI_BASE_URL')
        self.aclient = AsyncOpenAI(base_url=oai_base_url, api_key=oai_api_key)

        # Uncomment and modify if you need a tokenizer or reward model
        # self.tokenizer = AutoTokenizer.from_pretrained('path_to_model')
        # self.special_token = self.tokenizer.encode('<extra_0>')[0]
        # self.rm = Qwen2ForProcessRewardModel.from_pretrained('path_to_model', device_map='auto', torch_dtype=torch.bfloat16).eval()

    # ... (rest of the AgentGraph class remains, using the functions from the utility modules)


    async def select_agent_subset(self, subquestion):

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.aclient.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Subquestion: {subquestion}"}
                    ],
                    temperature=0.2
                )
                answer = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)
        target_profile = answer
        target_profile_emb = await get_embedding_async(self.aclient, target_profile, self.embedding_cache)

        tasks = [self.evaluate_agent(agent, target_profile_emb) for agent in self.agent_pool]
        candidates = await asyncio.gather(*tasks)     
        #random.shuffle(candidates)
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [agent for agent, score in candidates[:self.top_k]]
        print(f"#########{self.top_k}")
        name = [agent.agent_id for agent in top_candidates]
        logging.info(f"Selected candidate agent {name} from {len(top_candidates)} candidates out of {len(self.agent_pool)} agents")
        return top_candidates

    async def evaluate_agent(self, agent, target_profile_emb):
        agent_prompt_emb = await asyncio.wait_for(get_embedding_async(self.aclient,agent.prompt,self.embedding_cache), timeout=600)
        sim_profile = cosine_similarity(target_profile_emb, agent_prompt_emb)
        sim = self.similarity_weight * (sim_profile - 0.5) * 2  
        alpha, beta = 1, 1
        if agent.scoring_history:
            avg_reward = (sum(agent.scoring_history) + alpha) / (len(agent.scoring_history) + beta)
        else:
            avg_reward = alpha / beta
        reputation = self.reputation_weight * avg_reward

        if agent.inference_costs:
            cost = sum(agent.inference_costs) / len(agent.inference_costs)
        else:
            cost = 0
        cost_penalty = math.log1p(cost)  # log(1 + cost)

        final_score = (reputation - cost_penalty + sim) / 2

        visit_number = agent.visit + 1e-8
        exploration_term = self.exploration_const * math.sqrt(math.log(self.total_visit + 1.01) / visit_number)

        similarity_factor = 1 / (1 + math.exp(-10 * (sim - self.threshold)))  
        random_noise = random.uniform(-1e-5, 1e-5)
        ucb_score = (final_score + exploration_term) * similarity_factor + random_noise
        #sim=sim+random_noise
        logging.info(f"Agent {agent.agent_id}: sim={sim:.3f}, reputation={reputation:.3f}, "
                    f"cost={cost_penalty:.3f}, final_score={final_score:.3f}, ucb_score={ucb_score:.3f}")
        return (agent, ucb_score)

    async def agent_inference(self, agent, message):
        tries = 0
        while tries < 3:
            try:
                answer, cost,prompt_len,completion_len=await asyncio.wait_for(agent.inference(message),timeout=600)
                return answer, cost,prompt_len,completion_len
            except Exception as e:
                print(f"Error during execution of node {agent.agent_id}: {e}")
                tries += 1
        raise RuntimeError(f"Node {agent.agent_id} failed after 3 retries")

    async def mcts_inference_async(self, message, candidate_agents):

        tasks = [
            self.agent_inference(agent, message=[{"role": "system", "content": agent.prompt}] + copy.deepcopy(message))
            for agent in candidate_agents
        ]
        results = await asyncio.gather(*tasks)

        candidate_numeric_answers = []
        for agent, (answer, cost,prompt_len,completion_len) in zip(candidate_agents, results):
            numeric_value = parse_math_answer(answer)
            if numeric_value is None:
                continue 
            candidate_numeric_answers.append((agent, numeric_value, cost, answer,prompt_len,completion_len))

        if not candidate_numeric_answers:
            best_agent = candidate_agents[0]
            answer, cost ,prompt_len,completion_len= results[0]
            logging.info(f"[TEST MODE] All candidate answers could not be parsed as numbers. Defaulting to agent {best_agent.agent_id}, cost: {cost}, prompt_len: {prompt_len}, completion_len: {completion_len}")

            return best_agent, answer, cost

        groups = []
        tol = 0.05  
        for candidate in candidate_numeric_answers:
            agent, value, cost, answer,prompt_len,completion_len = candidate
            value = float(value)
            found_group = False
            for group in groups:
                rep = float(group["group_value"])
                rel_diff = abs(value - rep) / abs(rep) if rep != 0 else abs(value - rep)
                if rel_diff <= tol:
                    group["candidates"].append(candidate)
                    group["group_value"] = sum([float(cand[1]) for cand in group["candidates"]]) / len(group["candidates"])
                    found_group = True
                    break
            if not found_group:
                groups.append({"group_value": value, "candidates": [candidate]})
        #print(groups)
        best_group = max(groups, key=lambda g: len(g["candidates"]))
        final_numeric = best_group["group_value"]

        best_candidate = None
        best_diff = float('inf')
        best_cost = None
        best_candidate_answer = "None"
        for candidate in best_group["candidates"]:
            agent, value, cost, answer,prompt_len,completion_len = candidate
            diff = abs(float(value) - final_numeric)
            if diff < best_diff:
                best_diff = diff
                best_candidate = agent
                best_cost = cost
                best_candidate_answer = answer
                best_prompt_len=prompt_len
                best_completion_len=completion_len

        logging.info(f"[TEST MODE] Voted numerical answer: {final_numeric:.2f} (Number of candidates in group: {len(best_group['candidates'])}) "
             f"Final answer provided by agent {best_candidate.agent_id}. Cost: {best_cost}, "
             f"prompt_len: {best_prompt_len}, completion_len: {best_completion_len}")

        return best_candidate, best_candidate_answer, best_cost
    
    async def ReSo_training_llm(self, message, candidate_agents,gt_answer):

        tasks = [
                self.agent_inference(agent, message=[{"role": "system", "content": agent.prompt}] + copy.deepcopy(message))
                for agent in candidate_agents
            ]
        results = await asyncio.gather(*tasks)

        best_agent = None
        best_reward = -float('inf')
        best_reward_gt=-float('inf')
        best_answer = None
        best_cost = None
        for agent, (answer, cost,prompt_len,completion_len) in zip(candidate_agents, results):
            agent.inference_costs.append(cost)
            agent._update_dynamic_field("inference_costs", agent.inference_costs)
            pre=[entry['content'] for entry in message if entry['role'] == 'assistant']
            data = {
            "instruction": f"**Previous Steps**: {pre}\n\n{message[-1]['content']}",
            "answer": answer
        }
            data_test={
    "instruction": "**Previous Steps**: []\n\nyour current sub-question: A $4.0$-foot by $8$-foot rectangular piece of plywood will be cut into $4$ congruent rectangles with no wood left over and no wood lost due to the cuts. What is the positive difference, in feet, between the greatest possible perimeter of a single piece and the least possible perimeter of a single piece?. The answer is recorded as Answer[1]\n\n Conclude the answer by stating 'The answer is therefore \\boxed{[ANSWER]}'",
    "answer": "To solve this problem, we need to determine the possible dimensions of the congruent rectangles that can be obtained by cutting a 4-foot by 8-foot piece of plywood into 4 pieces, and then find the greatest and least possible perimeters of these rectangles.\n\n### Step 1: Determine Possible Dimensions\n\nThe area of the original plywood is \\(4 \\times 8 = 32\\) square feet. Since the plywood is cut into 4 congruent rectangles, each rectangle must have an area of \\(\\frac{32}{4} = 8\\) square feet.\n\nThe dimensions of a rectangle with area 8 can be any pair of factors of 8. The possible integer pairs are:\n- \\(1 \\times 8\\)\n- \\(2 \\times 4\\)\n\n### Step 2: Calculate Perimeters\n\nFor each pair of dimensions, calculate the perimeter of the rectangle:\n- For dimensions \\(1 \\times 8\\), the perimeter is \\(2(1 + 8) = 18\\) feet.\n- For dimensions \\(2 \\times 4\\), the perimeter is \\(2(2 + 4) = 12\\) feet.\n\n### Step 3: Find the Positive Difference\n\nThe greatest possible perimeter is 18 feet, and the least possible perimeter is 12 feet. The positive difference between these perimeters is:\n\\[ 18 - 12 = 6 \\text{ feet} \\]\n\nThe answer is therefore \\(\\boxed{6}\\)."
}
            conversation = self.tokenizer.apply_chat_template([
            {"role": "user", "content": data["instruction"]},
            {"role": "assistant", "content": data["answer"]}
        ], tokenize=False)
            inputs = self.tokenizer.encode(conversation)
            inputs.append(self.special_token)
            inputs = torch.tensor([inputs], device='cuda:0')
            with torch.no_grad():
                logits = self.rm(inputs, return_dict=True).logits
            reward_gt = F.softmax(logits, dim=-1)[0, -1, 1].item()
            
            #reward_gt = self.reward_model(answer, gt_answer)
            agent.scoring_history.append(reward_gt)
            agent._update_dynamic_field("scoring_history", agent.scoring_history)
            agent.visit += 1
            self.total_visit += 1
            agent._update_dynamic_field("visit", agent.visit)
            if reward_gt > best_reward:
                best_reward = reward_gt
                best_agent = agent
                best_answer = answer
                best_cost = cost
                best_prompt_len=prompt_len
                best_completion_len=completion_len
        logging.info(f"[MCTS] Final selection: agent {best_agent.agent_id}, reward={best_reward:.3f}, cost={best_cost}, "
             f"prompt_len: {best_prompt_len}, completion_len: {best_completion_len}")

        return best_agent, best_answer, best_reward, best_cost
    async def ReSo_training_rule_based(self, message, candidate_agents,gt_answer):

        tasks = [
                self.agent_inference(agent, message=[{"role": "system", "content": agent.prompt}] + copy.deepcopy(message))
                for agent in candidate_agents
            ]
        results = await asyncio.gather(*tasks)

        best_agent = None
        best_reward = -float('inf')
        best_answer = None
        best_cost = None
        for agent, (answer, cost,prompt_len,completion_len) in zip(candidate_agents, results):
            reward_gt = reward_model(answer, gt_answer)
            agent.inference_costs.append(cost)
            agent._update_dynamic_field("inference_costs", agent.inference_costs)
            agent.scoring_history.append(reward_gt)
            agent._update_dynamic_field("scoring_history", agent.scoring_history)
            agent.visit += 1
            self.total_visit += 1
            agent._update_dynamic_field("visit", agent.visit)
            if reward_gt > best_reward:
                best_reward = reward_gt
                best_agent = agent
                best_answer = answer
                best_cost = cost
                best_prompt_len=prompt_len
                best_completion_len=completion_len
        logging.info(f"[MCTS] Final choice: agent {best_agent.agent_id}, reward={best_reward:.3f}, cost={best_cost}, "
             f"prompt_len: {best_prompt_len}, completion_len: {best_completion_len}")

        return best_agent, best_answer, best_reward, best_cost

    async def build_agent_graph_test(self, question, task_graph,random_select):
        agent_graph = {"nodes": [], "cost": 0}
        pre_step = []
        for node in task_graph:
            subquestion = node
            start_time = time.time()
            if random_select is True:
                candidate_agents=random.sample(self.agent_pool,self.top_k)
            else:
                candidate_agents = await self.select_agent_subset(subquestion)
            elapsed_time = time.time() - start_time
            print(f"Time taken to select candidate agent: {elapsed_time:.6f} seconds")
            user_prompt = f"""your current sub-question: {subquestion}"""+"Conclude the answer by stating 'The answer is recorded as [what].The answer is therefore \\boxed{[ANSWER]}'"
            message = []
            message=message+pre_step
            message.append({"role": "user", "content": user_prompt})

            best_agent, answer, cost = await self.mcts_inference_async(message, candidate_agents)
            pre_step.append({"role": "user", "content":f"One subquestion: {subquestion}" })
            pre_step.append({"role": "assistant", "content": answer}) 
            elapsed_infer = time.time() - start_time
            print(f"Inference time: {elapsed_infer:.6f} seconds")
            agent_graph["cost"] += cost
            print(f"Agent {best_agent.agent_id} cost: {cost}")
            new_node = {"sub_task": node}
            new_node["selected_agent"] = best_agent.agent_id
            new_node["answer"] = answer
            agent_graph["nodes"].append(new_node)
        return agent_graph
    
    async def build_agent_graph_train(self, question, task_graph,random_select,gt_subtask):
        agent_graph = {"nodes": [], "cost": 0}
        pre_step = []
        i = 0
        for node in task_graph:
            subquestion = node
            start_time = time.time()
            if random_select is True:
                candidate_agents=random.sample(self.agent_pool,self.top_k)
            else:
                candidate_agents = await self.select_agent_subset(subquestion)
            elapsed_time = time.time() - start_time
            print(f"Time taken to select candidate agent: {elapsed_time:.6f} seconds")

            user_prompt = f"""your current sub-question: {subquestion}"""+"Conclude the answer by stating 'The answer is recorded as [what].The answer is therefore \\boxed{[ANSWER]}'"
            message = []
            message=message+pre_step
            message.append({"role": "user", "content": user_prompt})

            best_agent, answer,reward, cost = await self.ReSo_training_rule_based(message, candidate_agents,gt_subtask[i])
            pre_step.append({"role": "user", "content":f"One subquestion: {subquestion}" })
            pre_step.append({"role": "assistant", "content": answer}) 
            elapsed_infer = time.time() - start_time
            print(f"Inference time: {elapsed_infer:.6f} seconds")
            agent_graph["cost"] += cost
            print(f"Agent {best_agent.agent_id} cost: {cost}")
            new_node = {"sub_task": node}
            new_node["agent_score"] = reward
            new_node["selected_agent"] = best_agent.agent_id
            new_node["answer"] = answer
            agent_graph["nodes"].append(new_node)
            i += 1
            if reward == 0:
                break
        return agent_graph
