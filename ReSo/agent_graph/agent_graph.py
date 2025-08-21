"""Agent Graph Module for Multi-Agent Reasoning System.

This module implements the AgentGraph class which manages agent pools, performs
agent selection based on problem requirements, and coordinates multi-agent inference
using MCTS-based approach for mathematical problem solving.
"""

import asyncio
import copy
import math
import os
import random
import json
import time
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

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
    """Multi-Agent Graph for Mathematical Problem Solving.
    
    AgentGraph manages a pool of specialized agents and coordinates their selection
    and inference for solving complex mathematical problems. It uses MCTS-based
    approach with Upper Confidence Bound (UCB) scoring for agent selection.
    
    Attributes:
        agent_pool: List of available agents
        similarity_weight: Weight for similarity scoring in agent selection
        reputation_weight: Weight for agent reputation in scoring
        cost_weight: Weight for cost consideration in agent selection
        threshold: Similarity threshold for agent filtering
        exploration_const: Exploration constant for UCB scoring
        top_k: Number of top agents to select for inference
        mode: Operating mode ('train' or 'test')
        total_visit: Total number of agent visits across all agents
    """
    
    def __init__(self, agent_pool: List[Any], similarity_weight: float = 1.0, 
                 reputation_weight: float = 1.0, cost_weight: float = 1.0,
                 threshold: float = 0.5, exploration_const: float = 1.0, 
                 top_k: int = 10, mode: str = "train", total_visit: int = 0):
        """Initialize AgentGraph with configuration parameters.
        
        Args:
            agent_pool: List of available agents for problem solving
            similarity_weight: Weight for similarity scoring (default: 1.0)
            reputation_weight: Weight for agent reputation (default: 1.0)
            cost_weight: Weight for cost consideration (default: 1.0)
            threshold: Similarity threshold for filtering (default: 0.5)
            exploration_const: UCB exploration constant (default: 1.0)
            top_k: Number of top agents to select (default: 10)
            mode: Operating mode - 'train' or 'test' (default: 'train')
            total_visit: Initial total visit count (default: 0)
        """
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

        # Initialize tokenizer and reward model (uncomment if needed)
        # self.tokenizer = AutoTokenizer.from_pretrained('path_to_model')
        # self.special_token = self.tokenizer.encode('<extra_0>')[0]
        # self.rm = Qwen2ForProcessRewardModel.from_pretrained(
        #     'path_to_model', device_map='auto', torch_dtype=torch.bfloat16
        # ).eval()


    async def select_agent_subset(self, subquestion: str) -> List[Any]:
        """Select a subset of agents most suitable for solving a subquestion.
        
        This method generates a target expert profile for the subquestion,
        evaluates all agents against this profile, and returns the top-k
        candidates based on UCB scoring.
        
        Args:
            subquestion: The mathematical subquestion to solve
            
        Returns:
            List of top-k candidate agents for the subquestion
        """

        # Generate target expert profile with retry logic
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
                target_profile = response.choices[0].message.content.strip()
                break
            except Exception as e:
                logging.warning(f"Profile generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1)
        # Get embedding for target profile and evaluate all agents
        target_profile_emb = await get_embedding_async(
            self.aclient, target_profile, self.embedding_cache
        )
        
        evaluation_tasks = [
            self.evaluate_agent(agent, target_profile_emb) for agent in self.agent_pool
        ]
        candidates = await asyncio.gather(*evaluation_tasks)
        
        # Sort candidates by UCB score and select top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [agent for agent, score in candidates[:self.top_k]]
        
        selected_names = [agent.agent_id for agent in top_candidates]
        logging.info(
            f"Selected {len(top_candidates)} candidate agents {selected_names} "
            f"from pool of {len(self.agent_pool)} agents"
        )
        return top_candidates

    async def evaluate_agent(self, agent: Any, target_profile_emb: np.ndarray) -> Tuple[Any, float]:
        """Evaluate an agent's suitability for a task using UCB scoring.
        
        The evaluation considers:
        - Similarity between agent prompt and target profile
        - Agent's historical performance (reputation)
        - Agent's cost efficiency
        - Exploration bonus for less-visited agents
        
        Args:
            agent: The agent to evaluate
            target_profile_emb: Embedding of the target expert profile
            
        Returns:
            Tuple of (agent, ucb_score)
        """
        # Calculate similarity score
        agent_prompt_emb = await asyncio.wait_for(
            get_embedding_async(self.aclient, agent.prompt, self.embedding_cache), 
            timeout=600
        )
        sim_profile = cosine_similarity(target_profile_emb, agent_prompt_emb)
        similarity_score = self.similarity_weight * (sim_profile - 0.5) * 2
        
        # Calculate reputation score using Beta distribution
        alpha, beta = 1, 1  # Prior parameters
        if agent.scoring_history:
            avg_reward = (sum(agent.scoring_history) + alpha) / (len(agent.scoring_history) + beta)
        else:
            avg_reward = alpha / beta
        reputation_score = self.reputation_weight * avg_reward
        
        # Calculate cost penalty
        if agent.inference_costs:
            avg_cost = sum(agent.inference_costs) / len(agent.inference_costs)
        else:
            avg_cost = 0
        cost_penalty = math.log1p(avg_cost)  # log(1 + cost) for numerical stability
        
        # Combine scores
        base_score = (reputation_score - cost_penalty + similarity_score) / 2
        
        # Add exploration term (UCB)
        visit_count = agent.visit + 1e-8  # Avoid division by zero
        exploration_bonus = self.exploration_const * math.sqrt(
            math.log(self.total_visit + 1.01) / visit_count
        )
        
        # Apply similarity threshold with sigmoid
        similarity_threshold = 1 / (1 + math.exp(-10 * (similarity_score - self.threshold)))
        
        # Final UCB score with small random noise for tie-breaking
        random_noise = random.uniform(-1e-5, 1e-5)
        ucb_score = (base_score + exploration_bonus) * similarity_threshold + random_noise
        
        logging.debug(
            f"Agent {agent.agent_id}: similarity={similarity_score:.3f}, "
            f"reputation={reputation_score:.3f}, cost_penalty={cost_penalty:.3f}, "
            f"base_score={base_score:.3f}, ucb_score={ucb_score:.3f}"
        )
        
        return (agent, ucb_score)

    async def agent_inference(self, agent: Any, message: List[Dict[str, str]]) -> Tuple[str, float, int, int]:
        """Perform inference with an agent with retry logic.
        
        Args:
            agent: The agent to perform inference with
            message: The conversation messages for inference
            
        Returns:
            Tuple of (answer, cost, prompt_length, completion_length)
            
        Raises:
            RuntimeError: If agent fails after maximum retries
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(agent.inference(message), timeout=600)
                answer, cost, prompt_len, completion_len = result
                return answer, cost, prompt_len, completion_len
            except Exception as e:
                logging.warning(f"Agent {agent.agent_id} inference attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Agent {agent.agent_id} failed after {max_retries} retries")

    async def reso_testing(self, message: List[Dict[str, str]], candidate_agents: List[Any]) -> Tuple[Any, str, float]:
        """Perform testing inference using voting mechanism.
        
        Multiple candidate agents solve the problem independently, and the final
        answer is determined by majority voting among numerically similar solutions.
        
        Args:
            message: The conversation messages for inference
            candidate_agents: List of candidate agents for inference
            
        Returns:
            Tuple of (best_agent, best_answer, cost)
        """

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
    

    async def reso_training_rule_based(self, message: List[Dict[str, str]], 
                                      candidate_agents: List[Any], 
                                      gt_answer: Union[str, float]) -> Tuple[Any, str, float, float]:
        """Perform training inference with rule-based reward calculation.
        
        All candidate agents solve the problem, and rewards are calculated based
        on comparison with ground truth answers using rule-based metrics.
        
        Args:
            message: The conversation messages for inference
            candidate_agents: List of candidate agents for inference
            gt_answer: Ground truth answer for reward calculation
            
        Returns:
            Tuple of (best_agent, best_answer, best_reward, cost)
        """

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
    async def reso_training_llm(self, message: List[Dict[str, str]], 
                               candidate_agents: List[Any], 
                               gt_answer: Union[str, float]) -> Tuple[Any, str, float, float]:
        """Perform training inference with LLM-based reward calculation.
        
        All candidate agents solve the problem, and rewards are calculated using
        a learned reward model for more nuanced evaluation.
        
        Args:
            message: The conversation messages for inference
            candidate_agents: List of candidate agents for inference
            gt_answer: Ground truth answer for reward calculation
            
        Returns:
            Tuple of (best_agent, best_answer, best_reward, cost)
        """

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

    async def build_agent_graph(self, question: str, task_graph: List[str], 
                               random_select: bool, mode: str = "test", 
                               gt_subtask: Optional[List[Union[str, float]]] = None) -> Dict[str, Any]:
        """Build agent graph for solving complex multi-step problems.
        
        This method orchestrates the entire problem-solving process by:
        1. Selecting appropriate agents for each subtask
        2. Coordinating inference across multiple steps
        3. Managing conversation history and dependencies
        4. Calculating costs and performance metrics
        
        Args:
            question: The main question to solve
            task_graph: List of subtasks in dependency order
            random_select: Whether to randomly select agents (for ablation)
            mode: Operating mode - 'test' or 'train'
            gt_subtask: Ground truth subtask answers (required for training)
            
        Returns:
            Dictionary containing:
                - nodes: List of solved subtask nodes
                - cost: Total inference cost
                - Additional metrics based on mode
                
        Raises:
            ValueError: If training mode is used without ground truth
        """
        if mode == "train" and gt_subtask is None:
            raise ValueError("Ground truth subtasks required for training mode")
            
        agent_graph = {"nodes": [], "cost": 0, "total_time": 0}
        conversation_history = []
        subtask_index = 0
        
        for subtask in task_graph:
            start_time = time.time()
            
            # Agent selection phase
            if random_select:
                candidate_agents = random.sample(self.agent_pool, self.top_k)
                logging.info(f"Randomly selected {len(candidate_agents)} agents")
            else:
                candidate_agents = await self.select_agent_subset(subtask)
            
            selection_time = time.time() - start_time
            logging.info(f"Agent selection completed in {selection_time:.3f}s")
            
            # Prepare inference message
            user_prompt = (
                f"Your current sub-question: {subtask}\n"
                "Conclude the answer by stating 'The answer is recorded as [what]. "
                "The answer is therefore \\boxed{[ANSWER]}'"
            )
            
            inference_message = conversation_history + [
                {"role": "user", "content": user_prompt}
            ]
            
            # Perform inference based on mode
            inference_start = time.time()
            if mode == "test":
                best_agent, answer, cost = await self.reso_testing(inference_message, candidate_agents)
                reward = None
            elif mode == "train":
                best_agent, answer, reward, cost = await self.reso_training_rule_based(
                    inference_message, candidate_agents, gt_subtask[subtask_index]
                )
                subtask_index += 1
            
            inference_time = time.time() - inference_start
            total_time = time.time() - start_time
            
            # Update conversation history
            conversation_history.extend([
                {"role": "user", "content": f"Sub-question: {subtask}"},
                {"role": "assistant", "content": answer}
            ])
            
            # Update agent graph
            agent_graph["cost"] += cost
            agent_graph["total_time"] += total_time
            
            # Create node record
            node_record = {
                "subtask": subtask,
                "selected_agent": best_agent.agent_id,
                "answer": answer,
                "cost": cost,
                "selection_time": selection_time,
                "inference_time": inference_time,
                "total_time": total_time
            }
            
            if mode == "train":
                node_record["reward"] = reward
                # Stop if agent fails (reward = 0)
                if reward == 0:
                    logging.warning(f"Agent failed on subtask {subtask_index}, stopping")
                    agent_graph["nodes"].append(node_record)
                    break
            
            agent_graph["nodes"].append(node_record)
            logging.info(
                f"Completed subtask {subtask_index + 1}/{len(task_graph)} "
                f"with agent {best_agent.agent_id} (cost: {cost:.4f}, time: {total_time:.3f}s)"
            )
        
        return agent_graph


