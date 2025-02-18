import ast
import asyncio
import os
import argparse
import sys
import json
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ReSo.llm_agent.llm_agent import LLMAgent
from ReSo.task_graph.task_graph import TaskGraph
from ReSo.agent_graph.agent_graph import AgentGraph
from datasets.get_answer import equiv, parse_math_answer
logging.basicConfig(level=logging.INFO)
def load_dataset(dataset_path):
    """
    Load dataset from the specified JSON file.

    Args:
        dataset_path (str): Path to the dataset JSON file.

    Returns:
        list: Loaded dataset as a list of dictionaries.
    """
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            logging.info(f"Dataset loaded successfully with {len(dataset)} samples.")
        return dataset
    else:
        logging.error(f"Dataset not found at {dataset_path}.")
        return []

async def main():
    """
    Main function to execute the agent-based reasoning process.
    """
    # Argument parser configuration
    parser = argparse.ArgumentParser(description="Train in scibench-mas dataset")
    parser.add_argument('--dataset_path', type=str, default="datasets/Scibench-MAS/Scibench-MAS-Easy.json")
    parser.add_argument('--random_select', action='store_true', default=False)
    parser.add_argument('--plan_mode', type=str, default="gt")
    parser.add_argument("--error_tolerance", type=float, default=0.15)
    args = parser.parse_args()
    error_tolerance=args.error_tolerance
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    random_select = args.random_select
    plan_mode = args.plan_mode
    logging.info(f"Running Mode: {plan_mode} | Random: {random_select} | Dataset: {args.dataset_path}")

    # Load agent pool from configuration
    try:
        agent_pool = LLMAgent.load_agents_from_config("config.ini", "DB/llm_agent_train.db")
    except Exception as e:
        logging.error(f"Failed to load agent pool: {e}")
        return

    # Compute total visits across all agents
    total_visit = sum(agent.visit for agent in agent_pool)

    # Load environment variables
    load_dotenv(override=True)
    similarity_weight = float(os.getenv("similarity_weight", 1.0))
    reputation_weight = float(os.getenv("reputation_weight", 1.0))
    cost_weight = float(os.getenv("cost_weight", 1.0))
    threshold = float(os.getenv("THRESHOLD", 0.5))
    exploration_const = float(os.getenv("EXPLORATION_CONST", 1.0))
    mcts_iterations = int(os.getenv("MCTS_ITERATIONS", 3))
    top_k = int(os.getenv("TOP_K", 3))

    logging.info(f"Top-K selection: {top_k}")

    # Initialize agent graph builder
    agent_graph_builder = AgentGraph(
        agent_pool=agent_pool,
        similarity_weight=similarity_weight,
        reputation_weight=reputation_weight,
        cost_weight=cost_weight,
        threshold=threshold,
        exploration_const=exploration_const,
        mcts_iterations=mcts_iterations,
        top_k=top_k,
        mode=plan_mode,
        total_visit=total_visit
    )

    # Initialize variables for evaluation metrics
    acc_list = []
    cost_list = []
    total_solved = 0
    total_executed = 0
    task_graph_builder = TaskGraph()

    # Iterate through the dataset
    for idx, sample in enumerate(tqdm(dataset, desc="Processing dataset", unit="sample")):
        question = sample.get("problem_text_sort", "")
        gt_final = sample.get("answer_number", "")
        sorted_gt_subtask = sample.get("gt_subtask", "")

        logging.info(f"Processing sample {idx + 1}")

        # Generate task graph based on the selected planning mode
        if plan_mode == "gt":
            parsed_list = ast.literal_eval(sample.get("gt_plan", "[]"))
            task_graph = parsed_list
        elif plan_mode == "gpt":
            task_graph = task_graph_builder.build_task_graph_oai(question)
        else:
            for attempt in range(3):
                try:
                    task_graph_str = task_graph_builder.build_task_graph(question)
                    task_graph_dic = json.loads(task_graph_str)
                    task_graph = [item["instruction"] for item in task_graph_dic]
                    break
                except Exception as e:
                    logging.warning(f"Data parsing error (Attempt {attempt + 1}/3): {e}")
            else:
                task_graph = [question]

        logging.info(f"Ground Truth Subtasks: {sorted_gt_subtask}")

        # Construct the agent graph
        agent_graph = await agent_graph_builder.build_agent_graph_train(
            question, task_graph, random_select=random_select,gt_subtask=sorted_gt_subtask
        )

        # Track cost
        cost_list.append(agent_graph.get("cost", 0))

        # Extract final predicted answer
        if agent_graph.get("nodes"):
            predicted_final = agent_graph["nodes"][-1].get("answer", "")
        else:
            predicted_final = ""

        # Parse and compare predicted answer with ground truth
        predict_answer = parse_math_answer(predicted_final)
        try:
            is_solved = equiv(predict_answer, gt_final, error_tolerance)
        except Exception:
            is_solved = False

        total_solved += is_solved
        total_executed += 1
        accuracy = total_solved / total_executed
        acc_list.append(accuracy)

        # Log evaluation metrics
        logging.info(f"Accuracy: {accuracy:.3f}, Cost: {agent_graph.get('cost', 0):.3f}")
        print(f"Accuracy: {accuracy:.3f}, Cost: {agent_graph.get('cost', 0):.3f}")
        
        
    logging.info(f"acc_list: {acc_list} ")
    avg_loss = sum(acc_list) / len(acc_list)
    logging.info("acc = {:.3f}".format(avg_loss))
    
    if plt is not None:
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 3, 1)
        plt.plot(acc_list, marker='o', label="acc")
        plt.title("acc")
        plt.xlabel("n")
        plt.ylabel("Acc")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("training_curves.png")
        plt.show()

if __name__ == "__main__":
    asyncio.run(main())
