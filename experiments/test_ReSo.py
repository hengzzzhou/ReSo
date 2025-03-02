import ast
import asyncio
import configparser
import os
import argparse
import sys
import json
import logging
from tqdm import tqdm
from dotenv import load_dotenv

# Add the parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import necessary modules
from ReSo.llm_agent.llm_agent import LLMAgent
from ReSo.task_graph.task_graph import TaskGraph
from ReSo.agent_graph.agent_graph import AgentGraph
from datasets.get_answer import equiv, parse_math_answer,load_gsmhard_dataset,load_dataset


# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    """
    Main function to execute the agent-based reasoning process.
    """
    # Argument parser configuration
    parser = argparse.ArgumentParser(description="Test in math-mas dataset")
    parser.add_argument('--dataset_path', type=str, default="datasets/MATH-MAS/MATH-MAS-Easy.json")
    parser.add_argument('--random_select', action='store_true', default=False)
    parser.add_argument('--plan_mode', type=str, default="gt")
    parser.add_argument("--error_tolerance", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples to process in a batch")
    args = parser.parse_args()
    error_tolerance=args.error_tolerance
    batch_size = args.batch_size

    # Load dataset
    dataset = load_dataset(args.dataset_path)
    random_select = args.random_select
    plan_mode = args.plan_mode
    logging.info(f"Running Mode: {plan_mode} | Random: {random_select} | Dataset: {args.dataset_path}")

    # Load agent pool from configuration
    try:
        agent_pool = LLMAgent.load_agents_from_config("config.ini", "DB/llm_agent.db")
    except Exception as e:
        logging.error(f"Failed to load agent pool: {e}")
        return

    # Compute total visits across all agents
    total_visit = sum(agent.visit for agent in agent_pool)

    config = configparser.ConfigParser()
    config.read('config_hyper.ini')

    similarity_weight = float(config['hyperparameters']['similarity_weight'])
    reputation_weight = float(config['hyperparameters']['reputation_weight'])
    cost_weight = float(config['hyperparameters']['cost_weight'])
    threshold = float(config['hyperparameters']['THRESHOLD'])
    exploration_const = float(config['hyperparameters']['EXPLORATION_CONST'])
    top_k = int(config['hyperparameters']['TOP_K'])

    logging.info(f"Top-K selection: {top_k}")

    # Initialize agent graph builder
    agent_graph_builder = AgentGraph(
        agent_pool=agent_pool,
        similarity_weight=similarity_weight,
        reputation_weight=reputation_weight,
        cost_weight=cost_weight,
        threshold=threshold,
        exploration_const=exploration_const,
        top_k=top_k,
        mode=plan_mode,
        total_visit=total_visit
    )

    # Initialize variables for evaluation metrics
    loss_list = []
    cost_list = []
    total_solved = 0
    total_executed = 0
    task_graph_builder = TaskGraph()

    # Iterate through the dataset in batches with progress bar
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing Batches"):
        batch_samples = dataset[batch_start:batch_start + batch_size]
        batch_tasks = []

        for idx, sample in enumerate(batch_samples):
            question = sample.get("problem_text_sort", "")
            gt_final = sample.get("answer_number", "")
            #sorted_gt_subtask = sample.get("gt_subtask", "")

            logging.info(f"Processing sample {batch_start + idx + 1}")

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

            #logging.info(f"Ground Truth Subtasks: {sorted_gt_subtask}")

            # Construct the agent graph
            batch_tasks.append(agent_graph_builder.build_agent_graph(
                question, task_graph, random_select=random_select,mode="test"
            ))

        # Await all tasks in the batch
        batch_results = await asyncio.gather(*batch_tasks)

        for agent_graph, sample in zip(batch_results, batch_samples):
            # Track cost
            cost_list.append(agent_graph.get("cost", 0))

            # Extract final predicted answer
            if agent_graph.get("nodes"):
                predicted_final = agent_graph["nodes"][-1].get("answer", "")
            else:
                predicted_final = ""

            # Parse and compare predicted answer with ground truth
            predict_answer = parse_math_answer(predicted_final)
            gt_final = sample.get("answer_number", "")
            try:
                is_solved = equiv(predict_answer, gt_final, error_tolerance)
            except Exception:
                is_solved = False

            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed
            loss_list.append(accuracy)

            # Log evaluation metrics
            logging.info(f"Accuracy: {accuracy:.3f}, Cost: {agent_graph.get('cost', 0):.3f}")
            print(f"Accuracy: {accuracy:.3f}, Cost: {agent_graph.get('cost', 0):.3f}")

if __name__ == "__main__":
    asyncio.run(main())