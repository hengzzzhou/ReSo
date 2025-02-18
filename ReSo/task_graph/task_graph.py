import ast
import json
import time
import random
import re
import logging
from openai import OpenAI

class TaskGraph:
    """
    TaskGraph is responsible for decomposing complex tasks into subtasks
    using a language model-based approach.
    """

    def __init__(self):
        # Initialize OpenAI clients for task graph generation
        self.client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="EMPTY")
        self.client_oai = OpenAI(base_url='https://api.claudeshop.top/v1', 
                                 api_key='sk-vTw7ftEbOSsnVWEFsXUICrnQs1PUHQh6TPqKpmRDOzNqlNiO')

    def build_task_graph(self, question: str) -> str:
        """
        Constructs a task graph by decomposing the given question into subtasks.

        Args:
            question (str): The complex task that needs to be decomposed.

        Returns:
            str: JSON-formatted task graph with ordered subtasks.
        """
        messages = [
            {
                "role": "system",
                "content": "Please decompose the following complex task into some subtasks. "
                           "Use a list to represent them, sorted in the best order to solve."
            },
            {
                "role": "user",
                "content": question
            }
        ]

        response = self.client.chat.completions.create(
            model="graph-planning-lora",
            messages=messages,
            temperature=0.3
        )
        response_text = response.choices[0].message.content

        # Attempt to parse JSON response
        try:
            task_graph = json.loads(response_text)
        except json.JSONDecodeError:
            task_graph = self._extract_json(response_text)

        return json.dumps(task_graph, ensure_ascii=False, indent=4)

    def build_task_graph_oai(self, question: str) -> list:
        """
        Uses GPT-4 API to decompose a complex question into an ordered list of subtasks.

        Args:
            question (str): The complex task that needs to be decomposed.

        Returns:
            list: Ordered list of subtasks for solving the given problem.
        """
        messages = [
            {
                "role": "system",
                "content": "Please decompose the following complex task into some subtasks. "
                           "The output must be a Python list, formatted correctly for direct use with "
                           "ast.literal_eval(response_text). Ensure the list is ordered optimally "
                           "to solve the task step by step. Use concise descriptions for each step."
            },
            {
                "role": "user",
                "content": question
            }
        ]

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response_oai = self.client_oai.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.3
                )
                response_text = response_oai.choices[0].message.content.strip()

                try:
                    # Attempt to parse using `ast.literal_eval`
                    parsed_list = ast.literal_eval(response_text)
                except (ValueError, SyntaxError) as e:
                    logging.warning(f"Parsing failed: {e}. Using fallback method (split on ',').")
                    parsed_list = [item.strip() for item in response_text.split("',")]

                return parsed_list  # Successfully parsed response

            except Exception as e:
                logging.warning(f"Request failed: {e}. Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2 ** attempt)  # Exponential backoff for retries

        raise RuntimeError("Maximum retry attempts reached. Unable to obtain a valid response.")

    def _extract_json(self, text: str) -> dict:
        """
        Attempts to extract a JSON structure from raw text.

        Args:
            text (str): Raw response text that may contain JSON content.

        Returns:
            dict: Extracted JSON data if successful, otherwise a default empty structure.
        """
        # Regular expression to extract JSON content from a response
        pattern = re.compile(r'(\{.*\})', re.DOTALL)
        match = pattern.search(text)

        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logging.warning("Failed to parse extracted JSON.")

        # Return default structure if parsing fails
        return {"nodes": [], "edges": []}
