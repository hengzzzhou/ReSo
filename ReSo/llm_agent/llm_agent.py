import asyncio
import configparser
import logging
import sqlite3
import json
from openai import AsyncOpenAI
import configparser
import json
import sqlite3

from ReSo.llm_agent.cost import cost_count

class LLMAgent:
    """
    LLMAgent encapsulates the basic information and inference capabilities of an agent,
    and is responsible for persisting dynamic data (such as cost, scores, and visit counts)
    to a database.
    """
    def __init__(self, db_path, agent_id, base_model, profile, prompt,
                 inference_costs=None, scoring_history=None, visit=0):
        self.agent_id = agent_id
        self.base_model = base_model
        self.profile = profile
        self.prompt = prompt
        self.visit = visit
        self.db_path = db_path
        self.inference_costs = inference_costs if inference_costs is not None else []
        self.scoring_history = scoring_history if scoring_history is not None else []
        self._initialize_database()

    def _initialize_database(self):
        """
        Initialize the database and ensure the 'agents' table exists.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            base_model TEXT,
            profile TEXT,
            prompt TEXT,
            inference_costs TEXT,
            scoring_history TEXT,
            visit INTEGER DEFAULT 0
        )
        """
        cursor.execute(create_table_query)
        conn.commit()
        conn.close()

    @classmethod
    def create_agent(cls, db_path, base_model, profile, prompt=""):
        """
        Create or load an agent instance from the database.

        Args:
            db_path: Path to the database file.
            base_model: The model name.
            profile: The agent's role description.
            prompt: The default prompt content.

        Returns:
            An instance of LLMAgent.
        """
        agent_id = f"{base_model}_{profile}"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Ensure the 'agents' table exists
        create_table_query = """
        CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            base_model TEXT NOT NULL,
            profile TEXT NOT NULL,
            prompt TEXT,
            inference_costs TEXT,
            scoring_history TEXT,
            visit INTEGER DEFAULT 0
        )
        """
        cursor.execute(create_table_query)
        conn.commit()

        # Query if the agent already exists
        cursor.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,))
        row = cursor.fetchone()

        if row:
            inference_costs = json.loads(row["inference_costs"]) if row["inference_costs"] else []
            scoring_history = json.loads(row["scoring_history"]) if row["scoring_history"] else []
            visit = row["visit"] if row["visit"] is not None else 0
            loaded_prompt = row["prompt"] if row["prompt"] else ""
            conn.close()
            return cls(db_path, agent_id, row["base_model"], row["profile"], loaded_prompt,
                       inference_costs=inference_costs, scoring_history=scoring_history, visit=visit)
        else:
            inference_costs = []
            scoring_history = []
            visit = 0
            insert_query = """
            INSERT INTO agents (agent_id, base_model, profile, prompt, inference_costs, scoring_history, visit)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_query, (agent_id, base_model, profile, prompt,
                                          json.dumps(inference_costs), json.dumps(scoring_history), visit))
            conn.commit()
            conn.close()
            return cls(db_path, agent_id, base_model, profile, prompt,
                       inference_costs=inference_costs, scoring_history=scoring_history, visit=visit)

    @classmethod
    def load_agents_from_config(cls, config_file, db_path):
        """
        Load a list of agents from a configuration file.

        Args:
            config_file: Path to the configuration file.
            db_path: Path to the database file.

        Returns:
            A list of LLMAgent instances.
        """
        config = configparser.ConfigParser()
        config.read(config_file, encoding="utf-8")

        agents = []
        for section in config.sections():
            base_model = config[section].get("model")
            profile = config[section].get("role")
            prompt = config[section].get("prompt", "")
            agents.append(cls.create_agent(db_path, base_model, profile, prompt))
        return agents

    async def inference(self, message, temperature=0.2, max_tokens=4096, max_retries=3):
        """
        Call the OpenAI API to generate an inference and return the answer along with cost information.

        Args:
            message: A list of conversation messages.
            temperature: Sampling temperature for text generation.
            max_tokens: Maximum number of tokens to generate.
            max_retries: Maximum number of retries on failure.

        Returns:
            A tuple (answer, price, prompt_tokens, completion_tokens).
        """
        # Initialize multiple API clients
        client_qwen = AsyncOpenAI(
            api_key='sk-Gny5Nu8yJioYX4DAnrPNII9El7n2HmZd772g9dozbogexqmr',
            base_url='https://boyuerichdata.chatgptten.com/v1'
        )
        client_oai = AsyncOpenAI(
            api_key='sk-ug1gib8yNPuOFzcpM0gHv8QOhrA2fSfMu0Wi5dxt92vQTDQ6',
            base_url='https://boyuerichdata.chatgptten.com/v1'
        )
        client_claude = AsyncOpenAI(
            api_key='sk-W5M5gmYJALZh1HPiWBZjnQIUxssw5PSZC3fg5cqzuulQqvXe',
            base_url='https://api.claudeshop.top/v1'
        )

        # Choose the appropriate API client based on base_model
        if "qwen" in self.base_model:
            client = client_qwen
        elif "claude" in self.base_model:
            client = client_claude
        else:
            client = client_oai

        retries = 0
        while retries < max_retries:
            try:
                response = await client.chat.completions.create(
                    model=self.base_model,
                    messages=message,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                answer = response.choices[0].message.content.strip()
                price, prompt_len, completion_len = cost_count(response, self.base_model)
                return answer, price, prompt_len, completion_len  # Successful return
            except Exception as e:
                retries += 1
                logging.warning(f"{self.base_model} OpenAI API call failed (retry {retries}): {e}")
                # Wait with exponential backoff (1s, 2s, 4s, ...)
                await asyncio.sleep(2 ** retries)

        logging.error(f"{self.base_model} API request failed after {max_retries} retries, aborting.")
        return "API error", 0.05, 0, 0

    def _update_dynamic_field(self, field, data):
        """
        Update a dynamic field in the database for this agent.

        Args:
            field: The field name to update ("inference_costs", "scoring_history", "visit").
            data: The data to update.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if field == "inference_costs":
            update_query = "UPDATE agents SET inference_costs = ? WHERE agent_id = ?"
            cursor.execute(update_query, (json.dumps(data), self.agent_id))
        elif field == "scoring_history":
            update_query = "UPDATE agents SET scoring_history = ? WHERE agent_id = ?"
            cursor.execute(update_query, (json.dumps(data), self.agent_id))
        elif field == "visit":
            update_query = "UPDATE agents SET visit = ? WHERE agent_id = ?"
            cursor.execute(update_query, (data, self.agent_id))
        else:
            conn.close()
            raise ValueError("Unknown field name for dynamic update.")
        conn.commit()
        conn.close()


