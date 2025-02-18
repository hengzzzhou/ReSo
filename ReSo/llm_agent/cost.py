#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLMAgent Module

This module defines the LLMAgent class which encapsulates the agent's basic information,
inference capabilities, and dynamic data persistence to a database.

It also contains singleton classes for cost and token counting, as well as
the cost_count function which calculates the API call cost based on the model used.
"""

import asyncio
import configparser
import difflib
import json
import logging
import random
import sqlite3
import time
from openai import AsyncOpenAI, OpenAI

from ReSo.llm_agent.model_info import OPENAI_MODEL_INFO

# ------------------------------------------------------------------
# Cost and Token Count Singletons
# ------------------------------------------------------------------
class Cost:
    """Singleton class for accumulating API call cost."""
    _instance = None

    def __init__(self):
        self.value = 0.0

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class PromptTokens:
    """Singleton class for accumulating the number of prompt tokens."""
    _instance = None

    def __init__(self):
        self.value = 0

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class CompletionTokens:
    """Singleton class for accumulating the number of completion tokens."""
    _instance = None

    def __init__(self):
        self.value = 0

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ------------------------------------------------------------------
# Cost Count Function
# ------------------------------------------------------------------
def cost_count(response, model_name):
    """
    Calculate the cost of an API call based on the response and model used,
    and update the global cost and token statistics.

    Args:
        response: The response object returned from the OpenAI API.
        model_name: The name of the model used, which is used to lookup pricing info.

    Returns:
        tuple: (price, prompt_tokens, completion_tokens)
    """
    branch = None
    if "gpt-4" in model_name:
        branch = "gpt-4"
    elif "gpt-3.5" in model_name:
        branch = "gpt-3.5"
    elif "deepseek" in model_name:
        branch = "deepseek"
    elif "gemini" in model_name:
        branch = "gemini"
    elif "claude" in model_name:
        branch = "claude"
    elif "qwen" in model_name:
        branch = "qwen"
    else:
        logging.error("Unknown model type: %s", model_name)
        return 0, 0, 0

    prompt_len = response.usage.prompt_tokens
    completion_len = response.usage.completion_tokens
    price = (prompt_len * OPENAI_MODEL_INFO[branch][model_name]["input"] +
             completion_len * OPENAI_MODEL_INFO[branch][model_name]["output"]) / 1000

    # Update global cost and token statistics
    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len

    return price, prompt_len, completion_len