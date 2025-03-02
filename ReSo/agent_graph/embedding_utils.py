import asyncio
import numpy as np
from openai import AsyncOpenAI

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors and map it from [-1,1] to [0,1].
    """
    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return (cosine_sim + 1) / 2.0

async def get_embedding_async(aclient, text, cache):
    """
    Asynchronously get the embedding of a text using the provided OpenAI client.
    Use a cache to avoid duplicate API calls.
    """
    if text in cache:
        return cache[text]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await aclient.embeddings.create(input=[text], model="text-embedding-ada-002")
            embedding = np.array(response.data[0].embedding)
            cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Embedding attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.1)
