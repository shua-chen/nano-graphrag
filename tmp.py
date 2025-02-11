import os
import logging
import numpy as np
from openai import AsyncOpenAI,APIConnectionError, RateLimitError
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

LLM_BASE_URL = "https://api.agicto.cn/v1"
LLM_API_KEY = "sk-sYrWJZFIjyvgGQIWW2HKUcsih5S3OLMQHarq7QVs2SaTXkBr"
MODEL = "deepseek-v3"


# Assumed embedding model settings
EMBEDDING_BASE_URL = "https://api.agicto.cn/v1"
EMBEDDING_API_KEY = "sk-sYrWJZFIjyvgGQIWW2HKUcsih5S3OLMQHarq7QVs2SaTXkBr"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL_DIM = 1536
EMBEDDING_MODEL_MAX_TOKENS = 8192

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=61),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def llm_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content

@wrap_embedding_func_with_attrs(
    embedding_dim= EMBEDDING_MODEL_DIM,
    max_token_size= EMBEDDING_MODEL_MAX_TOKENS,
)
async def my_embedding(texts :list[str]) -> np.ndarray:
    openai_async_client = AsyncOpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY)
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


WORKING_DIR = "./sanguo"

def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=llm_model_if_cache,
        cheap_model_func=llm_model_if_cache,
        embedding_func=my_embedding,
    )
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="global",global_max_consider_community=1024)
        )
    )


def insert():
    from time import time

    with open("./三国.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=llm_model_if_cache,
        cheap_model_func=llm_model_if_cache,
        embedding_func=my_embedding,
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)
    # rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    # rag.insert(FAKE_TEXT[half_len:])



if __name__ == "__main__":
    insert()
    #query()