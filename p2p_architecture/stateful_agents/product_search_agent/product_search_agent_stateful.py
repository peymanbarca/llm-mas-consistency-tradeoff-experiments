import os
import logging
import time
import uuid
import datetime
import httpx

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from httpx import AsyncClient
import json
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
import asyncio

######################
# [load_memory] → Inference search filters → fetch_candidates_tool → fetch_prices_tool → assemble_tool → [update_memory] → END
#######################


logger = logging.getLogger("product_search_agent")
logging.basicConfig(
    filename='../logs/product_search_agent.log',
    level=logging.INFO,  # Log all messages with severity DEBUG or higher
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the message format
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8008))
PRICING_SERVICE_URL = os.getenv("PRICING_SERVICE_URL", "http://localhost:8002")

llm = ChatOllama(model="llama3", temperature=0.2, reasoning=False)

app = FastAPI(title="Product Search Agent")

# DB client will be set on startup
db_client: Optional[AsyncIOMotorClient] = None
db = None


class ProductOut(BaseModel):
    sku: str
    name: str
    description: str


class ProductSearchResultItem(ProductOut):
    price: float
    score: float


class ProductCreate(BaseModel):
    sku: str
    name: str
    description: str


class ProductSearchResponse(BaseModel):
    query: str
    previous_memory: Optional[str]
    current_memory: Optional[str]
    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int
    search_filters: dict
    results: List[ProductSearchResultItem]


class ProductSearchAgentState(TypedDict):
    main_query: str
    search_filters: Dict[str, Any]
    user_id: str
    previous_memory_summary: Optional[str]
    current_memory_summary: Optional[str]
    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int
    candidates: List[Dict[str, Any]]
    prices: Dict[str, float]
    results: List[Dict[str, Any]]


@app.on_event("startup")
async def startup():
    global db_client, db
    db_client = AsyncIOMotorClient(MONGO_URI)
    db = db_client[MONGO_DB]
    logger.info("Connected to MongoDB at %s db=%s", MONGO_URI, MONGO_DB)


@app.on_event("shutdown")
async def shutdown():
    global db_client
    if db_client:
        db_client.close()
        logger.info("MongoDB connection closed")


async def load_user_memory(user_id: str) -> Optional[str]:
    doc = await db.user_memory.find_one({"user_id": user_id, "type": "search_preferences"})
    logger.info(f"memory loaded for user_id = {user_id} as {doc}")
    return doc["summary"] if doc else None


async def delete_user_memory(user_id: str):
    await db.user_memory.delete_many({"user_id": user_id, "type": "search_preferences"})
    logger.info(f"memory deleted for user_id = {user_id}")


async def save_user_memory(user_id: str, summary: str):
    await db.user_memory.update_one(
        {"user_id": user_id},
        {"$set": {"summary": summary, "type": "search_preferences", "updated_at": datetime.datetime.utcnow()}},
        upsert=True
    )
    doc = await db.user_memory.find_one({"user_id": user_id, "type": "search_preferences"})
    logger.info(f"memory updated for user_id = {user_id} as {doc}")


async def load_memory_node(
        state: ProductSearchAgentState
) -> ProductSearchAgentState:
    user_id = state.get("user_id")
    if not user_id:
        logger.info("user_id not provided, previous_memory_summary set as null")
        state["previous_memory_summary"] = None
        return state

    summary = await load_user_memory(user_id)
    state["previous_memory_summary"] = summary
    if summary is None:
        logger.info(f"user_id={user_id}, previous_memory_summary set as null")

    return state


async def infer_search_query(state: ProductSearchAgentState) -> ProductSearchAgentState:
    memory_block = (
        f"User preference summary:\n{state['previous_memory_summary']}\n\n"
        if state.get("previous_memory_summary")
        else ""
    )

    prompt = f"""
     You are a product search inference agent.

    Task:
    - Infer the search preferences from user raw query for parts only related to product name / description and pricing filter
    - Respect user preference summary if provided to adjust product name / description and pricing filter based on it
    - Return only a JSON with below schema without intermediate reasoning and analysis text:

    Schema:
    {{
            "product": string,
            "min_price": number,
            "max_price": number
    }}

    User raw query:
    {state.get("main_query")}
    
    {memory_block}

    """

    logger.info(f'infer_search_query -> LLM Call Prompt: {prompt}')

    st = time.time()
    response = await asyncio.to_thread(llm.invoke, prompt)
    raw_response = response.text()
    et = time.time()

    input_tokens = response.usage_metadata.get("input_tokens")
    output_tokens = response.usage_metadata.get("output_tokens")
    total_tokens = response.usage_metadata.get("total_tokens")
    logger.info(f'infer_search_query -> LLM Raw response: {raw_response}')

    logger.info(
        f'infer_search_query -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
        f' total_tokens: {total_tokens}, Took: f{round((et-st), 3)}')
    print(f'infer_search_query -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
          f' total_tokens: {total_tokens}, Took: f{round((et-st), 3)}')

    try:
        search_filters = parse_json_response(raw_response)
    except Exception as e:
        raise ValueError(f"Invalid result output: {raw_response}") from e

    state["search_filters"] = search_filters

    state["total_input_tokens"] += input_tokens
    state["total_output_tokens"] += output_tokens
    state["total_llm_calls"] += 1
    return state


async def update_memory_node(
        state: ProductSearchAgentState) -> ProductSearchAgentState:
    user_id = state.get("user_id")
    if not user_id:
        state["current_memory_summary"] = None
        return state

    # interaction = f"""
    #     User searched for: {state['search_filters']}
    #     Top results: {[f"description: {r['description']}, price={r['price']}, sku={r['sku']} " for r in state['results'][:3]]}
    #     """

    interaction = f"""
        User searched for: {state['search_filters']}
        """

    prompt = f"""
        You are a memory summarization agent.

        Tasks:
        - Update the existing summary concisely about search preferences of user with new interaction, by considering
            product name / description and pricing filter
        - Return only text of updated summary without any additional prefix or suffix

        Existing summary:
        {state.get("previous_memory_summary", "None")}

        New interaction:
        {interaction}

        """

    logger.info(f'update_memory_node -> LLM Call Prompt: {prompt}')

    st = time.time()
    response = await asyncio.to_thread(llm.invoke, prompt)
    new_summary = response.text().strip()
    et = time.time()

    input_tokens = response.usage_metadata.get("input_tokens")
    output_tokens = response.usage_metadata.get("output_tokens")
    total_tokens = response.usage_metadata.get("total_tokens")
    logger.info(f'update_memory_node -> LLM Raw response: {new_summary}')

    logger.info(
        f'update_memory_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
        f' total_tokens: {total_tokens}, Took: f{round((et-st), 3)}')
    print(f'update_memory_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
          f' total_tokens: {total_tokens}, Took: f{round((et-st), 3)}')

    state["current_memory_summary"] = new_summary
    state["total_input_tokens"] += input_tokens
    state["total_output_tokens"] += output_tokens
    state["total_llm_calls"] += 1
    await save_user_memory(user_id, new_summary)
    return state


async def fetch_candidates_tool(state: ProductSearchAgentState) -> ProductSearchAgentState:
    search_filters = state["search_filters"]
    product = search_filters.get('product', '')

    # Prefer text search
    cursor = db.products.find(
        {"$text": {"$search": product}},
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(10)

    docs = await cursor.to_list(length=10)

    # Fallback to regex (still deterministic DB logic)
    if not docs:
        docs = await db.products.find(
            {"name": {"$regex": product, "$options": "i"}}
        ).limit(10).to_list(length=10)

    state["candidates"] = docs
    return state


async def fetch_prices_tool(state: ProductSearchAgentState) -> ProductSearchAgentState:
    product_ids = [d["sku"] for d in state["candidates"]]

    prices = {}
    if not product_ids:
        state["prices"] = prices
        return state

    payload = {
        "items": [{"product_id": pid, "qty": 1} for pid in product_ids],
        "promo_codes": []
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(f"{PRICING_SERVICE_URL}/item-price", json=payload, timeout=10)
        r.raise_for_status()
        jr = r.json()

    for it in jr.get("items", []):
        prices[it["product_id"]] = it["unit_price"]

    state["prices"] = prices
    return state


def parse_json_response(text: str):
    import re
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return None
    except Exception as e:
        logging.error(f"parse error: {e} -- {text}")
        return None


def assemble_response_tool(state: ProductSearchAgentState) -> ProductSearchAgentState:
    prices = state.get("prices", {})
    candidates = state.get("candidates", {})

    search_filters = state["search_filters"]
    min_price = search_filters.get('min_price', 0)
    max_price = search_filters.get('max_price', 1000000)
    if min_price is None:
        min_price = 0
    if max_price is None:
        max_price = 1000000

    results = []
    for i in range(len(candidates)):
        # Apply price filtering
        price = prices.get(candidates[i]["sku"])
        if price is None or price < min_price or price > max_price:
            continue
        results.append({
            "sku": candidates[i]["sku"],
            "name": candidates[i]["name"],
            "description": candidates[i].get("description", ""),
            "price": prices.get(candidates[i]["sku"]),
            "score": float(candidates[i]["score"])
        })

    state["results"] = results
    return state


def build_product_search_agent():
    graph = StateGraph(ProductSearchAgentState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("infer_search_query", infer_search_query)
    graph.add_node("fetch_candidates", fetch_candidates_tool)
    graph.add_node("fetch_prices", fetch_prices_tool)
    graph.add_node("assemble_response", assemble_response_tool)
    graph.add_node("update_memory", update_memory_node)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "infer_search_query")
    graph.add_edge("infer_search_query", "fetch_candidates")
    graph.add_edge("fetch_candidates", "fetch_prices")
    graph.add_edge("fetch_prices", "assemble_response")
    graph.add_edge("assemble_response", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile()


search_graph = build_product_search_agent()


@app.post("/products")
async def create_product(p: ProductCreate):
    await db.products.insert_one(p.dict())
    return {"status": "created", "sku": p.sku}


@app.delete("/delete_memory")
async def delete_memory(user_id: str):
    await delete_user_memory(user_id)
    return {"status": "memory_deleted", "user_id": user_id}


@app.get("/search", response_model=ProductSearchResponse)
async def search_products(q: str = Query(...), user_id: Optional[str] = Query(None), limit: int = 5):
    state = {
        "main_query": q,
        "user_id": user_id,
        "candidates": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_llm_calls": 0,
        "prices": {},
        "results": []
    }

    logger.info(f"***************************************\nInput request prompt {q}, user_id: {user_id}")

    try:
        out = await search_graph.ainvoke(state)
        print(f'------------\n {out}')
        return ProductSearchResponse(
            query=q,
            previous_memory=out["previous_memory_summary"],
            current_memory=out["current_memory_summary"],
            total_input_tokens=out["total_input_tokens"],
            total_output_tokens=out["total_output_tokens"],
            total_llm_calls=out["total_llm_calls"],
            search_filters=out["search_filters"],
            results=out["results"][:limit]
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
