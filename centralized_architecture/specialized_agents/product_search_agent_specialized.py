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
# receives search filters as input → fetch_candidates_tool
# → fetch_prices_tool → assemble_tool → END
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

llm = ChatOllama(model="gpt-oss", temperature=0.2, reasoning=False)

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


class ProductSearchRequest(BaseModel):
    previous_memory: Optional[str]  # receives this as input from planner agent (instead of directly access to memory)
    search_filters: dict  # receives this as input from planner agent (instead of directly infer from prompt)


class ProductSearchResponse(BaseModel):
    previous_memory: Optional[str]
    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int
    search_filters: dict
    results: List[ProductSearchResultItem]


class ProductSearchAgentState(TypedDict):
    search_filters: Dict[str, Any]
    user_id: str
    previous_memory_summary: Optional[str]
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

    graph.add_node("fetch_candidates", fetch_candidates_tool)
    graph.add_node("fetch_prices", fetch_prices_tool)
    graph.add_node("assemble_response", assemble_response_tool)

    graph.set_entry_point("fetch_candidates")
    graph.add_edge("fetch_candidates", "fetch_prices")
    graph.add_edge("fetch_prices", "assemble_response")
    graph.add_edge("assemble_response", END)

    return graph.compile()


search_graph = build_product_search_agent()


@app.post("/products")
async def create_product(p: ProductCreate):
    await db.products.insert_one(p.dict())
    return {"status": "created", "sku": p.sku}


@app.post("/search", response_model=ProductSearchResponse)
async def search_products(req: ProductSearchRequest, user_id: Optional[str] = Query(None), limit: int = 5):
    state = {
        "user_id": user_id,
        "search_filters": req.search_filters,
        "previous_memory_summary": req.previous_memory,
        "candidates": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_llm_calls": 0,
        "prices": {},
        "results": []
    }

    logger.info(f"***************************************\nInput request {req}, user_id: {user_id}")

    try:
        out = await search_graph.ainvoke(state)
        print(f'------------\n {out}')
        return ProductSearchResponse(
            previous_memory=out["previous_memory_summary"],
            total_input_tokens=out["total_input_tokens"],
            total_output_tokens=out["total_output_tokens"],
            total_llm_calls=out["total_llm_calls"],
            search_filters=out["search_filters"],
            results=out["results"][:limit]
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
