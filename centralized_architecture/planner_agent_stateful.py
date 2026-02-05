import os
import logging
import time
import uuid
import datetime
import httpx
import sys
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Optional, Literal
from pymongo import MongoClient
from httpx import AsyncClient
import json
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph import StateGraph, END
import asyncio
import requests

logger = logging.getLogger("planner_agent")
logging.basicConfig(
    filename='logs/planner_agent.log',
    level=logging.INFO,  # Log all messages with severity DEBUG or higher
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the message format
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8010))

PRODUCT_AGENT_SEARCH_URL = "http://127.0.0.1:8008/search"
ORDER_AGENT_URL = "http://127.0.0.1:8000/"
INVENTORY_AGENT_RESERVE_URL = "http://127.0.0.1:8001/reserve"
INVENTORY_AGENT_RESERVE_ROLLBACK_URL = "http://127.0.0.1:8001/reserve-rollback"
CART_AGENT_URL = "http://127.0.0.1:8003/cart/"
PRICING_SERVICE_URL = "http://127.0.0.1:8002"
PAYMENT_AGENT_URL = "http://127.0.0.1:8007/pay-order"
SHIPMENT_AGENT_URL = "http://127.0.0.1:8006/book"

llm = ChatOllama(model="gpt-oss", temperature=0.3, reasoning=False)

app = FastAPI(title="Planner Agent")

# DB client will be set on startup
db_client: Optional[MongoClient] = None
db = None


class ProductOut(BaseModel):
    sku: str
    name: str
    description: str


class ProductSearchResultItem(ProductOut):
    price: float
    score: float


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


class CartItem(BaseModel):
    sku: str
    qty: int = Field(1, gt=0)


class Cart(BaseModel):
    cart_id: str
    items: List[CartItem] = []


class PriceResponseItem(BaseModel):
    product_id: str
    qty: int
    unit_price: float
    line_total: float
    discounts: float


class PriceResponse(BaseModel):
    items: List[PriceResponseItem]
    subtotal: float
    total_discount: float
    total: float
    currency: str


class OrderCreate(BaseModel):
    cart_id: str
    items: List[CartItem]


class OrderUpdate(BaseModel):
    order_id: str
    status: str


class OrderResponse(BaseModel):
    order_id: str
    status: str
    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int


class ReservationReq(BaseModel):
    order_id: str
    items: List[CartItem] = []
    atomic_update: bool = False
    delay: float = 0.0
    drop: int = 0
    ledger_pending: Dict[str, int]


class ReservationRes(BaseModel):
    order_id: str
    items: List = []
    status: str
    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int


class ShipmentPreferences(BaseModel):
    speed: Literal["fastest", "standard", "cheapest"]
    eco_friendly: bool
    avoid_weekend_delivery: bool
    preferred_carrier: Optional[str]


class ShipmentRequest(BaseModel):
    order_id: str
    address: str
    shipment_prefs: ShipmentPreferences  # receives this as input from planner agent (instead of directly infer from prompt)
    previous_memory_summary: Optional[str]  # receives this as input from planner agent (instead of directly access to memory)


class ShipmentResponse(BaseModel):
    previous_memory: Optional[str]
    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int
    shipment_prefs: dict
    shipment_id: str
    tracking_id: str


# ----------------- Main DTOs -------------------

class ProductSearchReq(BaseModel):
    search_prompt: str


class ProductSearchFinalRes(BaseModel):
    query: str
    previous_search_memory: Optional[str]
    current_search_memory: Optional[str]
    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int
    search_filters: dict
    results: List[ProductSearchResultItem]
    cart_id: str


class CartCheckoutReq(BaseModel):
    cart_id: str
    shipment_prompt: str


class CartCheckoutRes(BaseModel):
    order_id: str
    status: str
    shipment_prefs: Optional[dict]
    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int


# -------------------- Agent State --------------------------

class PlannerState(TypedDict):
    trace_id: str
    user_id: Optional[str]

    search_prompt: Optional[str]
    search_filters: Dict[str, Any]
    previous_search_memory_summary: Optional[str]
    current_search_memory_summary: Optional[str]
    search_results: List[Dict[str, Any]]

    cart_id: Optional[str]
    order_id: Optional[str]

    ledger_pending: Dict[str, int]

    shipment_prompt: Optional[str]
    shipment_prefs: Optional[dict]
    previous_shipment_memory_summary: Optional[str]
    current_shipment_memory_summary: Optional[str]

    items: List[dict]
    final_price: float

    atomic_update: bool
    delay: float
    drop: int

    inventory_status: Optional[str]
    payment_status: Optional[str]
    shipment_status: Optional[str]

    decision: Optional[str]
    phase: Optional[str]
    status: Optional[str]
    order_status: Optional[str]

    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int


@app.on_event("startup")
async def startup():
    global db_client, db
    db_client = MongoClient(MONGO_URI)
    db = db_client[MONGO_DB]
    logger.info("Connected to MongoDB at %s db=%s", MONGO_URI, MONGO_DB)


@app.on_event("shutdown")
async def shutdown():
    global db_client
    if db_client:
        db_client.close()
        logger.info("MongoDB connection closed")


# ------------------------- Memory helpers ---------------

def load_user_memory(user_id: str, type: str) -> Optional[str]:

    # type = shipment_preferences / search_preferences

    doc = db.user_memory.find_one({"user_id": user_id, "type": type})
    return doc["summary"] if doc else None


def delete_user_memory(user_id: str, type: str):
    db.user_memory.delete_many({"user_id": user_id, "type": type})


def save_user_memory(user_id: str, summary: str, type: str):
    db.user_memory.update_one(
        {"user_id": user_id},
        {"$set": {"summary": summary, "type": type, "updated_at": datetime.datetime.utcnow()}},
        upsert=True
    )


def load_search_memory_node(state: PlannerState) -> PlannerState:
    user_id = state.get("user_id")
    if not user_id:
        state["previous_search_memory_summary"] = None
        return state

    summary = load_user_memory(user_id, "search_preferences")
    state["previous_search_memory_summary"] = summary
    return state


def load_shipment_memory_node(state: PlannerState) -> PlannerState:
    user_id = state.get("user_id")
    if not user_id:
        state["previous_shipment_memory_summary"] = None
        return state

    summary = load_user_memory(user_id, "shipment_preferences")
    state["previous_shipment_memory_summary"] = summary
    return state


# ------------------------- TOOLS ------------------

def search_products(search_prefs: dict, user_id: Optional[str], previous_memory_summary: Optional[str]):
    """Search Products"""
    r = requests.post(PRODUCT_AGENT_SEARCH_URL + f"?user_id={user_id}" if user_id else PRODUCT_AGENT_SEARCH_URL,
                      json={"previous_memory": previous_memory_summary,
                            "search_filters": search_prefs},
                      timeout=10)
    r.raise_for_status()
    return r.json()


@tool
def fetch_cart(cart_id: str):
    """Fetch shopping cart items"""
    r = requests.get(CART_AGENT_URL + cart_id, timeout=10)
    r.raise_for_status()
    return r.json()


def add_to_cart(state):
    """Create new shopping cart items"""
    if state["search_results"] and len(["search_results"]) > 0:
        target_sku = state["search_results"][0]["sku"]
        r = requests.post(CART_AGENT_URL + "-1/items", json={"sku": target_sku, "qty": 1}, timeout=10)
        r.raise_for_status()
        return r.json()
    else:
        return None


def price_cart(state):
    """Fetch latest prices for cart items"""
    items = state['items']
    payload = {
        "items": [{"product_id": i["sku"], "qty": i["qty"]} for i in items],
        "promo_codes": [],
        "only_final_price": True
    }
    r = requests.post(f"{PRICING_SERVICE_URL}/item-price", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def initialize_order(state):
    """Initialize new order with cart items"""
    r = requests.post(ORDER_AGENT_URL + "order/create",
                      json={"cart_id": state["cart_id"], "items": state["items"]},
                      timeout=10)
    r.raise_for_status()
    return r.json()


def update_order_status(state):
    """Update order status"""
    r = requests.put(ORDER_AGENT_URL + "order/update",
                      json={"order_id": state["order_id"], "status": state["order_status"]},
                      timeout=10)
    r.raise_for_status()
    return r.json()


def reserve_inventory(state):
    """Reserve inventory with sending ledger"""

    ledger_pending = {}
    for item in state["items"]:
        # Read ledger pending reservations (in-flight)
        cursor = db.inventory_ledger.aggregate([
            {"$match": {"sku": item["sku"], "status": "PENDING"}},
            {"$group": {"_id": "$sku", "pending_qty": {"$sum": "$qty"}}}
        ])
        rows = cursor.to_list(length=1)
        pending_qty = rows[0]["pending_qty"] if rows else 0
        ledger_pending[item["sku"]] = pending_qty
    state["ledger_pending"] = ledger_pending

    payload = {
        "order_id": state['order_id'],
        "items": state['items'],
        "atomic_update": state['atomic_update'],
        "delay": state['delay'],
        "drop": state['drop'],
        "ledger_pending": state["ledger_pending"]
    }
    r = requests.post(INVENTORY_AGENT_RESERVE_URL, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def rollback_inventory(state):
    """Rollback inventory reservation"""
    payload = {
        "order_id": state['order_id'],
        "items": state['items'],
        "atomic_update": state['atomic_update'],
        "delay": state['delay'],
        "drop": state['drop'],
        "ledger_pending": {}
    }
    r = requests.post(INVENTORY_AGENT_RESERVE_ROLLBACK_URL, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def process_payment(state):
    """Process payment"""
    r = requests.post(PAYMENT_AGENT_URL,
                      json={"order_id": state['order_id'], "final_price": state['final_price']},
                      timeout=10)
    r.raise_for_status()
    return r.json()


def book_shipment(order_id: str, shipment_prefs: dict, user_id: Optional[str], previous_memory_summary: Optional[str]):
    """Book shipment"""
    r = requests.post(SHIPMENT_AGENT_URL + f"?user_id={user_id}" if user_id else SHIPMENT_AGENT_URL,
                      json={"order_id": order_id, "address": "SAMPLE_ADDRESS",
                            "previous_memory_summary": previous_memory_summary,
                            "shipment_prefs": shipment_prefs},
                      timeout=10)
    r.raise_for_status()
    return r.json()


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


# ----------------- Reasoning Nones (LLM-Driven) ------

def infer_search_query_node(state: PlannerState) -> PlannerState:
    memory_block = (
        f"User preference summary:\n{state['previous_search_memory_summary']}\n\n"
        if state.get("previous_search_memory_summary")
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
    {state.get("search_prompt")}

    {memory_block}

    """

    logger.info(f'infer_search_query -> LLM Call Prompt: {prompt}')

    st = time.time()
    response = llm.invoke(prompt)
    raw_response = response.text()
    et = time.time()

    input_tokens = response.usage_metadata.get("input_tokens")
    output_tokens = response.usage_metadata.get("output_tokens")
    total_tokens = response.usage_metadata.get("total_tokens")
    logger.info(f'infer_search_query -> LLM Raw response: {raw_response}')

    logger.info(
        f'infer_search_query -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
        f' total_tokens: {total_tokens}, Took: f{round((et - st), 3)}')
    print(f'infer_search_query -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
          f' total_tokens: {total_tokens}, Took: f{round((et - st), 3)}')

    try:
        search_filters = parse_json_response(raw_response)
    except Exception as e:
        raise ValueError(f"Invalid result output: {raw_response}") from e

    state["search_filters"] = search_filters
    state["phase"] = "IN_PROGRESS"

    state["total_input_tokens"] += input_tokens
    state["total_output_tokens"] += output_tokens
    state["total_llm_calls"] += 1
    return state


def infer_shipment_params_node(state: PlannerState) -> PlannerState:
    memory_block = (
        f"User shipment preference summary:\n{state['previous_shipment_memory_summary']}\n\n"
        if state.get("previous_shipment_memory_summary")
        else "No user preferences available.\n\n"
    )

    prompt = f"""
    You are a shipment planning agent.

    Task:
    - Infer shipment preferences from summarized memory if present
    - If memory is missing, try to infer preferences from user input, otherwise choose safe defaults
    - Return only a JSON with below schema without intermediate reasoning and analysis text:


    Schema:
    {{
      "speed": "fastest" | "standard" | "cheapest",
      "eco_friendly": bool,
      "avoid_weekend_delivery": bool,
      "preferred_carrier": string | null
    }}

    User input:
    {state.get("shipment_prompt")}

    {memory_block}

    Defaults:
    - speed = "standard"
    - eco_friendly = false
    - avoid_weekend_delivery = false
    - preferred_carrier = null

    """
    logger.info(f'infer_shipment_params_node -> LLM Call Prompt: {prompt}')

    st = time.time()
    response = llm.invoke(prompt)
    raw = response.text()
    et = time.time()

    input_tokens = response.usage_metadata.get("input_tokens")
    output_tokens = response.usage_metadata.get("output_tokens")
    total_tokens = response.usage_metadata.get("total_tokens")
    logger.info(f'infer_shipment_params_node -> LLM Raw response: {raw}')

    logger.info(
        f'infer_shipment_params_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
        f' total_tokens: {total_tokens}, Took: f{round((et - st), 3)}')
    print(
        f'infer_shipment_params_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
        f' total_tokens: {total_tokens}, Took: f{round((et - st), 3)}')

    prefs = parse_json_response(raw)
    ShipmentPreferences(**prefs)  # validation
    state["shipment_prefs"] = prefs

    state["total_input_tokens"] += input_tokens
    state["total_output_tokens"] += output_tokens
    state["total_llm_calls"] += 1

    return state


def update_search_memory_node(
        state: PlannerState) -> PlannerState:
    user_id = state.get("user_id")
    if not user_id:
        state["current_search_memory_summary"] = None
        return state

    # interaction = f"""
    #     User searched for: {state['search_filters']}
    #     Top results: {[f"description: {r['description']}, price={r['price']}, sku={r['sku']} " for r in state['search_results'][:3]]}
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
        {state.get("previous_search_memory_summary", "None")}

        New interaction:
        {interaction}

        """

    logger.info(f'update_search_memory_node -> LLM Call Prompt: {prompt}')

    st = time.time()
    response = llm.invoke(prompt)
    new_summary = response.text().strip()
    et = time.time()

    input_tokens = response.usage_metadata.get("input_tokens")
    output_tokens = response.usage_metadata.get("output_tokens")
    total_tokens = response.usage_metadata.get("total_tokens")
    logger.info(f'update_search_memory_node -> LLM Raw response: {new_summary}')

    logger.info(
        f'update_search_memory_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
        f' total_tokens: {total_tokens}, Took: f{round((et-st), 3)}')
    print(f'update_search_memory_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
          f' total_tokens: {total_tokens}, Took: f{round((et-st), 3)}')

    state["current_search_memory_summary"] = new_summary
    state["total_input_tokens"] += input_tokens
    state["total_output_tokens"] += output_tokens
    state["total_llm_calls"] += 1
    save_user_memory(user_id, new_summary, type="search_preferences")
    return state


def update_shipment_memory_node(
        state: PlannerState) -> PlannerState:
    user_id = state.get("user_id")
    if not user_id:
        state["current_shipment_memory_summary"] = None
        return state

    interaction = f"""
        User requested for: {state['shipment_prefs']}
        """

    prompt = f"""
        You are a memory summarization agent.

        Tasks:
        - Update the existing summary concisely about shipment preferences of user with new interaction
        - Return only text of updated summary without any additional prefix or suffix

        Existing summary:
        {state.get("previous_shipment_memory_summary", "None")}

        New interaction:
        {interaction}

        """

    logger.info(f'update_shipment_memory_node -> LLM Call Prompt: {prompt}')

    st = time.time()
    response = llm.invoke(prompt)
    new_summary = response.text().strip()
    et = time.time()

    input_tokens = response.usage_metadata.get("input_tokens")
    output_tokens = response.usage_metadata.get("output_tokens")
    total_tokens = response.usage_metadata.get("total_tokens")
    logger.info(f'update_shipment_memory_node -> LLM Raw response: {new_summary}')

    logger.info(
        f'update_shipment_memory_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
        f' total_tokens: {total_tokens}, Took: f{round((et-st), 3)}')
    print(f'update_shipment_memory_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
          f' total_tokens: {total_tokens}, Took: f{round((et-st), 3)}')

    state["current_shipment_memory_summary"] = new_summary
    state["total_input_tokens"] += input_tokens
    state["total_output_tokens"] += output_tokens
    state["total_llm_calls"] += 1
    save_user_memory(user_id, new_summary, type="shipment_preferences")
    return state


def orchestration_reason_node(state: PlannerState):

    orchestration_reasoning_prompt = f"""
        You are an autonomous planner agent that should orchestrate the workflow of retail supply chain.

        Tasks:
        - Your goal is to complete the workflow from product search, to finalizing shopping cart and purchase it.
        - You must decide the next_action as output based on PREVIOUS_ACTION, PHASE, and CURRENT_STATUS input
        - Return the next action as valid json in this schema: {{"next_action": string}} not programming code  without thinking steps in response


        Input:
        PREVIOUS_ACTION: {state['decision']}
        PHASE: {state['phase']}
        CURRENT_STATUS: {state['status']}


        Rules for choosing next_action:        
        
        If phase = "START1":
            choose next_action as INFER_SEARCH
        
        If phase = "START2": 
            choose next_action as FETCH_CART
        
        If phase = "IN_PROGRESS":
            Use ONLY this mapping to choose next_action based on PREVIOUS_ACTION:
                INFER_SEARCH -> SEARCH
                SEARCH -> ADD_TO_CART
                ADD_TO_CART -> FINISH
                FETCH_CART -> PRICE_CART
                PRICE_CART -> INIT_ORDER
                INIT_ORDER -> RESERVE_INVENTORY
                RESERVE_INVENTORY -> PROCESS_PAYMENT
                PROCESS_PAYMENT -> INFER_SHIPMENT
                INFER_SHIPMENT -> BOOK_SHIPMENT
                BOOK_SHIPMENT -> ORDER_UPDATE
                ORDER_UPDATE -> FINISH
        
        If phase = "EXCEPTION":
            - If CURRENT_STATUS is OUT_OF_STOCK -> next_action = "ORDER_UPDATE"
            - If CURRENT_STATUS is PAYMENT_FAILED -> next_action = "ROLLBACK_INVENTORY"
            - If CURRENT_STATUS is ROLLBACK_INVENTORY -> next_action = "ORDER_UPDATE"
        
        Hard constraints:
        - You MUST NOT output the same action as PREVIOUS_ACTION.
        
        Output ONLY valid JSON:
        {{
          "next_action": "..."
        }}
        
        """

    print(f'orchestrate_reason_node -> LLM Call Prompt: {orchestration_reasoning_prompt}')
    logger.info(f'orchestrate_reason_node -> LLM Call Prompt: {orchestration_reasoning_prompt}')
    st = time.time()
    response = llm.invoke(orchestration_reasoning_prompt)
    raw_response = response.text()
    et = time.time()

    input_tokens = response.usage_metadata.get("input_tokens")
    output_tokens = response.usage_metadata.get("output_tokens")
    total_tokens = response.usage_metadata.get("total_tokens")
    reasoning_text = response.additional_kwargs.get("reasoning_content", None)
    reasoning_tokens = response.usage_metadata.get("output_token_details", {}).get("reasoning", 0)

    print(f'orchestrate_reason_node -> LLM Reasoning Text: {reasoning_text}')
    logger.info(f'LLM Raw response: {raw_response}')
    print(f'orchestrate_reason_node -> LLM Raw response: {raw_response}')

    logger.info(
        f'orchestrate_reason_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
        f' reasoning_tokens: {reasoning_tokens}, total_tokens: {total_tokens}, Took: f{round((et - st), 3)}')
    print(f'orchestrate_reason_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
          f' reasoning_tokens: {reasoning_tokens}, total_tokens: {total_tokens}, Took: f{round((et - st), 3)}')

    decision = parse_json_response(raw_response)
    logger.info(f'orchestrate_reason_node -> LLM Parsed response: {decision}')
    print(f'orchestrate_reason_node -> LLM Parsed response: {decision}')

    state["decision"] = decision["next_action"]
    state["total_input_tokens"] += input_tokens
    state["total_output_tokens"] += output_tokens
    state["total_llm_calls"] += 1
    return state


# -------------- Action Nodes (Execution Tools) -----------

def product_search_node(state: PlannerState):
    logger.info(f'Calling product_search_node tool ... \n Current State is {state}')
    print(f'Calling product_search_node tool ... \n Current State is {state}')
    try:
        res = search_products(search_prefs=state["search_filters"], user_id=state["user_id"],
                              previous_memory_summary=state.get("previous_search_memory_summary", None))
        logger.info(f'Response of product_search_node tool ==> {res}, \n-------------------------------------')
        print(f'Response of product_search_node tool ==> {res}, \n-------------------------------------')

        state["search_results"] = res["results"]

        state["total_input_tokens"] += res["total_input_tokens"]
        state["total_output_tokens"] += res["total_output_tokens"]
        state["total_llm_calls"] += res["total_llm_calls"]

    except Exception as e:
        logger.info(f'Exception in response of product_search_node tool ==> {e}, \n-------------------------------------')
        print(f'Exception in response of product_search_node tool ==> {e}, \n-------------------------------------')

    return state


def add_to_cart_node(state: PlannerState):
    logger.info(f'Calling add_to_cart_node tool ... \n Current State is {state}')
    print(f'Calling add_to_cart_node tool ... \n Current State is {state}')
    cart = add_to_cart(state)
    logger.info(f'Response of add_to_cart_node tool ==> {cart}, \n-------------------------------------')
    print(f'Response of add_to_cart_node tool ==> {cart}, \n-------------------------------------')

    if cart:
        state["cart_id"] = cart["cart_id"]
        state["items"] = cart["items"]
        state["total_input_tokens"] += cart["total_input_tokens"]
        state["total_output_tokens"] += cart["total_output_tokens"]
        state["total_llm_calls"] += cart["total_llm_calls"]
        return state
    else:
        logger.info('Could not find any product and not cart is created, \n-------------------------------------')
        print('Could not find any product and not cart is created, \n-------------------------------------')
        return state


def fetch_cart_node(state: PlannerState):
    logger.info(f'Calling fetch_cart_node tool ... \n Current State is {state}')
    print(f'Calling fetch_cart_node tool ... \n Current State is {state}')
    cart = fetch_cart.invoke(state["cart_id"])
    logger.info(f'Response of fetch_cart_node tool ==> {cart}, \n-------------------------------------')
    print(f'Response of fetch_cart_node tool ==> {cart}, \n-------------------------------------')

    state["items"] = cart["items"]
    state["phase"] = "IN_PROGRESS"

    state["total_input_tokens"] += cart["total_input_tokens"]
    state["total_output_tokens"] += cart["total_output_tokens"]
    state["total_llm_calls"] += cart["total_llm_calls"]
    return state


# call pricing service API as tool
def pricing_node(state: PlannerState):
    logger.info(f'Calling pricing_node tool ... \n Current State is {state}')
    print(f'Calling pricing_node tool ... \n Current State is {state}')
    pricing = price_cart(state)
    logger.info(f'Response of pricing_node tool ==> {pricing}, \n-------------------------------------')
    print(f'Response of pricing_node tool ==> {pricing}, \n-------------------------------------')

    state["final_price"] = pricing["total"]
    state["total_input_tokens"] += pricing["total_input_tokens"]
    state["total_output_tokens"] += pricing["total_output_tokens"]
    state["total_llm_calls"] += pricing["total_llm_calls"]

    return state


def initialize_order_node(state: PlannerState):
    logger.info(f'Calling initialize_order_node tool ... \n Current State is {state}')
    print(f'Calling initialize_order_node tool ... \n Current State is {state}')
    res = initialize_order(state)
    logger.info(f'Response of initialize_order tool ==> {res}, \n-------------------------------------')
    print(f'Response of initialize_order tool ==> {res}, \n-------------------------------------')

    state["order_id"] = res["order_id"]
    state["order_status"] = res["status"]
    state["total_input_tokens"] += res["total_input_tokens"]
    state["total_output_tokens"] += res["total_output_tokens"]
    state["total_llm_calls"] += res["total_llm_calls"]
    return state


def update_order_status_node(state: PlannerState):
    logger.info(f'Calling update_order_status_node tool ... \n Current State is {state}')
    print(f'Calling update_order_status_node tool ... \n Current State is {state}')
    res = update_order_status(state)
    logger.info(f'Response of update_order_status tool ==> {res}, \n-------------------------------------')
    print(f'Response of update_order_status tool ==> {res}, \n-------------------------------------')

    state["phase"] = "IN_PROGRESS"
    state["total_input_tokens"] += res["total_input_tokens"]
    state["total_output_tokens"] += res["total_output_tokens"]
    state["total_llm_calls"] += res["total_llm_calls"]
    return state


def reserve_inventory_node(state: PlannerState):
    logger.info(f'Calling reserve_inventory_node tool ... \n Current State is {state}')
    print(f'Calling reserve_inventory_node tool ... \n Current State is {state}')
    res = reserve_inventory(state)
    logger.info(f'Response of reserve_inventory_node tool ==> {res}, \n-------------------------------------')
    print(f'Response of reserve_inventory_node tool ==> {res}, \n-------------------------------------')

    state["inventory_status"] = res["status"]
    if res["status"] == "OUT_OF_STOCK":
        state["status"] = "OUT_OF_STOCK"
        state["phase"] = "EXCEPTION"

    state["total_input_tokens"] += res["total_input_tokens"]
    state["total_output_tokens"] += res["total_output_tokens"]
    state["total_llm_calls"] += res["total_llm_calls"]

    # update order status
    state["order_status"] = state["inventory_status"]
    return state


def payment_node(state: PlannerState):
    logger.info(f'Calling payment_node tool ... \n Current State is {state}')
    print(f'Calling payment_node tool ... \n Current State is {state}')
    try:
        res = process_payment(state)
        logger.info(f'Response of payment_node tool ==> {res}, \n-------------------------------------')
        print(f'Response of payment_node tool ==> {res}, \n-------------------------------------')

        state["payment_status"] = res["status"]
        state["status"] = "PAYMENT_SUCCEED" if res["status"] == "SUCCESS" else "PAYMENT_FAILED"
        state["total_input_tokens"] += res["total_input_tokens"]
        state["total_output_tokens"] += res["total_output_tokens"]
        state["total_llm_calls"] += res["total_llm_calls"]

    except Exception as e:
        logger.info(f'Exception in response of payment_node tool ==> {e}, \n-------------------------------------')
        print(f'Exception in response of payment_node tool ==> {e}, \n-------------------------------------')
        state["payment_status"] = "FAILED"
        state["status"] = "PAYMENT_FAILED"
        state["phase"] = "EXCEPTION"

    finally:
        # update order status
        state["order_status"] = state["status"]

    return state


def rollback_node(state: PlannerState):
    logger.info(f'Calling rollback_node tool ... \n Current State is {state}, \n-------------------------------------')
    print(f'Calling rollback_node tool ... \n Current State is {state}, \n-------------------------------------')
    res = rollback_inventory(state)
    state["total_input_tokens"] += res["total_input_tokens"]
    state["total_output_tokens"] += res["total_output_tokens"]
    state["total_llm_calls"] += res["total_llm_calls"]
    state["phase"] = "EXCEPTION"
    return state


def shipment_node(state: PlannerState):
    logger.info(f'Calling shipment_node tool ... \n Current State is {state}')
    print(f'Calling shipment_node tool ... \n Current State is {state}')
    try:
        res = book_shipment(order_id=state["order_id"], shipment_prefs=state["shipment_prefs"],
                            user_id=state["user_id"], previous_memory_summary=state.get("previous_shipment_memory_summary", None))
        logger.info(f'Response of shipment_node tool ==> {res}, \n-------------------------------------')
        print(f'Response of shipment_node tool ==> {res}, \n-------------------------------------')

        state["shipment_status"] = "BOOKED"
        state["status"] = "COMPLETED"

        state["total_input_tokens"] += res["total_input_tokens"]
        state["total_output_tokens"] += res["total_output_tokens"]
        state["total_llm_calls"] += res["total_llm_calls"]

        # update order status
        state["order_status"] = "COMPLETED"

    except Exception as e:
        logger.info(f'Exception in response of shipment_node tool ==> {e}, \n-------------------------------------')
        print(f'Exception in response of shipment_node tool ==> {e}, \n-------------------------------------')
        state["shipment_status"] = "FAILED"
        state["status"] = "SHIPMENT_FAILED"

        # update order status
        state["order_status"] = "SHIPMENT_FAILED"

    return state


# ------------------- Langgraph --------

graph = StateGraph(state_schema=PlannerState)

# phase 1
graph.add_node("reason", orchestration_reason_node)
graph.add_node("infer_search", infer_search_query_node)
graph.add_node("product_search", product_search_node)
graph.add_node("add_to_cart", add_to_cart_node)

# phase 2
graph.add_node("fetch_cart", fetch_cart_node)
graph.add_node("price", pricing_node)
graph.add_node("init_order", initialize_order_node)
graph.add_node("reserve", reserve_inventory_node)
graph.add_node("pay", payment_node)
graph.add_node("rollback", rollback_node)
graph.add_node("infer_shipment", infer_shipment_params_node)
graph.add_node("ship", shipment_node)
graph.add_node("order_update", update_order_status_node)

graph.set_entry_point("reason")

graph.add_conditional_edges(
    "reason",
    lambda s: s["decision"],
    {
        "INFER_SEARCH": "infer_search",
        "SEARCH": "product_search",
        "FETCH_CART": "fetch_cart",
        "ADD_TO_CART": "add_to_cart",
        "PRICE_CART": "price",
        "INIT_ORDER": "init_order",
        "RESERVE_INVENTORY": "reserve",
        "PROCESS_PAYMENT": "pay",
        "ROLLBACK_INVENTORY": "rollback",
        "INFER_SHIPMENT": "infer_shipment",
        "BOOK_SHIPMENT": "ship",
        "ORDER_UPDATE": "order_update",
        "FINISH": END
    }
)

# loop back to reasoning at each step
for n in ["infer_search", "product_search", "add_to_cart",
          "fetch_cart", "price", "init_order", "reserve", "pay", "rollback", "infer_shipment", "ship", "order_update"]:
    graph.add_edge(n, "reason")

planner_agent = graph.compile()


# -------------------- Phase 1 of workflow --------------------------
def run_search_add_to_cart_procedure(search_prompt: str, user_id: Optional[str]):
    state: PlannerState = {
        "trace_id": str(uuid.uuid4()),
        "order_id": None,
        "cart_id": None,
        "search_prompt": search_prompt,
        "user_id": user_id,

        "items": [],
        "final_price": 0.0,

        "atomic_update": False,
        "delay": 0.0,
        "drop": 0,

        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_llm_calls": 0,

        "inventory_status": None,
        "payment_status": None,
        "shipment_status": None,

        "decision": None,
        "phase": "START1",
        "status": None,
        "order_status": None
    }

    try:
        logger.info(f'Request for search_add_to_cart, search_prompt = {search_prompt}, state={state}')
        print(f'Request for search_add_to_cart, search_prompt = {search_prompt}, state={state}')

        final_state = planner_agent.invoke(state, config={"recursion_limit": 14})

        return {
            "query": search_prompt,
            "previous_search_memory": final_state.get("previous_search_memory_summary", None),
            "current_search_memory": final_state.get("current_search_memory_summary", None),
            "total_input_tokens": final_state["total_input_tokens"],
            "total_output_tokens": final_state["total_output_tokens"],
            "total_llm_calls": final_state["total_llm_calls"],
            "search_filters": final_state["search_filters"],
            "results": final_state["search_results"],
            "cart_id": final_state["cart_id"]
        }
    except Exception as e:
        print(e)
        logger.exception(e)
        exc_type, exc_obj, tb = sys.exc_info()
        line_number = tb.tb_lineno
        print(f"Error on line {line_number}: {type(e).__name__}, {e}")


@app.post("/cart/add", response_model=ProductSearchFinalRes, summary="Search Products & Add to Shopping Cart")
async def search_and_add_to_cart(req: ProductSearchReq, user_id: Optional[str] = Query(None)):
    result = run_search_add_to_cart_procedure(search_prompt=req.search_prompt, user_id=user_id)
    logger.info(f'Request for search_and_add_to_cart processed successfully, search_prompt = {req.search_prompt}, result={result}')
    print(f'Request for search_and_add_to_cart processed successfully, search_prompt = {req.search_prompt}, result={result}')
    return result


# -------------------- Phase 2 of workflow --------------------------
def run_checkout_cart_procedure(cart_id: str, shipment_prompt: str, user_id: Optional[str]):
    state: PlannerState = {
        "trace_id": str(uuid.uuid4()),
        "order_id": None,
        "cart_id": cart_id,
        "shipment_prompt": shipment_prompt,
        "user_id": user_id,

        "items": [],
        "final_price": 0.0,

        "atomic_update": False,
        "delay": 0.0,
        "drop": 0,

        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_llm_calls": 0,

        "inventory_status": None,
        "payment_status": None,
        "shipment_status": None,

        "decision": None,
        "phase": "START2",
        "status": None,
        "order_status": None
    }

    logger.info(f'Request for checkout_cart, cart_id = {cart_id}, state={state}')
    print(f'Request for checkout_cart, cart_id = {cart_id}, state={state}')

    final_state = planner_agent.invoke(state, config={"recursion_limit": 14})

    return {
        "order_id": final_state["order_id"],
        "status": final_state["status"],
        "shipment_prefs": final_state.get("shipment_prefs", None),
        "total_input_tokens": final_state["total_input_tokens"],
        "total_output_tokens": final_state["total_output_tokens"],
        "total_llm_calls": final_state["total_llm_calls"]
    }


@app.post("/cart/checkout", response_model=CartCheckoutRes, summary="Purchase Shopping Cart")
async def checkout_cart(req: CartCheckoutReq, user_id: Optional[str] = Query(None)):
    result = run_checkout_cart_procedure(cart_id=req.cart_id, shipment_prompt=req.shipment_prompt, user_id=user_id)
    logger.info(f'Request for checkout_cart processed successfully, cart_id = {req.cart_id}, result={result}')
    print(f'Request for checkout_cart processed successfully, cart_id = {req.cart_id}, result={result}')
    return result

