import os
import logging
import time
import uuid
import datetime
import httpx

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from httpx import AsyncClient
import json
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph import StateGraph, END
import asyncio
import requests

logger = logging.getLogger("order_agent")
logging.basicConfig(
    filename='logs/order_agent.log',
    level=logging.INFO,  # Log all messages with severity DEBUG or higher
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the message format
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8000))

llm = ChatOllama(model="gpt-oss", temperature=0.0, reasoning=False)

app = FastAPI(title="Order Agent")

# DB client will be set on startup
db_client: Optional[AsyncIOMotorClient] = None
db = None

PRICING_SERVICE_URL = "http://127.0.0.1:8002"


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


# -------------------- Agent State --------------------------

class OrderState(TypedDict):
    trace_id: str
    order_id: str
    cart_id: Optional[str]

    items: List[dict]
    final_price: float

    decision: Optional[str]
    update_status: Optional[str]
    status: Optional[str]

    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int


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


# ------------------------- TOOLS ------------------

def price_cart(state):
    """Fetch latest prices for cart items"""
    items = state['items']
    print(items)
    payload = {
        "items": [{"product_id": i.sku, "qty": i.qty} for i in items],
        "promo_codes": [],
        "only_final_price": True
    }
    r = requests.post(f"{PRICING_SERVICE_URL}/item-price", json=payload, timeout=10)
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


# ----------------- Reasoning None (LLM-Driven) ------

def order_reason_node(state: OrderState):
    order_reasoning_prompt = f"""
        You are an autonomous order agent.

        Tasks:
        - Your goal is to complete an order workflow.
        - You must decide the next_action as output based on PREVIOUS_ACTION and UPDATE_STATUS inputs
        - Return ONLY a JSON response not python code

        - Do not return middle steps and thinking procedure in response
        - Return the next action as valid json in this schema: {{"next_action": string}}

        Possible actions:
        - PRICE_CART
        - CREATE_ORDER
        - UPDATE_ORDER

        Rules:
        - If PREVIOUS_ACTION is empty, choose the next_action as PRICE_CART
        - Else, choose the next_action as CREATE_ORDER

        Rule Exceptions:
        - If UPDATE_STATUS is not null choose next action as UPDATE_ORDER
        - Never skip any steps

        Input:
        PREVIOUS_ACTION: {state['decision']}
        UPDATE_STATUS: {state['update_status']}
        """

    logger.info(f'orchestrate_reason_node -> LLM Call Prompt: {order_reasoning_prompt}')
    st = time.time()
    response = llm.invoke(order_reasoning_prompt)
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

# call pricing service API as tool
def pricing_node(state: OrderState):
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


def create_order_node(state: OrderState):
    logger.info(f'Calling create_order_node tool ... \n Current State is {state}')
    print(f'Calling create_order_node tool ... \n Current State is {state}')

    # init order in DB
    db.orders.insert_one(
        {"_id": state['order_id'], "items": [{'sku': item.sku, 'qty': item.qty} for item in state['items']],
         "cart_id": state['cart_id'], "status": "INIT",
         "final_price": state['final_price']})

    state["status"] = "INIT"
    return state


def update_order_node(state: OrderState):
    logger.info(f'Calling update_order_node tool ... \n Current State is {state}')
    print(f'Calling update_order_node tool ... \n Current State is {state}')

    # update order status in DB
    db.orders.update_one({"_id": state['order_id']}, {"$set": {"status": state["update_status"]}})

    state["status"] = state["update_status"]
    return state


# ------------------- Langgraph --------

graph = StateGraph(OrderState)

graph.add_node("reason", order_reason_node)
graph.add_node("price", pricing_node)
graph.add_node("create_order", create_order_node)
graph.add_node("update_order", update_order_node)

graph.set_entry_point("reason")

graph.add_conditional_edges(
    "reason",
    lambda s: s["decision"],
    {
        "PRICE_CART": "price",
        "CREATE_ORDER": "create_order",
        "UPDATE_ORDER": "update_order",
        "FINISH": END
    }

)
graph.add_edge("price", "create_order")
graph.add_edge("create_order", END)
graph.add_edge("update_order", END)

order_agent = graph.compile()


@app.post("/order/create", response_model=OrderResponse, summary="Initialize Order")
async def initialize_order(req: OrderCreate):
    state: OrderState = {
        "trace_id": str(uuid.uuid4()),
        "order_id": str(uuid.uuid4()),
        "cart_id": req.cart_id,
        "status": None,

        "items": req.items,
        "final_price": 0.0,

        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_llm_calls": 0,

        "decision": None,
        "update_status": None
    }

    logger.info(f'Request for initialize_order, cart_id = {req.cart_id}, state={state}')
    print(f'Request for initialize_order, cart_id = {req.cart_id}, state={state}')

    final_state = order_agent.invoke(state, config={"recursion_limit": 6})

    return OrderResponse(order_id=final_state["order_id"],
                         status=final_state["status"],
                         total_input_tokens=final_state["total_input_tokens"],
                         total_output_tokens=final_state["total_output_tokens"],
                         total_llm_calls=final_state["total_llm_calls"])


@app.put("/order/update", response_model=OrderResponse, summary="Update Order")
async def update_order(req: OrderUpdate):
    state: OrderState = {
        "trace_id": str(uuid.uuid4()),
        "order_id": req.order_id,
        "cart_id": None,
        "status": None,

        "items": [],
        "final_price": 0.0,

        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_llm_calls": 0,

        "decision": None,
        "update_status": req.status
    }

    logger.info(f'Request for update_order, order_id = {req.order_id}, state={state}')
    print(f'Request for update_order, order_id = {req.order_id}, state={state}')

    final_state = order_agent.invoke(state, config={"recursion_limit": 6})

    return OrderResponse(order_id=final_state["order_id"],
                         status=final_state["status"],
                         total_input_tokens=final_state["total_input_tokens"],
                         total_output_tokens=final_state["total_output_tokens"],
                         total_llm_calls=final_state["total_llm_calls"])


@app.post("/clear_orders")
async def clear_orders():
    await db.orders.delete_many({})
