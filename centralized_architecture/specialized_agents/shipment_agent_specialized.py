import os
import logging
import time
import uuid
import datetime
import httpx

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Optional, Literal
from motor.motor_asyncio import AsyncIOMotorClient
from httpx import AsyncClient
import json
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
import asyncio

################################
# load_memory → LLM infer shipment params → carrier_booking_tool(params) → LLM verify shipment booking → update_memory
################################

logger = logging.getLogger("shipment_agent")
logging.basicConfig(
    filename='../logs/shipment_agent.log',
    level=logging.INFO,  # Log all messages with severity DEBUG or higher
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the message format
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8006))

llm = ChatOllama(model="gpt-oss", temperature=0.0, reasoning=False)

app = FastAPI(title="Shipment Booking Agent")

# DB client will be set on startup
db_client: Optional[AsyncIOMotorClient] = None
db = None


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


class ShipmentState(TypedDict):
    request: Dict[str, Any]
    user_id: Optional[str]
    user_id: str
    previous_memory_summary: Optional[str]
    total_input_tokens: int
    total_output_tokens: int
    total_llm_calls: int
    shipment_prefs: Optional[Dict[str, Any]]
    carrier_result: Dict[str, Any]
    result: Dict[str, Any]


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


# Tool: call external carrier
async def carrier_booking_tool(state: ShipmentState) -> ShipmentState:
    logger.info(f'Calling carrier_booking_tool ... \n Current State is {state}')
    print(f'Calling carrier_booking_tool ... \n Current State is {state}')

    prefs = state.get("shipment_prefs", {})
    logger.info(f"Booking shipment with prefs: {prefs}")

    # Simulate carrier API latency
    time.sleep(0.2)

    tracking_id = str(uuid.uuid4())
    state["carrier_result"] = {
        "tracking_id": tracking_id,
        "carrier": prefs.preferred_carrier if prefs.preferred_carrier else "MockCarrier",
        "speed": prefs.speed if prefs.speed else "standard"
    }
    logger.info(f'Response state of carrier_booking_tool ==> {state}, \n-------------------------------------')
    print(f'Response state of carrier_booking_tool ==> {state}, \n-------------------------------------')
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


async def verify_shipment_reasoning(state: ShipmentState) -> ShipmentState:
    prompt = f"""
    Verify shipment booking result.

    Rules:
    - tracking_id must exist
    - Generate shipment_id as UUID
    - success = true only if tracking_id exists

    - Return only a JSON (not python code) with below schema without intermediate reasoning and analysis text:
    Schema:
    {{
      "shipment_id": string,
      "success": bool
    }}

    Input:
    CARRIER_RESULT = {json.dumps(state["carrier_result"])}


    """

    logger.info(f'verify_shipment_reasoning_node -> LLM Call Prompt: {prompt}')

    st = time.time()
    response = await asyncio.to_thread(llm.invoke, prompt)
    raw = response.text()
    et = time.time()

    input_tokens = response.usage_metadata.get("input_tokens")
    output_tokens = response.usage_metadata.get("output_tokens")
    total_tokens = response.usage_metadata.get("total_tokens")
    logger.info(f'verify_shipment_reasoning_node -> LLM Raw response: {raw}')

    logger.info(
        f'verify_shipment_reasoning_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
        f' total_tokens: {total_tokens}, Took: f{round((et - st), 3)}')
    print(
        f'verify_shipment_reasoning_node -> LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
        f' total_tokens: {total_tokens}, Took: f{round((et - st), 3)}')

    parsed = parse_json_response(raw)
    state["result"] = parsed
    state["result"]["tracking_id"] = state["carrier_result"]["tracking_id"]

    state["total_input_tokens"] += input_tokens
    state["total_output_tokens"] += output_tokens
    state["total_llm_calls"] += 1

    return state


def build_shipment_graph():
    graph = StateGraph(ShipmentState)

    graph.add_node("carrier_call", carrier_booking_tool)
    graph.add_node("verify", verify_shipment_reasoning)

    graph.set_entry_point("carrier_call")
    graph.add_edge("carrier_call", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


shipment_graph = build_shipment_graph()


@app.post("/book", response_model=ShipmentResponse)
async def book_shipment(req: ShipmentRequest, user_id: Optional[str] = Query(None)):
    try:
        state = {
            "request": req.dict(),
            "user_id": user_id,
            "previous_memory_summary": req.previous_memory_summary,
            "shipment_prefs": req.shipment_prefs,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_llm_calls": 0,
            "carrier_result": {},
            "result": {}
        }
        logger.info(f'Request for book_shipment, req = {req}, state={state}')
        print(f'Request for book_shipment, req = {req}, state={state}')

        out = await shipment_graph.ainvoke(state)
        logger.info(f'Request for process_payment processed successfully, req = {req}, result={out.get("result")}')
        print(f'Request for process_payment processed successfully, req = {req}, result={out.get("result")}')

        success = out["result"]["success"]
        if success is None or success is not True:
            raise HTTPException(status_code=500, detail='Carrier unavailable')

        shipment_id = out["result"]["tracking_id"]
        tracking_id = out["result"]["shipment_id"]

        doc = {
            "shipment_id": shipment_id,
            "order_id": req.order_id,
            "address": req.address,
            "tracking_id": tracking_id,
            "shipment_prefs": dict(out["shipment_prefs"]),
            "created_at": datetime.datetime.utcnow()
        }

        await db.shipments.insert_one(doc)

        response = ShipmentResponse(
            previous_memory=out["previous_memory_summary"],
            total_input_tokens=out["total_input_tokens"],
            total_output_tokens=out["total_output_tokens"],
            total_llm_calls=out["total_llm_calls"],
            shipment_prefs=dict(out["shipment_prefs"]),
            shipment_id=out["result"]["shipment_id"],
            tracking_id=out["result"]["tracking_id"]
        )

        return response

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/clear_bookings")
async def clear_bookings():
    await db.shipments.delete_many({})
