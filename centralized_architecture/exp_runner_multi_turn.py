import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import MongoClient
import os
import statistics
import urllib.parse
from collections import Counter
from thefuzz import fuzz

# ---------------- CONFIG ----------------
PLANNER_CART_CHECKOUT_URL = "http://127.0.0.1:8010/cart/checkout"
PLANNER_SEARCH_URL = "http://localhost:8010/cart/add"
DELETE_SEARCH_MEMORY_URL = "http://localhost:8010/delete_search_memory"
DELETE_SHIPMENT_MEMORY_URL = "http://localhost:8010/delete_shipment_memory"

# ---------------------------- Turn 1 Explicit (to create meaningful memory) ---------------
USER_SEARCH_PROMPTS_EXPLICIT = [
    "looking for noise cancelling white headphone  under 300$"
]
USER_SHIPMENT_PROMPTS_EXPLICIT = [
    "Please ship this between Monday to Wednesday with the cheapest shipping option available."
]
USER_SEARCH_PROMPT_GTS_EXPLICIT = [
    {"product": "noise cancelling white headphone", "min_price": None, "max_price": 300}
]
USER_SHIPMENT_PROMPT_GTS_EXPLICIT = [
    {"speed": "cheapest", "eco_friendly": False, "avoid_weekend_delivery": True, "preferred_carrier": None}
]

# ---------------------------- Turn 2 Vague (to check role of memory) ---------------
USER_SEARCH_PROMPTS_VAGUE = [
    "looking for cheap headphone"
]
USER_SHIPMENT_PROMPTS_VAGUE = [
    "Please ship ASAP with the cheapest shipping option available."
]
USER_SEARCH_PROMPT_GTS_VAGUE = [
    {"product": "noise cancelling white headphone", "min_price": None, "max_price": None}
]
USER_SHIPMENT_PROMPT_GTS_VAGUE = [
    {"speed": "cheapest", "eco_friendly": False, "avoid_weekend_delivery": False, "preferred_carrier": None}
]

INIT_STOCK = 5
QTY = 1
USER_ID = "123"

N_TRIALS = 10
MAX_WORKERS = int(N_TRIALS / N_TRIALS)  # Number of concurrent threads
total_runs = 1

DELAY = float(os.environ.get("DELAY", "0"))  # seconds to sleep inside inventory agent
DROP_RATE = int(os.environ.get("DROP_RATE", "0"))  # percent 0-100
atomic_update = False
stateless = False

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("DB_NAME", "ms_baseline")

# clear previous logs
logs = ['logs/order_agent.log', 'logs/inventory_agent.log', 'logs/payment_agent.log',
        'logs/pricing_agent.log', 'logs/planner_agent.log',
        'logs/procurement_agent.log', 'logs/product_search_agent.log',
        'logs/shipment_agent.log',
        'logs/shopping_cart_agent.log']
for log in logs:
    with open(file=log, mode='w') as f:
        f.write('')


def real_db():
    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]
    return client, db


def run_trial_for_full_workflow_multi_turn(trial_id: int, delay: float, drop_rate: int,
                                           user_search_prompt: str, user_shipment_prompt: str,
                                           user_search_prompt_next: str, user_shipment_prompt_next: str):
    result = {"trial": trial_id, "threads": MAX_WORKERS,
              "total_input_tokens": 0, "total_output_tokens": 0,
              "total_llm_calls": 0, "stateless": stateless, "total_without_search_result": 0}

    # delete memory in each trial of multi-session
    if stateless is False:
        requests.delete(url=DELETE_SEARCH_MEMORY_URL + f"?user_id={USER_ID}")
        requests.delete(url=DELETE_SHIPMENT_MEMORY_URL + f"?user_id={USER_ID}")
        print(f"Memory cleaned, Trial {trial_id} started ... ")

    start = time.time()
    try:

        # --------------------------- phase 1: product search & finalize cart -------------------------------
        params = {}
        if stateless is False:
            params["user_id"] = USER_ID

        # --------------------------- product search turn 1 & leave -------------------------------
        r = requests.post(url=PLANNER_SEARCH_URL, params=params, json={"search_prompt": user_search_prompt})
        r.raise_for_status()
        res = r.json()
        # print(f"Result of product search: {res}, latency: {search_latency}")
        if len(res["results"]) == 0:
            result["total_input_tokens"] += res["total_input_tokens"]
            result["total_output_tokens"] += res["total_output_tokens"]
            result["total_llm_calls"] += res["total_llm_calls"]
            result["selected_sku"] = None
            print(f"Trial {trial_id}: Could not find any product")
            return result

        result["total_input_tokens"] += res["total_input_tokens"]
        result["total_output_tokens"] += res["total_output_tokens"]
        result["total_llm_calls"] += res["total_llm_calls"]

        print(f"Trial {trial_id}: Phase 1, Product search turn 1 stop, going to resume product search turn 2")

        # --------------------------- product search turn 2 -------------------------------
        st = time.time()
        params = {}
        if stateless is False:
            params["user_id"] = USER_ID

        r = requests.post(url=PLANNER_SEARCH_URL, params=params, json={"search_prompt": user_search_prompt_next})
        r.raise_for_status()
        et = time.time()
        search_latency = round((et - st), 3)
        res = r.json()
        # print(f"Result of product search: {res}, latency: {search_latency}")
        if len(res["results"]) == 0:
            search_filters = res["search_filters"]
            result["total_input_tokens"] += res["total_input_tokens"]
            result["total_output_tokens"] += res["total_output_tokens"]
            result["total_llm_calls"] += res["total_llm_calls"]
            result["search_filters"] = search_filters
            result["previous_search_memory"] = res["previous_search_memory"]
            result["current_search_memory"] = res["current_search_memory"]

            result["search_latency"] = search_latency
            result["selected_sku"] = None
            result["total_without_search_result"] += 1
            print(f"Trial {trial_id}: Could not find any product")
            return result

        sku = res["results"][0]["sku"]
        search_filters = res["search_filters"]
        result["total_input_tokens"] += res["total_input_tokens"]
        result["total_output_tokens"] += res["total_output_tokens"]
        result["total_llm_calls"] += res["total_llm_calls"]
        result["search_filters"] = search_filters
        result["previous_search_memory"] = res["previous_search_memory"]
        result["current_search_memory"] = res["current_search_memory"]
        result["search_latency"] = search_latency
        result["selected_sku"] = sku
        cart_id = res["cart_id"]
        result["cart_id"] = res["cart_id"]

        # todo: initial shipment prompt and leave

        # ------------------------------ phase 2: complete main workflow for purchase cart ---------------------------
        resp = requests.post(PLANNER_CART_CHECKOUT_URL + f"?user_id={USER_ID}" if stateless is False else PLANNER_CART_CHECKOUT_URL,
                             json={"cart_id": cart_id, "shipment_prompt": user_shipment_prompt},
                             timeout=30)
        elapsed = time.time() - start
        if resp.status_code == 200:
            res = resp.json()
            result["order_id"] = res["order_id"]
            result["status"] = res["status"]
            result["shipment_prefs"] = res["shipment_prefs"]
            result["previous_shipment_memory"] = res["previous_shipment_memory"]
            result["current_shipment_memory"] = res["current_shipment_memory"]
            result["total_input_tokens"] += res["total_input_tokens"]
            result["total_output_tokens"] += res["total_output_tokens"]
            result["total_llm_calls"] += res["total_llm_calls"]

            result["total_latency"] = round(elapsed, 3)
            print(f"Trial {trial_id}: {result}")
            return result
        else:
            print(f"Trial {trial_id}: ERROR: {resp.json()}")
            return {"trial": trial_id, "status": "error", "elapsed": round(elapsed, 3)}

    except Exception as e:
        elapsed = time.time() - start
        print(f"Trial {trial_id}: Exception {e}")
        return {"trial": trial_id, "status": "error", "elapsed": round(elapsed, 3)}


# ---------------------------- Checking invariants for data consistency ------------------
def get_final_state(sku: str):
    _, db = real_db()
    final_stock = db.inventory.find_one({"sku": sku})
    stock_left = final_stock["stock"] if final_stock else 0
    total_completed_orders = db.orders.count_documents({"status": "COMPLETED"})
    total_pending_orders = db.orders.count_documents({"status": "INIT"})
    total_oos_orders = db.orders.count_documents({"status": "OUT_OF_STOCK"})
    total_payments = db.payments.count_documents({"status": "SUCCESS"})
    total_shipment_bookings = db.shipments.count_documents({})

    # basic heuristics used previously: compute failure rate loosely
    final_ec_state = "SUCCESS"
    failure_rate = 0.0
    expected_total_reserved = int((INIT_STOCK) / QTY)  # approximate expectation from your earlier code

    if stock_left < 0:
        failure_rate += -stock_left / QTY
        final_ec_state = "FAIL"
    elif stock_left + total_completed_orders != expected_total_reserved:
        failure_rate += abs((total_completed_orders - (expected_total_reserved - stock_left)))
        final_ec_state = "FAIL"
    if total_pending_orders > 0:
        failure_rate += total_pending_orders
        final_ec_state = "FAIL"
    if total_payments != expected_total_reserved:
        failure_rate += expected_total_reserved - total_payments
        final_ec_state = "FAIL"
    if total_shipment_bookings != expected_total_reserved:
        failure_rate += expected_total_reserved - total_shipment_bookings
        final_ec_state = "FAIL"
    return stock_left, total_completed_orders, total_pending_orders, total_oos_orders, expected_total_reserved, \
           total_shipment_bookings, total_payments, \
           final_ec_state, failure_rate


class check_shipment_semantic_consistency_and_reproducibility():

    def __init__(self):
        pass

    def filters_match(self, pred, gt):
        speed_match = fuzz.ratio(str(pred.get("speed")).lower(), str(gt.get("speed")).lower()) > 95
        if pred.get("eco_friendly") is None:
            pred["eco_friendly"] = False
        eco_friendly_match = pred.get("eco_friendly") == gt.get("eco_friendly")
        avoid_weekend_delivery_match = pred.get("avoid_weekend_delivery") == gt.get("avoid_weekend_delivery")
        if pred.get("preferred_carrier") and gt.get("preferred_carrier"):
            preferred_carrier_match = fuzz.ratio(str(pred.get("preferred_carrier")).lower(),
                                                 str(gt.get("preferred_carrier")).lower()) > 95
        elif pred.get("preferred_carrier") and gt.get("preferred_carrier") is not None:
            preferred_carrier_match = False
        else:
            preferred_carrier_match = True

        return (
                speed_match
                and
                eco_friendly_match
                and
                avoid_weekend_delivery_match
                and
                preferred_carrier_match
        )

    def compute_PAR(self, filters, gt):
        matches = sum(
            self.filters_match(f, gt) for f in filters
        )
        return matches / len(filters)

    def normalize_filters(self, f, gt):
        product_match = fuzz.ratio(str(f.get("speed")).lower(), str(gt.get("speed")).lower()) > 95
        if product_match:
            f["speed"] = str(gt.get("speed")).lower()
        if f.get("preferred_carrier") and gt.get("preferred_carrier"):
            preferred_carrier_match = fuzz.ratio(str(f.get("preferred_carrier")).lower(),
                                                 str(gt.get("preferred_carrier")).lower()) > 95
            if preferred_carrier_match:
                f["preferred_carrier"] = str(gt.get("preferred_carrier")).lower()

        return json.dumps(f, sort_keys=True)

    def compute_PRR(self, filters, gt):
        normalized = [self.normalize_filters(f, gt) for f in filters]
        most_common, count = Counter(normalized).most_common(1)[0]
        return count / len(filters)


class check_search_semantic_consistency_and_reproducibility():

    def __init__(self):
        pass

    def filters_match(self, pred, gt):
        product_match = fuzz.ratio(str(pred.get("product")).lower(), str(gt.get("product")).lower()) > 85
        if pred.get("min_price") == 0:
            pred["min_price"] = None
        if gt.get("min_price") == 0:
            gt["min_price"] = None
        min_price_match = pred.get("min_price") == gt.get("min_price")
        max_price_match = pred.get("max_price") == gt.get("max_price")

        return (
                product_match
                and
                min_price_match
                and
                max_price_match
        )

    def compute_PAR(self, filters, gt):
        matches = sum(
            self.filters_match(f, gt) for f in filters
        )
        return matches / len(filters)

    def normalize_filters(self, f, gt):
        product_match = fuzz.ratio(str(f.get("product")).lower(), str(gt.get("product")).lower()) > 85
        if product_match:
            f["product"] = str(gt.get("product")).lower()
        return json.dumps(f, sort_keys=True)

    def compute_PRR(self, filters, gt):
        normalized = [self.normalize_filters(f, gt) for f in filters]
        most_common, count = Counter(normalized).most_common(1)[0]
        return count / len(filters)


check_search_semantic_consistency_and_reproducibility = check_search_semantic_consistency_and_reproducibility()
check_shipment_semantic_consistency_and_reproducibility = check_shipment_semantic_consistency_and_reproducibility()


def run(user_search_prompt, user_shipment_prompt, user_search_prompt_next, user_shipment_prompt_next,
        user_search_prompt_gt, user_shipment_prompt_gt, run_results, i, j):
    requests.post("http://localhost:8000/clear_orders", json={})
    requests.post("http://localhost:8001/reset_stocks", json={
        "items": [
            {"sku": "4cc0770f-91bc-4c0d-a26f-7b872f02ca94", "stock": INIT_STOCK},
            {"sku": "b2926dc2-cc6d-4c3e-ae40-7a127c173b16", "stock": INIT_STOCK},
        ]
    })
    requests.post("http://localhost:8007/clear_payments", json={})
    requests.post("http://localhost:8006/clear_bookings", json={})
    requests.post("http://localhost:8003/clear_carts", json={})

    print('DB state cleaned ...')

    results = []

    # ---------------- PARALLEL EXECUTION (x N_Trials) ----------------
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_trial_for_full_workflow_multi_turn, n, DELAY, DROP_RATE,
                                   user_search_prompt, user_shipment_prompt,
                                   user_search_prompt_next, user_shipment_prompt_next)
                   for n in range(1, N_TRIALS + 1)]
        for future in as_completed(futures):
            results.append(future.result())

    sku = None
    for r in results:
        if r["selected_sku"] is not None:
            sku = r["selected_sku"]
    print(f"Selected SKU: {sku}")
    if sku is None:
        run_results.append({"run_number": i + 1, "prompt_number": j + 1,
                            "trial_results": results, "final_summary": None})
        return

    # --------------- check invariants correctness -------------
    print("Checking invariants correctness ... ")
    stock_left, total_completed_orders, total_pending_orders, total_oos_orders, expected_total_reserved, \
    total_shipment_bookings, total_payments, \
    final_ec_state, failure_rate = get_final_state(sku=sku)

    summary = {
        "n_trials": N_TRIALS,
        "delay": DELAY,
        "drop_rate": DROP_RATE,
        "n_threads": MAX_WORKERS,
        "stateless": stateless,
        "user_search_prompt": user_search_prompt,
        "user_shipment_prompt": user_shipment_prompt,
        "user_search_prompt_gt": user_search_prompt_gt,
        "user_shipment_prompt_gt": user_shipment_prompt_gt,
        "total_without_search_result": sum([x['total_without_search_result'] for x in results]),
        "stock_left": stock_left,
        "total_completed_orders": total_completed_orders,
        "total_pending_orders": total_pending_orders,
        "total_oos_orders": total_oos_orders,
        "expected_total_reserved": expected_total_reserved,
        "total_shipment_bookings": total_shipment_bookings,
        "total_payments": total_payments,
        "final_ec_state": final_ec_state,
        "failure_rate": (failure_rate / N_TRIALS) * 100,
        "avg_total_latency": statistics.mean([x['total_latency'] for x in results if x.get('total_latency')]),
        "std_total_latency": statistics.stdev([x['total_latency'] for x in results if x.get('total_latency')]),
        "p95_total_latency":
            statistics.quantiles(data=[x['total_latency'] for x in results if x.get('total_latency')], n=100)[95],
        "med_total_latency": statistics.median([x['total_latency'] for x in results if x.get('total_latency')]),
        "avg_search_latency": statistics.mean([x['search_latency'] for x in results if x.get('search_latency')]),
        "std_search_latency": statistics.stdev([x['search_latency'] for x in results if x.get('search_latency')]),
        "p95_search_latency":
            statistics.quantiles(data=[x['search_latency'] for x in results if x.get('search_latency')], n=100)[95],
        "med_search_latency": statistics.median([x['search_latency'] for x in results if x.get('search_latency')]),
        "total_input_tokens": sum([x['total_input_tokens'] for x in results]),
        "total_output_tokens": sum([x['total_output_tokens'] for x in results]),
        "total_llm_calls": sum([x['total_llm_calls'] for x in results]),
        "search_semantic_consistency": check_search_semantic_consistency_and_reproducibility.
            compute_PAR(filters=[x['search_filters'] for x in results if x.get('search_filters')], gt=user_search_prompt_gt),
        "search_reproducibility": check_search_semantic_consistency_and_reproducibility.
            compute_PRR(filters=[x['search_filters'] for x in results if x.get('search_filters')], gt=user_search_prompt_gt),
        "shipment_semantic_consistency": check_shipment_semantic_consistency_and_reproducibility.
            compute_PAR(filters=[x['shipment_prefs'] for x in results if x.get('shipment_prefs')], gt=user_shipment_prompt_gt),
        "shipment_reproducibility": check_shipment_semantic_consistency_and_reproducibility.
            compute_PRR(filters=[x['shipment_prefs'] for x in results if x.get('shipment_prefs')], gt=user_shipment_prompt_gt),

    }
    print("Final summary:", summary)
    run_results.append({"run_number": i + 1, "prompt_number": j + 1,
                        "trial_results": results, "final_summary": summary})
    print(f"Run {i + 1} Done,\n-----------------------------------------")


if __name__ == '__main__':

    report_name = f"results/multi-turn/p2p_architecture_stateless_{stateless}_delay_{DELAY}_drop_{DROP_RATE}_results.json"
    with open(report_name, "w") as f:
        f.write("")

    run_results = []

    for i in range(total_runs):

        for j in range(len(USER_SEARCH_PROMPTS_EXPLICIT)):
            print(f"Prompts round {j} ")

            # first call with explicit prompts
            user_search_prompt = USER_SEARCH_PROMPTS_EXPLICIT[j]
            user_shipment_prompt = USER_SHIPMENT_PROMPTS_EXPLICIT[j]
            user_search_prompt_gt = USER_SEARCH_PROMPT_GTS_EXPLICIT[j]
            user_shipment_prompt_gt = USER_SHIPMENT_PROMPT_GTS_EXPLICIT[j]

            # then call with vague prompts (so stateful agents constructs memory and defuse LLM variability to infer)
            user_search_prompt_next = USER_SEARCH_PROMPTS_VAGUE[j]
            user_shipment_prompt_next = USER_SHIPMENT_PROMPTS_VAGUE[j]

            run(user_search_prompt, user_shipment_prompt,
                user_search_prompt_next, user_shipment_prompt_next,
                user_search_prompt_gt, user_shipment_prompt_gt,
                run_results, i, j)

    # Save all results
    with open(report_name, "w") as f:
        f.write("\n")
        json.dump(run_results, f)
        f.write("\n\n")
