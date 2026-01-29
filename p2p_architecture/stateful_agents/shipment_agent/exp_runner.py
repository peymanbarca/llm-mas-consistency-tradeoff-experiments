import asyncio
import httpx
import json
from collections import Counter
from typing import Dict, List, Optional
import time
import requests
from thefuzz import fuzz

BOOK_URL = "http://localhost:8006/book"
DELETE_MEMORY_URL = "http://localhost:8006/delete_memory"
N_RUNS = 2

logs = ['../logs/shipment_agent.log']
for log in logs:
    with open(file=log, mode='w') as f:
        f.write('')

report_file = "results/shipment_agent_stateful.txt"
with open(report_file, "w") as f:
    f.write("")


async def run_trials(
    prompt: str,
    user_id: Optional[int],
    stateless: bool = False
):
    results = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(N_RUNS):
            st = time.time()
            params = {}
            if stateless is False:
                params["user_id"] = user_id
            body = {"order_id": "order_123", "address": "210, NW6866 kings street", "main_query": prompt}

            r = await client.post(BOOK_URL, params=params, json=body)
            r.raise_for_status()
            et = time.time()
            data = r.json()

            results.append({
                "filters": data["shipment_prefs"],
                "previous_memory": data["previous_memory"],
                "current_memory": data["current_memory"],
                "total_input_tokens": data["total_input_tokens"],
                "total_output_tokens": data["total_output_tokens"],
                "total_llm_calls": data["total_llm_calls"],
                "latency": round((et - st), 3)
            })

    return results


def filters_match(pred, gt):
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


def compute_PAR(trials, ground_truth):
    matches = sum(
        filters_match(t["filters"], ground_truth)
        for t in trials
    )
    return matches / len(trials)


def normalize_filters(f, gt):
    product_match = fuzz.ratio(str(f.get("speed")).lower(), str(gt.get("speed")).lower()) > 95
    if product_match:
        f["speed"] = str(gt.get("speed")).lower()
    if f.get("preferred_carrier") and gt.get("preferred_carrier"):
        preferred_carrier_match = fuzz.ratio(str(f.get("preferred_carrier")).lower(),
                                             str(gt.get("preferred_carrier")).lower()) > 95
        if preferred_carrier_match:
            f["preferred_carrier"] = str(gt.get("preferred_carrier")).lower()

    return json.dumps(f, sort_keys=True)


def compute_PRR(trials, gt):
    normalized = [normalize_filters(t["filters"], gt) for t in trials]
    most_common, count = Counter(normalized).most_common(1)[0]
    return count / len(trials)


async def run_experiment(prompt, gt, user_id):

    stateless = await run_trials(
        prompt=prompt,
        user_id=None,
        stateless=True
    )

    with open(report_file, "a") as f:
        f.write("Stateless Trials Results:\n")
        f.write(f"{json.dumps(stateless)}")

    stateful = await run_trials(
        prompt=prompt,
        user_id=user_id,
        stateless=False
    )

    with open(report_file, "a") as f:
        f.write("\n\nStateful Trials Results:\n")
        f.write(f"{json.dumps(stateful)}")

    return {
        "prompt": prompt,
        "gt": gt,
        "stateful": {
            "PAR": compute_PAR(stateful, gt),
            "PRR": compute_PRR(stateful, gt)
        },
        "stateless": {
            "PAR": compute_PAR(stateless, gt),
            "PRR": compute_PRR(stateless, gt)
        }
    }


async def main():

    sample_prompt = "Please ship this between Monday to Wednesday with the cheapest shipping option available."
    sample_gt = {
        "speed": "cheapest",
        "eco_friendly": False,
        "avoid_weekend_delivery": True,
        "preferred_carrier": None
    }
    user_id = 123

    requests.delete(url=DELETE_MEMORY_URL + f"?user_id={user_id}")

    results = await run_experiment(sample_prompt, sample_gt, user_id)
    results = json.dumps(results, indent=4)
    print('results: \n', results)

    with open(report_file, "a") as f:
        f.write("\n\nFinal Trials Results:\n")
        f.write(f"{results}")


if __name__ == '__main__':
    asyncio.run(main())
