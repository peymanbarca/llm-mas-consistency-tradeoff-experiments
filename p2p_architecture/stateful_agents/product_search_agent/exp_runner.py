import asyncio
import httpx
import json
from collections import Counter
from typing import Dict, List, Optional
import urllib.parse
import time
import requests
from thefuzz import fuzz

SEARCH_URL = "http://localhost:8008/search"
DELETE_MEMORY_URL = "http://localhost:8008/delete_memory"
N_RUNS = 2

logs = ['../logs/product_search_agent.log']
for log in logs:
    with open(file=log, mode='w') as f:
        f.write('')

report_file = "results/product_search_agent_stateful.txt"
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
            params = {
                "q": urllib.parse.quote_plus(prompt),
            }
            if stateless is False:
                params["user_id"] = user_id

            r = await client.get(SEARCH_URL, params=params)
            r.raise_for_status()
            et = time.time()
            data = r.json()

            results.append({
                "filters": data["search_filters"],
                "previous_memory": data["previous_memory"],
                "current_memory": data["current_memory"],
                "total_input_tokens": data["total_input_tokens"],
                "total_output_tokens": data["total_output_tokens"],
                "total_llm_calls": data["total_llm_calls"],
                "latency": round((et - st), 3)
            })

    return results


def filters_match(pred, gt):
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


def compute_PAR(trials, ground_truth):
    matches = sum(
        filters_match(t["filters"], ground_truth)
        for t in trials
    )
    return matches / len(trials)


def normalize_filters(f, gt):
    product_match = fuzz.ratio(str(f.get("product")).lower(), str(gt.get("product")).lower()) > 85
    if product_match:
        f["product"] = str(gt.get("product")).lower()
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

    sample_prompt = "looking for noise cancelling white headphone under 300$"
    sample_gt = {
        "product": "noise cancelling white headphone",
        "min_price": None,
        "max_price": 300
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
