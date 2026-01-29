import uvicorn
from fastapi import FastAPI
import sys

app = FastAPI()

agent_name = "product_search_agent_stateful"
port = 8008

if __name__ == "__main__":
    uvicorn.run(f"{agent_name}:app", host="0.0.0.0", port=port, reload=True)
