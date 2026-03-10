import os
import time

import chromadb
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression

app = FastAPI(title="RAG Air Quality Project")

REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
INDEXED_RECORDS = Gauge(
    "app_indexed_records",
    "Number of records indexed in vector database",
)
QUERY_COUNT = Counter("app_queries_total", "Number of query requests")
FORECAST_COUNT = Counter("app_forecasts_total", "Number of forecast requests")
PIPELINE_RUNS = Counter("app_pipeline_runs_total", "Number of pipeline runs")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

CHROMA_PATH = os.getenv("CHROMA_PATH", "/data/chroma")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="air_quality")


def track_request(endpoint_name: str, status_code: str, elapsed: float, method: str = "GET") -> None:
    REQUEST_COUNT.labels(method=method, endpoint=endpoint_name, status=status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint_name).observe(elapsed)


def fetch_air_quality() -> dict:
    url = "https://api.openaq.org/v2/latest"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def process_data(data: dict) -> pd.DataFrame:
    records = []
    for result in data.get("results", []):
        for measurement in result.get("measurements", []):
            records.append(
                {
                    "location": result.get("location"),
                    "parameter": measurement.get("parameter"),
                    "value": measurement.get("value"),
                    "unit": measurement.get("unit"),
                }
            )
    return pd.DataFrame(records)


def index_documents(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    documents = df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()
    embeddings = embedding_model.encode(documents).tolist()
    ids = [f"doc_{i}" for i in range(len(documents))]

    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    collection.add(documents=documents, embeddings=embeddings, ids=ids)
    INDEXED_RECORDS.set(len(documents))
    return len(documents)


def retrieve_context(query: str) -> str:
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    docs = results.get("documents", [])
    if docs and docs[0]:
        return " ".join(docs[0])
    return ""


def forecast_pm(values) -> float:
    X = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)
    model = LinearRegression()
    model.fit(X, y)
    next_value = model.predict([[len(values)]])[0]
    return float(next_value)


@app.get("/")
def root():
    return {
        "message": "RAG Air Quality Project",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/run_pipeline")
def run_pipeline():
    start = time.perf_counter()
    try:
        data = fetch_air_quality()
        df = process_data(data)
        if df.empty:
            track_request("/run_pipeline", "400", time.perf_counter() - start)
            raise HTTPException(status_code=400, detail="No data fetched")

        indexed = index_documents(df)
        PIPELINE_RUNS.inc()
        elapsed = time.perf_counter() - start
        track_request("/run_pipeline", "200", elapsed)

        return {
            "status": "Data indexed successfully",
            "records": indexed,
            "chroma_path": CHROMA_PATH,
            "collection": "air_quality",
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        track_request("/run_pipeline", "500", elapsed)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query")
def query_agent(question: str):
    start = time.perf_counter()
    try:
        QUERY_COUNT.inc()
        context = retrieve_context(question)
        response = f"Na podstawie danych: {context[:1000]}\n\nOdpowiedź na pytanie: {question}"
        elapsed = time.perf_counter() - start
        track_request("/query", "200", elapsed)
        return {"response": response}
    except Exception as e:
        elapsed = time.perf_counter() - start
        track_request("/query", "500", elapsed)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast")
def forecast():
    start = time.perf_counter()
    try:
        FORECAST_COUNT.inc()
        data = fetch_air_quality()
        df = process_data(data)
        pm25 = df[df["parameter"] == "pm25"]["value"].tolist()

        if len(pm25) < 5:
            elapsed = time.perf_counter() - start
            track_request("/forecast", "400", elapsed)
            raise HTTPException(status_code=400, detail="Not enough PM2.5 data for forecasting")

        prediction = forecast_pm(pm25[:10])
        elapsed = time.perf_counter() - start
        track_request("/forecast", "200", elapsed)
        return {"predicted_pm25_next": prediction}
    except Exception as e:
        elapsed = time.perf_counter() - start
        track_request("/forecast", "500", elapsed)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector-db")
def vector_db_info():
    count = collection.count()
    sample = collection.peek(limit=5)
    return {
        "collection": "air_quality",
        "chroma_path": CHROMA_PATH,
        "document_count": count,
        "sample": sample,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
