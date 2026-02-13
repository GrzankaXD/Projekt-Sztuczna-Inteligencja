
import requests
import pandas as pd
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI()

# Load embedding model (identical in both environments)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection("air_quality")

def fetch_air_quality():
    url = "https://api.openaq.org/v2/latest"
    response = requests.get(url)
    return response.json()

def process_data(data):
    records = []
    for result in data.get("results", []):
        for measurement in result.get("measurements", []):
            records.append({
                "location": result.get("location"),
                "parameter": measurement.get("parameter"),
                "value": measurement.get("value"),
                "unit": measurement.get("unit")
            })
    return pd.DataFrame(records)

def index_documents(df):
    documents = df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()
    embeddings = embedding_model.encode(documents).tolist()
    ids = [str(i) for i in range(len(documents))]
    collection.add(documents=documents, embeddings=embeddings, ids=ids)

def retrieve_context(query):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    return " ".join(results["documents"][0]) if results["documents"] else ""

def forecast_pm(values):
    X = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)
    model = LinearRegression()
    model.fit(X, y)
    next_value = model.predict([[len(values)]])[0]
    return float(next_value)

@app.get("/run_pipeline")
def run_pipeline():
    data = fetch_air_quality()
    df = process_data(data)
    if df.empty:
        return {"error": "No data fetched"}
    index_documents(df)
    return {"status": "Data indexed successfully", "records": len(df)}

@app.get("/query")
def query_agent(question: str):
    context = retrieve_context(question)
    response = f"Na podstawie danych: {context[:1000]} ... \nOdpowiedź na pytanie: {question}"
    return {"response": response}

@app.get("/forecast")
def forecast():
    data = fetch_air_quality()
    df = process_data(data)
    pm25 = df[df["parameter"] == "pm25"]["value"].tolist()
    if len(pm25) < 5:
        return {"error": "Not enough PM2.5 data for forecasting"}
    prediction = forecast_pm(pm25[:10])
    return {"predicted_pm25_next": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)