# RAG Air Quality Project

Projekt implementuje prosty system RAG dla danych jakości powietrza z użyciem FastAPI, ChromaDB i embeddingów Sentence Transformers. Dodatkowo projekt spełnia wymagania zaliczeniowe dotyczące Docker Compose, trwałości zasobów, Prometheusa i Grafany.

## Technologie
- FastAPI
- ChromaDB
- sentence-transformers/all-MiniLM-L6-v2
- Prometheus (`prometheus-client`)
- Grafana
- Docker Compose

## Uruchomienie
```bash
docker compose up --build
```

## Usługi
- Aplikacja: `http://localhost:8000`
- Healthcheck: `http://localhost:8000/health`
- Metryki: `http://localhost:8000/metrics`
- Informacje o bazie wektorowej: `http://localhost:8000/vector-db`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Przykładowe wywołania
Najpierw załaduj dane do bazy wektorowej:
```bash
curl "http://localhost:8000/run_pipeline"
```

Pokaż zawartość bazy wektorowej:
```bash
curl "http://localhost:8000/vector-db"
```

Zadaj pytanie do RAG:
```bash
curl "http://localhost:8000/query?question=Jakie są dane o PM2.5?"
```

Uruchom prognozę:
```bash
curl "http://localhost:8000/forecast"
```

## Trwałość zasobów
Docker Compose używa wolumenów:
- `chroma_data`
- `prometheus_data`
- `grafana_data`

Dzięki temu dane ChromaDB, Prometheusa i Grafany są zachowane po restarcie kontenerów.

## Metryki
Aplikacja wystawia endpoint `/metrics`, z którego Prometheus pobiera dane. Dashboard Grafany jest provisionowany z pliku i pokazuje:
- liczbę zapytań `/query`
- liczbę uruchomień pipeline
- średni czas odpowiedzi `/query`
- liczbę rekordów w bazie wektorowej
