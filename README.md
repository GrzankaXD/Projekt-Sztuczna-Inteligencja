
# RAG Air Quality Project

## Opis projektu
Projekt implementuje dwóch agentów RAG (HuggingFace + Google Colab) 
z wykorzystaniem identycznego modelu językowego, embeddingów oraz bazy wektorowej (ChromaDB).

Model embeddingów:
- sentence-transformers/all-MiniLM-L6-v2

Baza wektorowa:
- ChromaDB

## Funkcjonalności
- Pobieranie danych jakości powietrza (OpenAQ API)
- Indeksowanie danych do bazy wektorowej
- Mechanizm Retrieval-Augmented Generation (RAG)
- Prognozowanie PM2.5 (Linear Regression)

## Uruchomienie (Docker)

1. Zbuduj obraz:
   docker build -t rag-air-quality .

2. Uruchom kontener:
   docker run -p 8000:8000 rag-air-quality

3. Endpointy:
   /run_pipeline
   /query?question=...
   /forecast

## Implementacja w Google Colab

1. Zainstaluj Docker:
   !apt install docker.io

2. Skopiuj pliki projektu
3. Zbuduj i uruchom kontener

## Implementacja w Hugging Face

1. Utwórz Space typu Docker
2. Wgraj pliki projektu
3. HF automatycznie zbuduje obraz

## Porównanie

W obu środowiskach:
- Ten sam model embeddingów
- Ta sama baza ChromaDB
- Identyczna konfiguracja chunkowania
- Te same dane wejściowe

Metryki porównania:
- Czas odpowiedzi
- Zużycie pamięci
- Stabilność działania
- Trafność odpowiedzi
