from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_health_and_keyword_search():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["symbols"] > 10

    response = client.post("/search", json={"query": "retry failed payment", "mode": "keyword", "limit": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "keyword"
    assert data["results"][0]["name"] == "retry_payment"
    assert data["results"][0]["path"] == "payments.py"


def test_request_validation():
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "test", "mode": "unknown"}).status_code == 422
    assert client.post("/search", json={"query": "test", "limit": 100}).status_code == 422
