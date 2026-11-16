from app.schemas import EXAMPLE


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["database"] == "up"


def test_predict_returns_risk_and_explanation(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["risk_probability"] <= 1
    assert body["risk_level"] in {"low", "elevated", "high"}
    assert len(body["top_factors"]) == 5
    assert body["model"] == {"name": "heart-risk", "version": "7"}


def test_prediction_is_stored_in_database(client):
    created = client.post("/predict", json=EXAMPLE).json()
    stored = client.get(f"/predictions/{created['id']}").json()
    assert stored["risk_probability"] == created["risk_probability"]
    assert stored["input"]["Sex"] == "Male"
    assert client.get("/predictions").json()[0]["id"] == created["id"]
    stats = client.get("/predictions/stats").json()
    assert stats["7"]["count"] >= 1


def test_missing_answers_are_rejected(client):
    # A missing answer means "higher risk" to the model (informative missingness), so it's refused
    incomplete = {k: v for k, v in EXAMPLE.items() if k != "HadStroke"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_bmi_or_height_and_weight_required(client):
    no_size = {k: v for k, v in EXAMPLE.items() if k not in ("HeightInMeters", "WeightInKilograms")}
    assert client.post("/predict", json=no_size).status_code == 422
    assert client.post("/predict", json={**no_size, "BMI": 27.5}).status_code == 200


def test_invalid_category_is_rejected(client):
    r = client.post("/predict", json={**EXAMPLE, "Sex": "Mael"})
    assert r.status_code == 422


def test_unknown_field_is_rejected(client):
    r = client.post("/predict", json={**EXAMPLE, "HadAngina": "Yes"})
    assert r.status_code == 422


def test_unknown_prediction_returns_404(client):
    assert client.get("/predictions/999999").status_code == 404


def test_version_shows_lineage(client):
    body = client.get("/version").json()
    assert body["model_version"] == "7"
    assert body["training_git_commit"] == "test"
