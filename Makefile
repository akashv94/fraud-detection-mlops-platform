.PHONY: install train test api ui docker-up docker-down drift

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	python -m src.models.train

test:
	pytest -v

api:
	uvicorn src.api.main:app --reload

ui:
	streamlit run dashboard/app.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down

drift:
	python -m src.monitoring.drift
