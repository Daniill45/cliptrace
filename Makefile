.PHONY: check up down logs worker-logs ps migrate test lint format typecheck eval-retrieval

PODMAN ?= podman
COMPOSE ?= $(PODMAN) compose
PODMAN_COMPOSE_CONFIG ?= $(CURDIR)/.podman
TEST_COMPOSE ?= $(COMPOSE) --profile tools
EVAL_DATASET ?= eval/retrieval_smoke.json
EVAL_CUTOFFS ?= --k 1 --k 3 --k 5
export DOCKER_CONFIG := $(PODMAN_COMPOSE_CONFIG)

check:
	@test -f .env || { echo "Файл .env не найден. Выполните: cp .env.example .env"; exit 1; }
	@command -v $(PODMAN) >/dev/null || { echo "Podman CLI не найден. Завершите настройку Podman Desktop."; exit 1; }
	@$(PODMAN) info >/dev/null || { echo "Podman machine не запущена. Запустите её в Podman Desktop."; exit 1; }
	@$(COMPOSE) version >/dev/null || { echo "Compose не настроен. В Podman Desktop откройте Settings > Resources > Compose > Setup."; exit 1; }
	@$(COMPOSE) config --quiet

up: check
	$(COMPOSE) up --detach --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs --follow api ml-worker

worker-logs:
	$(COMPOSE) logs --follow ml-worker

ps:
	$(COMPOSE) ps

migrate:
	$(COMPOSE) run --rm migrate

test:
	$(TEST_COMPOSE) run --rm --build test-migrate
	$(TEST_COMPOSE) run --rm --no-deps --build test pytest

lint:
	$(COMPOSE) --profile tools run --rm --no-deps --build test ruff check .

format:
	$(COMPOSE) --profile tools run --rm --no-deps --build test ruff format .

typecheck:
	$(COMPOSE) --profile tools run --rm --no-deps --build test mypy app

eval-retrieval:
	uv run cliptrace-eval --dataset $(EVAL_DATASET) --model-cache-dir data/models $(EVAL_CUTOFFS)
