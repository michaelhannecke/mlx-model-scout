.PHONY: sync lint fmt test check run run-http inspect client client-agent mlx-server image compose-up compose-down audit-ui purge-logs

sync:            ## install deps incl. dev
	uv sync

lint:            ## ruff lint
	uv run ruff check .

fmt:             ## ruff format + autofix
	uv run ruff format . && uv run ruff check --fix .

test:            ## run tests
	uv run pytest -q

check: lint test ## lint + format check + tests (what CI runs)
	uv run ruff format --check .

run:             ## MCP server over stdio
	uv run mlx-model-scout

run-http:        ## MCP server over streamable HTTP on 127.0.0.1:8765
	SCOUT_TRANSPORT=http uv run mlx-model-scout

inspect:         ## MCP Inspector UI against the stdio server (needs Node)
	npx @modelcontextprotocol/inspector uv run mlx-model-scout

client:          ## local MCP client, server mode (USE_CASE="..." to override)
	uv run python client/local_client.py "$${USE_CASE:-Python coding assistant in the IDE}"

client-agent:    ## local MCP client, agent mode (local LLM drives the workflow tools)
	uv run python client/local_client.py "$${USE_CASE:-Python coding assistant in the IDE}" --mode agent

mlx-server:      ## alternative to omlx: mlx_lm.server on the macOS host, same port (NOT in this project's env)
	uv tool run --from mlx-lm mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8000

image:           ## build OCI image with Apple container
	container build -t mlx-model-scout:latest .

compose-up:      ## start stack via community compose plugin
	container compose up

compose-down:
	container compose down

audit-ui:        ## Datasette on the audit DB with canned queries, http://127.0.0.1:8001 (read-only)
	uvx datasette@0.65.5 serve $${SCOUT_DB_PATH:-data/scout.db} --metadata config/datasette.yml \
		--host 127.0.0.1 --port 8001 --open

purge-logs:      ## delete audit rows older than SCOUT_LOG_RETENTION_DAYS
	uv run python -c "from mlx_model_scout.db.audit import purge_old_logs; print(purge_old_logs())"
