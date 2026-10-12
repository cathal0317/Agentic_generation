# Agentic generation

Conversation generation for emotional speech dataset.

This first patch sets up the Python package, local PostgreSQL, configuration, and
migration history. Its first migration enables pgvector. Conversation generation
and sample tables are the next step.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker with the Compose v2 plugin (Docker Desktop includes it)

PostgreSQL 17 runs in Docker. Python commands run on the host. The database is
available only on `127.0.0.1:5433`; the non-default port avoids an existing local
PostgreSQL instance on 5432.

## First run

Run these commands from the repository root after copying the starter files into
your `Agentic_generation` checkout:

```bash
cp .env.example .env
uv sync --locked
docker compose up -d --wait db
uv run alembic upgrade head
uv run agentic-generation check-db
```

The last command prints the database name, PostgreSQL and pgvector versions,
the current migration (`0002_add_conversations`), and `"vector_check": "ok"`.
It tests a cosine-distance operation without creating a permanent vector table.

## Scenario validation and prompt preview

Validate a scenario file and preview its prompts without database configuration
or external services:

```bash
uv run agentic-generation validate-scenario examples/user_healthcheck.json
uv run agentic-generation preview-prompts examples/user_healthcheck.json
```

`uv.lock` records the resolved Python dependencies. `.env` contains local
configuration and is ignored by Git. The example password is for local
development only. Both Compose and Python read the same `POSTGRES_*` settings,
so there is no second connection string to keep in sync. Exported environment
variables override `.env` values.

The database image includes pgvector's server files. The Alembic migration
activates the extension in this particular database. Running `upgrade head`
again should leave the database unchanged.

## Useful commands

```bash
uv run alembic current
uv run alembic history
uv run ruff check .
uv run ruff format --check .
uv run pytest
docker compose logs db
docker compose down
```

`docker compose down` stops the service and keeps the named database volume.
PostgreSQL uses `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` to
initialise a new volume; changing those variables later does not change the
credentials or database already stored in that volume.

## Database integration test

The default test run checks settings and generated migration SQL. The integration
test is skipped until `TEST_DATABASE_URL` points to a disposable database.
To create one alongside the local development database:

```bash
docker compose exec db sh -c 'createdb -U "$POSTGRES_USER" agentic_generation_test'
```

Then, with the unmodified example credentials:

```bash
TEST_DATABASE_URL='postgresql+psycopg://agentic:agentic-local-dev@127.0.0.1:5433/agentic_generation_test' uv run pytest -m integration
```

If the role, password, or port was changed in `.env`, adjust this test URL too.
The test applies the migrations twice and checks a real vector query using a
temporary table. It is not a mock of PostgreSQL. The test database stays available
for subsequent runs; only run `createdb` when it does not already exist.

## Adding the next migration

Add SQLAlchemy models based on `agentic_generation.db.Base`, and import their
module in `migrations/env.py` before `target_metadata` is used. Then run:

```bash
uv run alembic revision --autogenerate -m "add conversation tables"
```

Review the generated SQL changes before applying them with `upgrade head`.
Use migrations to create and change tables; do not add `Base.metadata.create_all()`
to application startup. The next patch can introduce conversations, turns, and
samples after their contracts are settled. Embedding dimensions are not fixed
by this initial setup.

Rolling back the first migration leaves the vector extension installed. It may
already have existed or be used by other tables, so this migration does not
claim ownership of it.

## Scope and references

The runtime dependencies are SQLAlchemy, psycopg, Alembic, and Pydantic Settings.
The Python pgvector adapter can be added with the first vector models; server-side
extension activation does not require it. No provider API key is needed for this
setup.

- [pgvector installation and Docker images](https://github.com/pgvector/pgvector)
- [Alembic tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

See `VALIDATION.md` for the checks actually run on this starter.
