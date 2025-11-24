## Agentic POC – Claims Orchestration

This repo contains an agentic prototype for P&C claims, including:
- `fnol_agent.py`: agent that creates FNOL records from raw email or text.
- `claim_agent.py`: agent that sets up a claim from FNOL / policy data.
- Orchestration notebooks that tie the agents together.

### Primary Notebooks
- `4. orchestrator-agentic.ipynb`
- `4. orchestrator.ipynb`

### Requirements

- Python 3.13+
- `uv` for dependency and environment management
- OpenAI account and API key (for `ChatOpenAI`)
- Jupyter Notebook (already included as a dependency)


# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate
uv sync

Create a `.env` file in the project root with at least:
Add your envs following env.example