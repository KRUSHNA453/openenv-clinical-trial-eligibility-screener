"""FastAPI server for the Clinical Trial Eligibility Screener environment."""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install dependencies with `uv sync`."
    ) from exc

from models import ClinicalTrialAction, ClinicalTrialObservation
from server.env import ClinicalTrialEligibilityScreenerEnvironment


app = create_app(
    ClinicalTrialEligibilityScreenerEnvironment,
    ClinicalTrialAction,
    ClinicalTrialObservation,
    env_name="clinical_trial_eligibility_screener",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    import sys

    if len(sys.argv) == 1:
        main()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=8000)
        args = parser.parse_args()
        main(port=args.port)
