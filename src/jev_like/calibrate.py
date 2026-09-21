from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


def scaled_row(probabilities: list[float], temperature: float) -> np.ndarray:
    values = np.power(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0), 1.0 / temperature)
    return values / values.sum()


def nll(rows: list[dict], temperature: float) -> float:
    losses = [-np.log(np.clip(scaled_row(row["probabilities"], temperature)[row["target"]], 1e-12, 1.0)) for row in rows]
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))["rows"]
    result = minimize_scalar(lambda log_t: nll(rows, float(np.exp(log_t))), bounds=(-4.0, 4.0), method="bounded")
    temperature = float(np.exp(result.x))
    output = {
        "temperature": temperature,
        "samples": len(rows),
        "nll_before": nll(rows, 1.0),
        "nll_after": nll(rows, temperature),
        "optimizer_success": bool(result.success),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
