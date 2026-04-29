import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import darcy_inverse_problem as d  # noqa: E402

exp = sys.argv[1]
setting = json.loads(sys.argv[2])
trial = int(sys.argv[3])
seed = int(sys.argv[4])

cache = d.build_truth_cache()
result = d.run_one_trial(cache, **setting, seed=seed)
row = {'experiment': exp, 'trial': trial, **setting, **result}
print(json.dumps(row), flush=True)
os._exit(0)
