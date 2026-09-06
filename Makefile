.PHONY: demo check train evaluate

NAV_DIR ?= artifacts/corgi-cafe/nav

demo:
	node scripts/serve_demo.mjs

check:
	python3 -m compileall -q scripts
	node --check web-demo/app.js
	node --check scripts/serve_demo.mjs
	python3 -c 'import json, pathlib; [json.load(path.open()) for path in pathlib.Path("web-demo/data").glob("*.json")]'

train:
	python3 scripts/train_nav_qlearning.py --nav-dir "$(NAV_DIR)"

evaluate:
	python3 scripts/eval_nav_policy.py --nav-dir "$(NAV_DIR)"
