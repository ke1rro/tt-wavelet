
# Setup enviroment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

# Run pre-commit hooks manually

```bash
pre-commit run --all-files
```
