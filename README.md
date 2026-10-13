# agent-harness



```
 docker build -f harness/sandbox/Dockerfile -t agent-harness:01 .
```

```
source .venv/bin/activate
```


```bash
deactivate
python3 -m venv .venv-host
source .venv-host/bin/activate
pip install -r requirements.txt
echo ".venv-host/" >> .gitignore
python -m harness.agent
```