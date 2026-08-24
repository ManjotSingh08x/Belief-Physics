## Installation and Setup
We will be using uv as our package manager. To sync and setup dependancies, ensure `uv` is installed
To build and work do :- 

```bash
uv sync
```
To run scripts and notebooks, use 

```bash
uv run python train.py
uv run jupyter lab
```

To add or remove dependancies, use format 

```bash
uv add torch 
uv remove torch
```
Also, always commit pyproject.toml and uv.lock together in the same PR.