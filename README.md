```bash
pip install torch
pip install -U git+https://github.com/NVIDIA/TransformerEngine.git@stable
pip install -U https://github.com/Lightning-AI/lightning/archive/refs/heads/carmocca/transformer-engine.zip
pip uninstall -y pydantic; pip install pydantic  # https://github.com/Lightning-AI/lightning/issues/17106

python pytorch.py
python fabric.py
python trainer.py
```
