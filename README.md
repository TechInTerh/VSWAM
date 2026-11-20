# VSWAM
## TabPFN notebooks

Two notebooks: `tabpfn_stage1.ipynb` and `tabpfn_stage2.ipynb`. Their input data is in `processed/`.

### Get a Prior Labs token

1. Sign up at [priorlabs.ai](https://priorlabs.ai) and log in.
2. Copy your API token from your account page.

You don't need to save it anywhere. The notebook asks for it when you run it. Or set it first:

```bash
export TABPFN_TOKEN="your-token"
```

### Run

1. Install the packages:
   ```bash
   pip install torch tabpfn pandas scikit-learn shap dice-ml shapiq matplotlib jupyter
   ```
2. Open Jupyter from the repo root, so the `processed/` paths work:
   ```bash
   jupyter notebook
   ```
3. Open a notebook and run all cells. Paste your token when it asks.

Stage 1 reads `processed/stage1_tfm.csv`. Stage 2 reads `processed/stage2_tfm.csv`.

A GPU makes it much faster. Without one it falls back to CPU (works on a Mac, just slow).
