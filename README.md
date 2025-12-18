# Using Embeddings' Geometric Similarity for In-Context Learning

**CS 685 Final Project - Fall 2025**

**Authors:** Om Mehta, Tanush Savadi

University of Massachusetts Amherst

## Abstract

We investigate whether prompt effectiveness is partly explained by the geometry of token embeddings. Specifically, we evaluate the correlation between geometric metrics—local intrinsic dimension (ID) and Local Linear Embedding (LLE) reconstruction error—and model performance on the GSM8K mathematical reasoning benchmark.

Using two models (TinyLlama-1.1B and Gemma-3-1b-it) across 11 prompt templates, we find that TinyLlama shows a moderate negative correlation between intrinsic dimension and accuracy (Spearman ρ = -0.54, p = 0.085), suggesting prompts with lower mean intrinsic dimension tend to achieve higher accuracy. However, Gemma-3-1b-it shows no significant correlation, indicating the geometry-performance relationship may be model-dependent.

## Key Findings

| Model | Best Template | Accuracy | ID-Accuracy Correlation |
|-------|--------------|----------|------------------------|
| TinyLlama-1.1B | no_cot | 4.3% | ρ = -0.54 (p = 0.085) |
| Gemma-3-1b-it | structured_steps | 8.0% | ρ = 0.10 (p = 0.77) |

## Repository Structure

```
.
├── CS685_Final_Project_Colab_Baseline.ipynb    # TinyLlama experiments (Colab notebook)
├── CS685_Final_Project_Colab_googleGemma_final.ipynb  # Gemma experiments (Colab notebook)
├── cs685_f25_final_report/                      # LaTeX report
│   ├── main.tex                                 # Main report file
│   ├── yourbib.bib                              # Bibliography
│   └── figs/                                    # Figures
├── results_baseline/                            # TinyLlama results
│   ├── results_summary.csv
│   ├── correlations.json
│   └── *.png                                    # Plots
├── results_final/                               # Gemma results
│   ├── results_summary.csv
│   ├── correlations.json
│   └── *.png                                    # Plots
└── README.md
```

## Running the Experiments

### Requirements

```bash
pip install transformers datasets faiss-cpu scipy tqdm matplotlib pandas numpy
```

For GPU support on Google Colab:
- Use T4 or A100 runtime
- For Gemma: requires HuggingFace login (`huggingface-cli login`)

### Quick Start

1. **Open in Google Colab:**
   - Upload the `.ipynb` notebook files to Google Colab
   - Select GPU runtime (Runtime → Change runtime type → T4 GPU)
   - Run all cells

2. **For TinyLlama (no login required):**
   ```python
   CONFIG = {
       "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
       "gsm8k_samples": 300,
       "max_new_tokens": 256,
   }
   ```

3. **For Gemma (requires HuggingFace login):**
   ```python
   from huggingface_hub import login
   login()
   
   CONFIG = {
       "model": "google/gemma-3-1b-it",
       "gsm8k_samples": 150,
       "max_new_tokens": 128,
   }
   ```

## Methodology

### Geometry Metrics

1. **Local Intrinsic Dimension (ID):** Number of PCA components needed to explain 95% of variance in a token's k=20 nearest neighborhood.

2. **LLE Reconstruction Error:** How well a token can be linearly reconstructed from its neighbors.

### Prompt Templates Tested

| Type | Example |
|------|---------|
| Minimal | "Q: {q}\nA:" |
| Direct | "Answer the question: {q}\n..." |
| Chain-of-Thought | "Explain step by step: {q}\n..." |
| Structured | "You are a math tutor..." |
| No-CoT | "Do NOT show work..." |

## Results

### Correlation Analysis

**TinyLlama-1.1B:**
- Spearman (ID vs Accuracy): ρ = -0.54, p = 0.085 ✓
- Lower ID → Higher Accuracy (supports hypothesis)

**Gemma-3-1b-it:**
- Spearman (ID vs Accuracy): ρ = 0.10, p = 0.77 ✗
- No significant correlation

### Interpretation

The geometry-accuracy relationship appears to be **model-dependent**. TinyLlama shows promising evidence that prompts using tokens in more organized embedding neighborhoods achieve better performance.

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{mehta2025geometry,
  title={Using Embeddings' Geometric Similarity for In-Context Learning},
  author={Mehta, Om and Savadi, Tanush},
  year={2025},
  institution={University of Massachusetts Amherst},
  note={CS 685 Final Project}
}
```

## References

- [Lee et al. (2025)](https://arxiv.org/abs/2501.00000) - Shared Global and Local Geometry of Language Model Embeddings
- [Huh et al. (2024)](https://arxiv.org/abs/2405.07987) - The Platonic Representation Hypothesis
- [GSM8K Dataset](https://github.com/openai/grade-school-math)

## License

MIT License - See LICENSE file for details.