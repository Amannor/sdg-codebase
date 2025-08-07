# SDG-Codebase: Sustainable Development Goals Classification for Startups

[![GitHub repo size](https://img.shields.io/github/repo-size/Amannor/sdg-codebase)](https://github.com/Amannor/sdg-codebase)
[![License](https://img.shields.io/github/license/Amannor/sdg-codebase)](LICENSE)
[![Issues](https://img.shields.io/github/issues/Amannor/sdg-codebase)](https://github.com/Amannor/sdg-codebase/issues)
[![Last Commit](https://img.shields.io/github/last-commit/Amannor/sdg-codebase)](https://github.com/Amannor/sdg-codebase/commits/master)

---

## Overview

This repository contains code, data, and articles related to the classification of startups according to the United Nations’ 17 Sustainable Development Goals (SDGs). The work extends and complements the original research by Kfir Bar, exploring techniques for automatic SDG labeling using natural language processing on startup descriptions and social media data (Twitter).

Key components include:

- Transformer-based models (e.g., BERT, Llama) fine-tuned for SDG classification tasks.
- Datasets of startup descriptions and tweets aggregated to infer SDG alignment.
- Experiments and evaluations reported on multi-class and multi-label classification settings.
- Supplementary articles detailing methodology, experimental results, and technical insights.

This repository aims to facilitate reproducible research and encourage further advances in SDG classification from text data in startups.

---

## Contents

- `articles/`: Academic papers and technical reports describing methods, experiments, and results.
- `data/`: Preprocessed datasets for training and evaluation (startup descriptions, tweets).
- `tweets_based_classification/`: Code and notebooks for performing SDG classification using Twitter data, including data preprocessing, model fine-tuning, and evaluation.
- Jupyter Notebooks demonstrating step-by-step processing pipelines and experimental setups.

---

## Getting Started

### Prerequisites

Ensure you have Python (>=3.8) installed. Key packages used include:

- `transformers`
- `datasets`
- `torch`
- `scikit-learn`
- `pandas`
- `numpy`

You can install requirements via:
pip install -r requirements.txt


(If `requirements.txt` is not present, install packages manually or request it.)

### Running the Classification Pipeline

The main pipeline for tweet-based SDG classification is located in the `tweets_based_classification/` folder. 

- The primary Jupyter notebook is [`fine-tune-on-tweets.ipynb`](https://github.com/Amannor/sdg-codebase/blob/master/tweets_based_classification/fine-tune-on-tweets.ipynb), which contains detailed code for:

  - Data loading and preprocessing (cleaning tweets, aggregating by company)
  - Tokenizer and model setup for Llama 3 fine-tuning
  - Training loops and evaluation metrics calculation

To reproduce experiments:

1. Clone this repository.
2. Prepare dataset folders as indicated inside the notebook.
3. Open the notebook and run cells sequentially.

### Using Pretrained Models

Pretrained and fine-tuned model weights (if available) are indicated in the notebook or can be requested from the authors.

---

## Articles and Documentation

Extensive documentation explaining the methodology, experiment details, and results can be found in the `articles/` folder:

- **IJCAI 2022 paper by Kfir Bar:** The foundational article on SDG classification with language models.
- **Extension articles by Alon Mannor:** Exploring tweet-based classification with Llama models, data aggregation strategies, and comparisons.

The articles provide context, review related work, and explain design choices.

---

## Contributing

Contributions are welcome! Feel free to:

- Open issues for bugs or questions.
- Submit pull requests for improvements to code, documentation, or data handling.
- Share additional datasets or experimental results relevant to SDG classification.

Please ensure contributions adhere to repository guidelines and code style.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, clarifications, or collaborations, please open an issue or contact the repository maintainer via GitHub.

---

## References

The methodology and experiments build on the following key works:

- Bar, Kfir (2022). Using Language Models for Classifying Startups Into the UN’s 17 Sustainable Development Goals. [GitHub PDF](https://github.com/Amannor/sdg-codebase/blob/master/articles/IJCAI_2022_SDGs_Methodology.pdf)
- Mannor, Alon et al. (2025). Extending SDG classification using tweet data and Llama models. [GitHub Notebook](https://github.com/Amannor/sdg-codebase/blob/master/tweets_based_classification/fine-tune-on-tweets.ipynb)
- Additional publications in `articles/` folder.

---

Thank you for your interest in promoting sustainable development through advanced NLP!

---
