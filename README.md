# Document Similarity using Spark
This project aims to compute the similarity between documents using Spark. It compares a sequential and a parallel implementation of the cosine similarity algorithm. Please take a look at the report for further details.

# Project structure
The project is structured as follows:
- `data/`: contains the dataset used,
- `results/`: contains the results of the experiments,
- `test.py`: script to test the model and compute specific metrics,
- `documents_similarity.ipynb`: notebook with the code step by step to compute the similarity between documents,
- `results_analysis.ipynb`: notebook with the code step by step to analyze the results of the experiments.

# How to run the code
To run the code, you need to have the following libraries installed:
- `pandas`,
- `numpy`,
- `pyspark`,
- `matplotlib`.

To run the code, you can follow the steps below:
1. Run the `documents_similarity.ipynb` notebook to compute the similarity between documents on specific configurations,
2. Run the `test.py` script to test the model and compute specific metrics that will be stored in the `results/` folder,
3. Run the `results_analysis.ipynb` notebook to analyze the results of the experiments.
