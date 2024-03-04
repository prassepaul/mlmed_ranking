# mlmed_ranking
Code for the paper "Matching Anticancer Compounds and Tumor Cell Lines by Neural Networks with Ranking Loss"

Drug sensitivity models are used to extract generalizable genomic or transcriptomic
patterns from cell line-drug susceptibility experiment data. An important goal is to select
the drug candidates that are most likely to work in a patient. In view of the fact that
doctors will only consider a limited number of treatment options, a model does not
have to be able to accurately predict inhibitory concentrations of ineffective drugs.
Ideally, an optimization criterion for a predictive drug sensitivity model should not
compromise the model's capability to identify the most effective drugs against its
ability to estimate the inhibitory concentrations of ineffective drugs.
Therefore, we treat drug sensitivity modeling as a Learning-to-Rank problem.
Here we present an extension of the drug sensitivity regression model PaccMann (https://github.com/PaccMann),
to perform rank learning using tensorflow-ranking (https://github.com/tensorflow/ranking).
We heuristically optimize ranking metric NDCG to emphasize a focus on highly promising compounds.

## HowTo
* To run all the experiments from the paper simply run 'run_experiments.sh'
    * This runs the PaccMann and NN baseline models on the data
* To plot the results after running 'run_experiments.sh' look into the notebook 'plot_figures.ipynb'

## Requirements
* See requirements.txt
    * using conda you can create the environment using the *.yml file: conda env create -f environment.yml
