## Source code change:
1. Separate the preprocessing pipeline from source code, run another service.

## Detection Strategy

1. ** Continuous Monitoring**  

    * Performance Metrics: 

        Rolling window (e.g. last 1 000 labelled samples)
        Collect the model's prediction, compare with the label(made by human), calculated the metrics.

        Set a threshold at 5%, if drop > 5 % from training score.

        Question: In real industry, how to detect this? In db? How to solve the problem that we don't have label?
        In real industry, do we have label?

    * Data Drift Metrics

        Make embeddings(BERT remove classifier layer or TF-IDF vectorizer) of the text. Then compare cosine similarity by a single dot product.








