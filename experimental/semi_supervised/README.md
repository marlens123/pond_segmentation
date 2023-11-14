# Semi-supervised Segmentation: Gradually improve model performance

Idea:
1) Use existing labeled data to train model
2) Select high-performance predictions
3) Add to training data
4) Retrain model

In specific:
1) Already done
2) Question: How to automatically evaluate which predictions are good? Idea: Train two different models, specify a measure of similarity under the assumption that similar predictions are more likely to be good, and add similar predictions to training data
    - which models? (optimally 2 good performing but with differences)
    - measure of similarity: per-class pixel similarity? more focus on specific classes? assume that some melt pond instance must be present to avoid incorporating totally failed predictions?
    --> manually inspect results to get idea
