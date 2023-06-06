# Energy demand prediction
Energy demand prediction for Lineage Logistics.

Running `main.py` will run the entire pipeline. You should provide the data from Lineage Logistics yourself (in `data/`), as this is private company data. Once provided, the pipeline will inspect the data using `utils/explore_dataset.py`, preprocess using `utils/preprocess.py`, and predict using the model files in `models/`. The validation set predictions of these models are appended into `output/model_predictions_val.csv`. Then the results are evaluated by `utils/evaluation.py` which generates some figures shown in `imgs/`. Finally, `utils/test_predictions.py` is called, which shows the predictions of the best model, Random Forest, superimposed on the validation set data.

The final predictions for handing in are generated into `output/y_test.csv`.
