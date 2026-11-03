# Digit Recognition Dashboard
```
pip install -r requirements.txt
streamlit run app.py
```
First launch trains all 9 models and stores real test-set metrics in `data/dashboard.db` (SQLite);
models are saved in `models/`. Predictions and training runs persist across restarts.

## Plugging in your existing project
Only `backend.py` touches ML code: edit `load_dataset()` (must return X scaled 0-1, y, images)
and the `CLASSIFIERS` / `REGRESSORS` factories (any object with fit/predict, + predict_proba for classifiers).
If your project has a Keras/PyTorch CNN, wrap it like `KerasCNN` (fit, predict, predict_proba, classes_).
Delete `data/` and `models/` to reset history.
