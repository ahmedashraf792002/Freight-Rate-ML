# Freight Rate Prediction – Solution

## Run
```bash
python -m pip install -r requirements.txt
# data/ already contains train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv
python src/train.py --data-dir data --out-dir outputs
python src/score.py --predictions outputs/validation_predictions.csv \
                --december-predictions outputs/december_chart_inputs.csv \
                --output-dir outputs/scorer_results
```
Outputs: `outputs/validation_predictions.csv`, `outputs/december_chart_inputs.csv` (filled), `outputs/cv_results.csv`,
`outputs/scorer_results/candidate_december.png`.

## Approach (short)
- **Split:** `validation.csv` is Nov–Dec 2025, strictly after `train_test.csv` (Jan–Oct 2025), so I validate out-of-time with
  expanding-window folds (train < Jul → test Jul–Aug; train < Sep → test Sep; train < Oct → test Oct). Random K-fold would be optimistic.
- **Target:** log(rate per mile); prediction = exp(ŷ) × distance. **Model:** LightGBM, L1 loss (robust to ~1.5% heavy label outliers), 3-seed average.
- **Cleaning:** negative weights (0.6%) → abs; missing weight (0.6%) → NaN + flag; 70-mile distance floor on very short lanes kept as-is;
  missing market_index (0.8%) irrelevant because the feature is dropped.
- **Dropped features:** `market_index` (adds −2 to −6% bias out of time, MAE 100 → ~145) and `quote_signal` (correlation with rate flips sign by month).
- **Result (mean over 3 out-of-time folds):** MAE $100, MAPE 4.35%, MedAPE 1.66% vs. $227 / 10.4% / 6.8% for an equipment-median $/mile baseline.
- **December chart:** inputs carry no market/quote values, and the model has no seasonal/date-trend feature, so only day-of-week varies (~±1%).
