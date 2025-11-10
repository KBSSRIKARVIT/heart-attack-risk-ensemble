# How to Monitor Training Progress

## Training is Currently Running! ✅

The model optimization is running in Docker container `heart-optimization-v2`.

## Quick Status Check

```bash
# Check if container is running
docker ps | grep heart-optimization

# See current progress (last 50 lines)
docker logs --tail 50 heart-optimization-v2

# Follow progress in real-time (like tail -f)
docker logs -f heart-optimization-v2
```

## View Log File

```bash
# View the log file
tail -f optimization_v2_log.txt

# Or view last 100 lines
tail -100 optimization_v2_log.txt
```

## Current Progress

Based on the logs, training is:
- **XGBoost:** Trial 4/300 (just started)
- **CatBoost:** Waiting (will start after XGBoost)
- **LightGBM:** Waiting (will start after CatBoost)

## Estimated Time Remaining

- **XGBoost (300 trials):** ~1.5-2 hours remaining
- **CatBoost (300 trials):** ~2-3 hours
- **LightGBM (300 trials):** ~1-1.5 hours
- **Total:** ~4.5-6.5 hours

## What to Look For

The logs show:
- Trial number (e.g., "Trial 4/300")
- Best score found so far
- Progress bar
- Estimated time remaining

## Stop Training (if needed)

```bash
docker stop heart-optimization-v2
```

## Check Results (when complete)

Results will be saved to:
- `content/models/model_metrics_optimized.csv`
- `content/models/*_optimized.joblib`
- `content/models/ensemble_info_optimized.json`

