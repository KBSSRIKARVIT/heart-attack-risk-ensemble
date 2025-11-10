# 🐳 Running Optimization with Docker

Yes! You can absolutely use Docker to run the optimization code. This is actually **recommended** because:

✅ **Isolated environment** - No conflicts with your system Python  
✅ **Reproducible** - Same results every time  
✅ **Easy cleanup** - Just remove the container when done  
✅ **Resource control** - Limit CPU/memory usage  

## Quick Start (3 Commands)

```bash
# 1. Make script executable (one time)
chmod +x run_optimization_docker.sh

# 2. Run optimization
./run_optimization_docker.sh

# 3. That's it! Results are saved to content/models/
```

## What Gets Created

The Docker setup includes:

1. **`Dockerfile.optimization`** - Docker image definition
2. **`docker-compose.optimization.yml`** - Easy container management
3. **`run_optimization_docker.sh`** - One-command runner script
4. **`DOCKER_OPTIMIZATION.md`** - Detailed documentation

## Simple Usage Examples

### Run Full Optimization
```bash
./run_optimization_docker.sh
```
Takes ~1-2 hours, 100 trials per model

### Faster Run (50 trials)
```bash
./run_optimization_docker.sh --trials 50
```
Takes ~30-60 minutes

### Run Feature Analysis
```bash
./run_optimization_docker.sh --script feature_importance_analysis.py
```
Takes ~5-10 minutes

### Compare Results
```bash
./run_optimization_docker.sh --script compare_models.py
```

## Using Docker Compose

If you prefer docker-compose:

```bash
# Build and run
docker-compose -f docker-compose.optimization.yml up --build

# View logs
docker-compose -f docker-compose.optimization.yml logs -f

# Stop when done
docker-compose -f docker-compose.optimization.yml down
```

## Using Docker Directly

```bash
# Build image
docker build -f Dockerfile.optimization -t heart-optimization .

# Run optimization
docker run --rm \
  -v "$(pwd)/content:/app/content" \
  -v "$(pwd)/model_assets:/app/model_assets:ro" \
  heart-optimization
```

## Results Location

All results are automatically saved to your host machine:
- `content/models/model_metrics_optimized.csv` - Performance metrics
- `content/models/*_optimized.joblib` - Optimized models
- `content/models/ensemble_info_optimized.json` - Ensemble configuration
- `content/reports/` - Feature importance visualizations

## Resource Requirements

**Minimum:**
- 4GB RAM
- 2 CPU cores
- 5GB disk space

**Recommended:**
- 8GB RAM
- 4 CPU cores
- 10GB disk space

## Time Estimates

| Configuration | Time |
|--------------|------|
| 30 trials | ~20-30 min |
| 50 trials | ~30-60 min |
| 100 trials | ~1-2 hours |
| 200 trials | ~2-4 hours |

## Troubleshooting

### Docker not running
```bash
# Check Docker status
docker info

# Start Docker Desktop (if on Mac/Windows)
# Or: sudo systemctl start docker (Linux)
```

### Out of memory
```bash
# Reduce trials
./run_optimization_docker.sh --trials 30

# Or reduce timeout
STUDY_TIMEOUT=1800 ./run_optimization_docker.sh
```

### Data file not found
```bash
# Verify data exists
ls -lh content/cardio_train_extended.csv
```

## Advanced Options

### Custom Resource Limits
Edit `docker-compose.optimization.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '8'      # Use more CPUs
      memory: 16G    # More RAM
```

### Environment Variables
```bash
N_TRIALS=50 STUDY_TIMEOUT=1800 ./run_optimization_docker.sh
```

### Interactive Shell
```bash
docker-compose -f docker-compose.optimization.yml run --rm optimization bash
```

## Next Steps

1. ✅ Run `./run_optimization_docker.sh`
2. ✅ Wait for completion (1-2 hours)
3. ✅ Check results in `content/models/`
4. ✅ Compare with baseline using `compare_models.py`
5. ✅ Deploy optimized models

## Full Documentation

For detailed instructions, see:
- **[DOCKER_OPTIMIZATION.md](DOCKER_OPTIMIZATION.md)** - Complete Docker guide
- **[QUICK_START.md](QUICK_START.md)** - General quick start
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Improvement details

---

**Pro Tip:** Run optimization overnight or during lunch break. The container will save all results automatically!

