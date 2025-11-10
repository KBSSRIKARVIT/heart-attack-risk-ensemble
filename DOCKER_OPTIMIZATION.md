# Running Model Optimization with Docker

This guide shows you how to run the model optimization scripts using Docker.

## Prerequisites

- Docker installed and running
- Docker Compose (usually comes with Docker Desktop)
- At least 8GB RAM available for Docker
- Data file: `content/cardio_train_extended.csv`

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and run optimization
docker-compose -f docker-compose.optimization.yml up --build

# Run in detached mode (background)
docker-compose -f docker-compose.optimization.yml up -d --build

# View logs
docker-compose -f docker-compose.optimization.yml logs -f

# Stop when done
docker-compose -f docker-compose.optimization.yml down
```

### Option 2: Using Docker Directly

```bash
# Build the image
docker build -f Dockerfile.optimization -t heart-optimization .

# Run optimization
docker run --rm \
  -v "$(pwd)/content:/app/content" \
  -v "$(pwd)/model_assets:/app/model_assets:ro" \
  --name heart-optimization \
  heart-optimization

# Run with resource limits
docker run --rm \
  -v "$(pwd)/content:/app/content" \
  -v "$(pwd)/model_assets:/app/model_assets:ro" \
  --cpus="4" \
  --memory="8g" \
  --name heart-optimization \
  heart-optimization
```

## Running Specific Scripts

### Run Model Optimization Only

```bash
docker-compose -f docker-compose.optimization.yml run --rm optimization python improve_models.py
```

### Run Feature Analysis Only

```bash
docker-compose -f docker-compose.optimization.yml run --rm optimization python feature_importance_analysis.py
```

### Run Comparison

```bash
docker-compose -f docker-compose.optimization.yml run --rm optimization python compare_models.py
```

## Customization

### Adjust Resource Limits

Edit `docker-compose.optimization.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '8'      # Use more CPUs if available
      memory: 16G    # More RAM for faster processing
```

### Reduce Optimization Time

Edit `improve_models.py` before building:

```python
n_trials = 50  # Reduce from 100 to 50 for faster results
```

Or override at runtime:

```bash
docker run --rm \
  -v "$(pwd)/content:/app/content" \
  -v "$(pwd)/improve_models.py:/app/improve_models.py" \
  heart-optimization python -c "
import sys
sys.path.insert(0, '/app')
# Modify n_trials here or use environment variable
exec(open('/app/improve_models.py').read().replace('n_trials = 100', 'n_trials = 50'))
"
```

### Use Environment Variables

Create a `.env` file:

```env
N_TRIALS=50
STUDY_TIMEOUT=1800
```

Then use it:

```bash
docker-compose -f docker-compose.optimization.yml --env-file .env up
```

## Monitoring Progress

### View Real-time Logs

```bash
# Using docker-compose
docker-compose -f docker-compose.optimization.yml logs -f

# Using docker
docker logs -f heart-optimization
```

### Check Container Status

```bash
docker ps
docker stats heart-optimization
```

## Results Location

All results are saved to your host machine in:
- `content/models/` - Optimized models and metrics
- `content/reports/` - Feature importance visualizations

These persist after the container stops.

## Troubleshooting

### Out of Memory

**Error:** `Killed` or memory errors

**Solution:**
1. Reduce `n_trials` in `improve_models.py`
2. Reduce memory limit in docker-compose.yml
3. Close other applications

### Build Fails

**Error:** Package installation fails

**Solution:**
```bash
# Clean build
docker-compose -f docker-compose.optimization.yml build --no-cache
```

### Data Not Found

**Error:** `Data file not found`

**Solution:**
```bash
# Verify data file exists
ls -lh content/cardio_train_extended.csv

# Check volume mount
docker-compose -f docker-compose.optimization.yml config
```

### Slow Performance

**Solutions:**
1. Increase CPU allocation in docker-compose.yml
2. Use fewer trials: `n_trials = 30`
3. Run on a machine with more resources

## Advanced Usage

### Interactive Shell

```bash
# Get a shell in the container
docker-compose -f docker-compose.optimization.yml run --rm optimization bash

# Then run scripts manually
python improve_models.py
```

### Run Multiple Optimizations

```bash
# Run optimization with different trial counts
for trials in 30 50 100; do
  docker run --rm \
    -v "$(pwd)/content:/app/content" \
    -e N_TRIALS=$trials \
    heart-optimization \
    python -c "import sys; sys.path.insert(0, '/app'); exec(open('/app/improve_models.py').read().replace('n_trials = 100', f'n_trials = {trials}'))"
done
```

### Save Container State

```bash
# Commit container to image
docker commit heart-optimization heart-optimization:snapshot

# Use later
docker run --rm -v "$(pwd)/content:/app/content" heart-optimization:snapshot
```

## Performance Tips

1. **Use SSD storage** - Faster I/O for data loading
2. **Allocate more CPUs** - Parallel processing in Optuna
3. **Increase memory** - Better for large datasets
4. **Run overnight** - Let it run while you sleep
5. **Use GPU** (if available) - Requires NVIDIA Docker runtime

## GPU Support (Optional)

If you have an NVIDIA GPU:

```yaml
# Add to docker-compose.optimization.yml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

Then build with:
```bash
docker build -f Dockerfile.optimization -t heart-optimization .
```

## Example Workflow

```bash
# 1. Build image
docker-compose -f docker-compose.optimization.yml build

# 2. Run optimization (takes 1-2 hours)
docker-compose -f docker-compose.optimization.yml up

# 3. In another terminal, check progress
docker-compose -f docker-compose.optimization.yml logs -f

# 4. When done, run feature analysis
docker-compose -f docker-compose.optimization.yml run --rm optimization \
  python feature_importance_analysis.py

# 5. Compare results
docker-compose -f docker-compose.optimization.yml run --rm optimization \
  python compare_models.py

# 6. Clean up
docker-compose -f docker-compose.optimization.yml down
```

## Benefits of Using Docker

✅ **Isolation** - No conflicts with your system Python  
✅ **Reproducibility** - Same environment every time  
✅ **Resource Control** - Limit CPU/memory usage  
✅ **Easy Cleanup** - Remove container when done  
✅ **Portability** - Run on any machine with Docker  

## Next Steps

After optimization completes:
1. Check results in `content/models/model_metrics_optimized.csv`
2. Review feature importance in `content/reports/`
3. Compare with baseline using `compare_models.py`
4. Deploy optimized models to your Streamlit app

---

**Note:** The optimization process can take 1-2 hours. Make sure your laptop is plugged in and won't go to sleep!

