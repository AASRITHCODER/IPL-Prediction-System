Link for Help --> https://github.com/simranjeet97/IPL2023_WinningPrediction_EDA_Dashboard/tree/main/
Project Based on Interview --> 
1. https://www.instagram.com/reel/DU5jBV3k2d-/
2. https://www.instagram.com/p/DUx1P8miqmr

~ 12.68x increase in data
"Hybrid Sequential Multi-Task Cricket Prediction System"

pre-commit run --all-files

mlflow ui --backend-store-uri sqlite:///experiments/mlruns.db --port 5000

“I built a versioned ML artifact system with controlled promotion to production”

# IPL Prediction System using LSTM

Developing a web application to predict outcomes of IPL cricket matches using historical match data. This system will use the MERN stack for the web application and integrate Deep Learning (DL) models like LSTM and RNN for the prediction functionality.

## Deployment

To run this project, The user must be having node and express installed. Then need to go to the src folder and type the following commands in the terminal or powershell.

FIrst go to src folder, for which type the command

```bash
    cd src
```

Then we need to make sure that flask has an environment to run in. If it is not there, we need to run the following commands:

### In Windows

```bash
    $env:FLASK_APP = "model.py"
```
### In Mac

```bash
    export FLASK_APP="model.py"
```

### Then run

```bash
    flask run
```

In another ternimal run

```bash
    npm start
```
## Team Members

- [@Krishna Chaitanya](https://github.com/Krishna752006)
- [@Guru Charan](https://github.com/gcrn2318)
- [@Lokesh](https://github.com/LokeshVasireddy)

### 3.5 Known Observations & Potential Improvements

- **No CLI arguments** — hyperparameters require direct file edits. Consider `argparse` or a config YAML.
- **Hardcoded data path** — `./data/data1.csv` assumes script is run from `ml-service/`. Will break if CWD differs.
- **No early stopping** — fixed epoch count only; no `EarlyStopping` or learning rate scheduling.
- **Hardcoded validation thresholds** — `MAE < 0.35` and `R2 > 0.70` are inline constants; externalising them would help CI/CD pipelines.
- **Silent exception handling** — the entire training block is wrapped in a broad `try/except` that only prints the error. Failed MLflow runs are not explicitly marked as `FAILED`.
- **mlruns cleanup workaround** — the stray folder merge is a symptom of an MLflow tracking URI config issue; worth fixing at the source.