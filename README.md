# STWP - Short Term Weather Prediction


This repository is specifically designed for the research component of the engineering thesis titled "A Machine Learning System for Short-Term Weather Prediction." It encompasses not only the implementation of baseline models and the main architecture but also includes modules for data preprocessing, training pipelines, hyperparameter optimization, result presentation, an API facilitating communication with a [mobile application](https://github.com/JaJasiok/meteo-mind/) for the best model, and scripts for downloading data from the [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels) dataset.

### Install prerequisites:
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Config pre-commit hooks
<!-- Instruction [here](pre-commit-instruction.md). -->
```shell
pip install -r requirements.txt
pre-commit install
```

### API dockerization
Create image:
```shell
docker build -t meteo-api ./api
```
Run the container
```shell
docker run -it --rm -p 8080:8888 --name api meteo-api
```

### Feature Examination
The examined features include key meteorological parameters, each contributing to a comprehensive understanding of atmospheric conditions. The table below outlines these features along with their respective symbols, quantities, and units:

| Symbol | Quantity                             | Unit     |
| ------ | ------------------------------------ | -------- |
| t2m    | Temperature at 2m above ground       | °C       |
| sp     | Surface pressure                     | hPa      |
| tcc    | Total cloud cover                    | (0 - 1)  |
| u10    | 10m U wind component                 | $m/s$    |
| v10    | 10m V wind component                 | $m/s$    |
| tp     | Total precipitation                  | mm       |

### Results and analysis
Objective functions used for evaluating model performance include MAE and RMSE. Below are the average scores for the entire test set (year 2021) across all baselines (SLR - Simple Linear Regression, LR - Linear Regression, GB - Gradient Boosting Trees, U-NET), the graph architecture, [TIGGE](https://www.ecmwf.int/en/research/projects/tigge) - a representative of the [NWP](https://www.weather.gov/media/ajk/brochures/NumericalWeatherPrediction.pdf) paradigm, and NAIVE method - an average of train set. Each model takes 5 weather states as input and predicts one step into the future.
<table>
<tr><th>RMSE Results</th><th>MAE Results</th></tr>
<tr><td>

|        | t2m   | sp    | tcc   | u10   | v10   | tp    |
|--------|-------|-------|-------|-------|-------|-------|
| TIGGE  | **1.122** | 3.241 | **0.228** | **0.654** | **0.650** | 0.314 |
| GNN    | 1.590 | **1.176** | 0.278 | 1.120 | 1.101 | 0.305 |
| U-NET  | 1.692 | 1.323 | 0.287 | 1.309 | 1.272 | 0.305 |
| GB     | 1.797 | 1.449 | 0.286 | 1.462 | 1.451 | **0.293** |
| LR     | 2.023 | 1.355 | 0.292 | 1.502 | 1.494 | 0.296 |
| SLR    | 2.123 | 1.427 | 0.295 | 1.561 | 1.529 | 0.302 |
| NAIVE  | 9.125 | 8.286 | 0.360 | 2.909 | 2.672 | 0.312 |

</td><td>

|        | t2m   | sp    | tcc   | u10   | v10   | tp    |
|--------|-------|-------|-------|-------|-------|-------|
| TIGGE  | **0.816** | 1.710 | **0.135** | **0.472** | **0.470** | **0.081** |
| GNN    | 1.188 | **0.880** | 0.186 | 0.820 | 0.807 | **0.081** |
| U-NET  | 1.270 | 0.994 | 0.194 | 0.977 | 0.937 | 0.082 |
| GB     | 1.319 | 1.062 | 0.228 | 1.078 | 1.071 | 0.104 |
| LR     | 1.530 | 0.990 | 0.237 | 1.117 | 1.103 | 0.114 |
| SLR    | 1.584 | 1.041 | 0.242 | 1.147 | 1.132 | 0.116 |
| NAIVE  | 7.608 | 6.517 | 0.324 | 2.267 | 2.131 | 0.123 |

</td></tr> </table>

To provide a comprehensive evaluation of each solution, we introduced the $\tilde{\mathcal{L}}_{RMSE}$ metric, which represents the mean RMSE for each standardized predicted feature and its target. As part of our analysis, we investigated how the models' performance varied with the length of the input and forecast sequence:

![data_sequence_page-0001](https://github.com/kamil271e/stwp/assets/82380348/c2bed9a9-1493-48f9-9778-9c2125f169df) | ![data_fh_page-0001](https://github.com/kamil271e/stwp/assets/82380348/8b6eace3-57bf-4196-a470-745b9116197b)
:---------------------:|:---------------------:

In a significant achievement, we integrated our top-performing solution with a selected approach from the numerical weather prediction paradigm, resulting in improved predictive performance. This integration involved a simple weighted average calculation: 
```math
\alpha \in [0,1]: \mathbf{\hat{Y}} = \alpha \mathbf{\hat{Y}}_{GNN} + (1 - \alpha) \mathbf{\hat{Y}}_{TIGGE}
```
<p align="center">
  <img src="https://github.com/kamil271e/stwp/assets/82380348/6c9fbcb8-e988-4f1e-a205-ccfe22cbe4f9" alt="alpha_loss_page-0001" width="600">
</p>
 
More detailed analyses regarding each feature are included in the thesis.

### Prediction Quality Visualization
For both visualizations below, the forecasting horizon is set to 1, representing the prediction timestamp t+1, which corresponds to a projection 6 hours into the future.

The video underneath demonstrates a prediction using the graph model after fine-tuning for temperatures 2m above ground for the first 3 weeks of January 2021:

https://github.com/kamil271e/stwp/assets/82380348/73e4b688-76f8-4e89-b2ae-458219b54844

In the demonstration below, we highlight the prediction quality for all features of the graph architecture using a randomly selected training example.
![gnn_sample_pred-1](https://github.com/iwamaciek/stwp/assets/82380348/12a3a40e-4baa-4274-807d-fa742fa7d710)


