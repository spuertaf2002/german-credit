# Model training process
## File content
This file contains all steps needed for end to end ML model training, from data extraction to model training.

## Model training steps
| Step name          | File                                                                                                                                                                                                                             |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Extraction         | [data_download.ipynb](https://github.com/spuertaf2002/german-credit/blob/main/notebooks/data_download.ipynb)                                                                                                                     |
| Transformation (1) | [init_data_exploration.ipynb](https://github.com/spuertaf2002/german-credit/blob/main/notebooks/init_data_exploration.ipynb)                                                                                                     |
| Transformation (2) | [feature_engineering.ipynb](https://github.com/spuertaf2002/german-credit/blob/main/notebooks/feature_engineering.ipynb)                                                                                                         |
| Model training     | [baseline_model.ipynb](https://github.com/spuertaf2002/german-credit/blob/main/notebooks/baseline_model.ipynb), [model_selection.ipynb](https://github.com/spuertaf2002/german-credit/blob/main/notebooks/model_selection.ipynb) |

### Extraction
Extract from raw CSV source file.  

### Transformation (1)
- Check for weird `NAN` values and replace them with actual `NAN`.
- Drop `data_source` and `Checking account` columns.
- Handle categorical nominal and ordinal features.
- Handle numerical columns.    

### Transformation (2)
- Cast `Risk` column as Integer.
- Replace with `NAN` weird values found in reamining feature columns.
- Remove outliers
- Build preprocessor pipeline. 

### Model training
- Cross Validation Boxplot.
- Training vs cross validation scores (bar graph).
- Model predictions confusion matrix.
- Model learing curve.
- Model scalability plot.  

