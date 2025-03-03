# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a Random Forest classifier implemented using the Scikit-learn library. It uses default hyperparameters and is designed to predict whether an individual's income exceeds $50,000 per year based on census data.

## Intended Use

The model is intended to predict whether an individual's income exceeds $50,000 annually based on demographic and socioeconomic features from the 1994 Census dataset. It is suitable for research and educational purposes .

## Training Data

The training data is sourced from [the UCI Machine Learning Repository - Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income). The dataset contains demographic and employment-related attributes from the 1994 Census database. After preprocessing, the dataset consists of 30,162 rows and 15 attributes, including age, workclass, education, marital status, occupation, race, sex, and native country.

## Evaluation Data

The dataset was split into training and test sets using an 80-20 split. The test set contains 6,032 rows and is used to evaluate the model's performance.

## Metrics

The model was evaluated on the test set using the following metrics:

Precision: 0.73

Recall: 0.65

Fbeta Score: 0.69

These metrics indicate moderate performance, with room for improvement, particularly in recall.

## Ethical Considerations

The dataset includes sensitive attributes such as race, sex, and native country, which may introduce bias. The model's performance should be carefully evaluated across different demographic groups to ensure fairness.

## Caveats and Recommendations

The model is trained on data from 1994, which may not reflect current socioeconomic conditions. Default hyperparameters were used, which may not be optimal for this task. To improve performance, consider hyperparameter tuning, feature engineering, and incorporating more recent and diverse data. Regularly evaluate the model for bias and fairness across demographic groups.
