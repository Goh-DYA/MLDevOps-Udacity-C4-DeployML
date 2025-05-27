# Model Card

## Model Details
- **Model Type**: Random Forest Classifier
- **Version**: 1.0
- **Training Framework**: scikit-learn
- **Parameters**: 
  - Random state: 42
  - Default parameters for RandomForestClassifier
- **Features**: Both numerical and categorical features are used, with categorical features being one-hot encoded

## Intended Use
- **Primary Use**: Predict whether an individual's income exceeds $50,000 per year based on census data
- **Intended Users**: Researchers and analysts studying income patterns
- **Out-of-Scope Uses**: Should not be used for individual financial decisions or discriminatory purposes

## Training Data
- **Dataset**: Census Income Dataset from UCI Machine Learning Repository
- **Size**: 80% of the original dataset (train-test split)
- **Features**: 
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
  - Numerical: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
- **Target Variable**: salary (binary: <=50K, >50K)
- **Preprocessing**: 
  - Categorical features: OneHotEncoder
  - Label: LabelBinarizer
  - Spaces stripped from all string values

## Evaluation Data
- **Dataset**: 20% of the original dataset
- **Split Method**: Random split with test_size=0.20, random_state=42
- **Preprocessing**: Same as training data, using fitted encoders from training

## Metrics
- **Metrics Used**: 
  - Precision
  - Recall
  - F1 Score
- **Slice Performance**: Performance metrics are computed for each categorical feature slice
- **Overall Performance**: Available in slice_output.txt
- **Slice Analysis**: Detailed performance metrics for different demographic slices are stored in model/slice_output.txt

## Ethical Considerations
- The model uses sensitive demographic information (race, sex, native-country) which could lead to biased predictions
- Care should be taken to regularly evaluate the model for fairness across different demographic groups
- The model should not be used as the sole decision-maker for employment or financial decisions
- Users should be aware of potential historical biases in the census data

## Caveats and Recommendations
- Model should be periodically retrained with fresh data to maintain relevance
- Performance across different demographic groups should be monitored for fairness
- Feature importance analysis recommended to understand prediction drivers
- Consider implementing feature importance analysis to understand which factors most influence predictions
- Regular bias testing across different demographic groups is recommended
- Model performance may vary significantly across different slices of the population


## Other references

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
