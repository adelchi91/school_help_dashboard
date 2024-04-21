import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import numpy as np

# PARAMETER 
NUMBER_OF_FEATURES_WE_CONSIDER = 15 # We decide to focus on the first 15 most important features
                                    # which will be displayed on the dashboard

def load_data():
    return pd.read_csv("./data/exercice_data.csv", sep=None, encoding='latin1')

def preprocess_data(df):
    replace_dict = {"yes": 1, "no": 0}
    df.replace(replace_dict, inplace=True)
    return df

def define_transformer_and_pipeline(df):
    columns_to_encode = ['sex', 'address', 'famsize', 'Pstatus', 'reason', 'guardian', 'Mjob', 'Fjob']
    encoder = OneHotEncoder(drop='first')
    preprocessor = ColumnTransformer(
        transformers=[('onehot', encoder, columns_to_encode)],
        remainder='passthrough'
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(
            random_state=42,
            n_estimators=100,  
            max_depth=2,  
            learning_rate=0.1,  
            min_child_weight=1,  
            subsample=0.8,  
            colsample_bytree=0.8,  
            reg_alpha=0,  
            reg_lambda=1,  
            gamma=0,  
        ))
    ])
    X = df.drop(columns=['FinalGrade', 'StudentID', 'FirstName', 'FamilyName'])
    y = df['FinalGrade'].values
    return pipeline, X, y

def train_model(pipeline, X_train, y_train, X_test):
    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_train), pipeline.predict(X_test)

def compute_metrics(y_train, y_train_pred, y_test, y_test_pred):
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)

    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    metrics_df = pd.DataFrame({
        'Set': ['Train', 'Test'],
        'RMSE': [train_rmse, test_rmse],
        'MAE': [train_mae, test_mae],
        'R2': [train_r2, test_r2],
        'MSE': [train_mse, test_mse]
    })
    return metrics_df

def compute_composite_score(df, feature_names, top_features, feature_importance):
    # Select numerical columns from the DataFrame
    numerical_data = df.select_dtypes(include=['number'])
    # Calculate the maximum and minimum values for each numerical column
    max_values = numerical_data.max()
    min_values = numerical_data.min()
    # Normalize the numerical data to a range between 0 and 1
    normalized_data = (numerical_data - min_values) / (max_values - min_values)
    # Create a dictionary mapping feature names to their importance scores
    weights = dict(zip([feature_names[i] for i in top_features], feature_importance[top_features]))
    # Calculate the total importance score across all selected features
    total_importance = sum(weights.values())
    # Normalize the importance scores to sum up to 1
    for feature in weights:
        weights[feature] /= total_importance
    # Compute the composite score by multiplying each normalized feature value by its corresponding importance weight,
    # and summing up across all features for each row
    composite_score = (normalized_data * pd.Series(weights)).sum(axis=1)
    # Assign the computed composite scores to a new column in the DataFrame
    df['composite_score'] = composite_score
    return df

def main():
    df = load_data()
    df = preprocess_data(df)
    pipeline, X, y = define_transformer_and_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train_pred, y_test_pred = train_model(pipeline, X_train, y_train, X_test)
    # We retrieve the most important features coming from the boosting model 
    feature_importance = pipeline.named_steps['model'].feature_importances_
    # We sort the feature depending on their importance 
    sorted_idx = np.argsort(feature_importance)[::-1]
    # We select the top features we are intered in for the dasboard 
    top_features = sorted_idx[:NUMBER_OF_FEATURES_WE_CONSIDER]
    # The categorical features were encoded using one hot encoding. The same preprocessing needs to be applied to 
    # retrieve the features names after encoding
    one_hot_encoded_feature_names = pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(input_features=['sex', 'address', 'famsize', 'Pstatus', 'reason', 'guardian', 'Mjob', 'Fjob'])
    # The remaining features won't have their names changed
    remaining_feature_names = [col for col in X.columns if col not in ['sex', 'address', 'famsize', 'Pstatus', 'reason', 'guardian', 'Mjob', 'Fjob']]
    # Final feature names list 
    feature_names = list(one_hot_encoded_feature_names) + remaining_feature_names
    # We compute the metrics that will be displayed so that the user can assess how reliable the model is
    metrics_df = compute_metrics(y_train, y_train_pred, y_test, y_test_pred)
    # We compute a composite score 
    df = compute_composite_score(df, feature_names, top_features, feature_importance)

    ###############
    ## Dashboard ##
    ###############
    st.title("Performance Dashboard")
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(top_features)), feature_importance[top_features], align="center")
    ax.set_xticks(range(len(top_features)))
    ax.set_xticklabels([feature_names[i] for i in top_features], rotation=45, ha='right')
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    st.pyplot(fig)

    st.subheader("Composite Score vs Final Grade")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['FinalGrade'], df['composite_score']*10, alpha=0.5)
    ax.set_title('Composite Score vs Final Grade')
    ax.set_ylabel('Composite Score')
    ax.set_xlabel('Final Grade')
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Evaluation Metrics")
    st.table(metrics_df)

if __name__ == "__main__":
    main()
