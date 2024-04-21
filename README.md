# school_help_dashboard

This work presents a streamlit app, in the form of a dashboard with three elements:

* A feature importance graph outlining which features impacted the most the model - this would help user to fous on specific variables that truly matter
* A Graph showing a composite score vs the final grade
    * Here the final grades are all present in the dataframe; however we build a boosting model predicting the final grade from the remaining features using 70% of the data as training and 30% to evaluate the model. Since the dataset is extremly small, no cross-validation was carried out and we unfortunately observe a low performance on the test set. The possibility to add new data in `./data/new_data.csv` adds the scenario of retraining automatically the model, which could hopefully increase the performance on the test set. The model's purpose is also to be able to predict final grades, before they occur, by leveraging simply the covariate. 
    * To evaluate the composite or improbability score for each student, we consider various factors that may indicate the level of support they require. We use the variables that impact the most the prediction of the final score. This score can then be used to prioritize students on the dashboard graph, with higher scores indicating a greater need for support. To compute a composite score or improbability for each student, we create a weighted sum of the different indicators mentioned earlier.
* A table with the evaluation metrics, which allows the user to have a certain level of trust depending on the performance of the test set. Furthermore, the hyperparameters of the model might require some change is a large overfit is observed when comparing train and test sets. A GridSearch or Optuna could be later added in the code, as soon as further data is added. 
 
# Running the dashboard 

Run `streamlit run app/dashboard.py` if you want to run the app locally 

Run `docker build -t my-streamlit-app .`, as root, if you want to build the docker app locally 

If Docker is not started, run `sudo systemctl start docker`

To run the app, simply type `docker run -p 8501:8501 my-streamlit-app`
