from DataProcessor import DataProcessor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


class TrainModel:

    def __init__(self, csv_path):  
        """Initialise le TrainModel.
        Args:
            csv_path (str): Chemin vers le fichier CSV
            Création de la classe DataProcessor à partir du csv
        """
        self.dataframe = DataProcessor(csv_path)
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataframe.load_and_split()


    def get_model(self):
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.dataframe.numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.dataframe.cat_cols)
        ])
  
        model = Pipeline([
            ('preproc', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        return model
    
    def train_model(self): 
        model = self.get_model()

        param_grid = {
            # hyperparamètres du RandomForest
            'regressor__n_estimators': [50, 60, 100, 125, 150, 200],
            'regressor__max_depth': [None, 5, 10, 15, 20],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2', 0.5],
        }

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            cv=5,
            n_iter=100,
            scoring='neg_mean_absolute_error',  # ou 'r2', 'neg_mean_squared_error', etc.
            n_jobs=-1,
            verbose=2
        )

        random_search.fit(self.X_train, self.y_train)

        print("Meilleurs paramètres :", random_search.best_params_)
        print("Score CV (MAE négative) :", random_search.best_score_)

        return random_search.best_estimator_


    def test_model(self):
        # évaluer sur le test set
        best_model = self.train_model()
        y_pred = best_model.predict(self.X_test)

        print("MAE test :", mean_absolute_error(self.y_test, y_pred))
        print("R2  test :", r2_score(self.y_test, y_pred))
