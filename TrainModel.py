from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from DataProcessor import DataProcessor

class TrainModel:
    def __init__(self, csv_path):
        # Initialisation du DataProcessor
        self.dataframe = DataProcessor(csv_path)

        # Chargement et split automatique des données
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataframe.load_and_split()

        # Colonne numériques et catégoriques (définies par DataProcessor)
        self.numeric_cols = self.dataframe.numeric_cols
        self.cat_cols = self.dataframe.cat_cols

        # Pour stocker le meilleur modèle après entraînement
        self.best_model = None

    def get_pipeline(self):
        """Crée le pipeline avec préprocesseur et modèle"""
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.cat_cols)
        ])

        pipeline = Pipeline([
            ('preproc', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        return pipeline

    def train(self):
        """Recherche d'hyperparamètres et entraînement final"""
        pipeline = self.get_pipeline()

        param_grid = {
            'regressor__n_estimators': [50, 60, 100, 125, 150, 200],
            'regressor__max_depth': [None, 5, 10, 15, 20],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2', 0.5],
        }

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            cv=3,
            n_iter=50,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=2
        )

        search.fit(self.X_train, self.y_train)

        print("Meilleurs paramètres :", search.best_params_)
        print("Score CV (MAE négative) :", search.best_score_)

        # Stocker le meilleur modèle pour réutilisation
        self.best_model = search.best_estimator_

    def test(self):
        """Évaluer le modèle sur le test set"""
        if self.best_model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez `train()` d'abord.")

        y_pred = self.best_model.predict(self.X_test)
        print("MAE test :", mean_absolute_error(self.y_test, y_pred))
        print("R2  test :", r2_score(self.y_test, y_pred))
