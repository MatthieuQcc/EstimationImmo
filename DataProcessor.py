import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from math import radians, sin, cos, asin, sqrt


class DataProcessor:
    """Classe complète pour charger et préparer les données depuis un fichier CSV.
    Retourne X_train, X_test, y_train, y_test prêts pour l'entraînement."""
    
    STATIONS_TOULOUSE = {
        # Ligne A
        "Balma-Gramont": (43.62971164231746, 1.4831444388334585),
        "Argoulets": (43.62439184264503, 1.4768072274086192), 
        "Roseraie": (43.619870148945814, 1.4693228826532732),
        "Jolimont": (43.61538472316486, 1.4636057521703651),
        "Marengo – SNCF": (43.61072986498919, 1.4550073139327766),
        "Jean Jaurès": (43.60585995727777, 1.4491864712489415),
        "Capitole": (43.60419516166457, 1.4449659097167153),
        "Esquirol": (43.60030772301489, 1.4441848939899071),  
        "Saint-Cyprien – République": (43.597832326764525, 1.4317676052962165),
        "Patte d'Oie": (43.59630560148399, 1.4233691391055898),
        "Arènes": (43.59336285730645, 1.418306789899077),  
        "Fontaine-Lestang": (43.58751549085796, 1.4183833603785048),
        "Mermoz": (43.58345482990229, 1.4153180750828234),  
        "Bagatelle": (43.579898082395694, 1.4119812195216574),
        "Mirail – Université": (43.574814562246765, 1.402110554498417),
        "Reynerie": (43.57071434748996, 1.4019491169204168),  
        "Bellefontaine": (43.56609357794097, 1.398335491324841),
        "Basso Cambo": (43.57002423377021, 1.3922718601581425),

        # Ligne B
        "Borderouge": (43.64097358525123, 1.452298505170243),
        "Trois Cocus": (43.6382946007621, 1.4440672462982207),
        "La Vache": (43.633626034149714, 1.4349464413809196),
        "Barrière de Paris": (43.62661607705671, 1.4337725625189777),
        "Minimes": (43.62057375762567, 1.4358736943998551),
        "Canal du Midi": (43.61535261294454, 1.4337298761297568),
        "Compans-Caffarelli": (43.61067018931617, 1.4357931429011612),
        "Jeanne d'Arc": (43.608577144508914, 1.4457416228090167),
        "François Verdier": (43.6004629288437, 1.452297507755522),
        "Carmes": (43.597852135960615, 1.4454169016576714),
        "Palais de Justice": (43.59220811162977, 1.444592987028543),
        "Saint-Michel Marcel-Langer": (43.58604790672622, 1.4471783419698658),
        "Empalot": (43.57991639870424, 1.442075253939026),
        "Saint-Agne SNCF": (43.57970775512508, 1.450212702539849),
        "Saouzelong": (43.579494535801096, 1.4593810546013988),
        "Rangueil": (43.57481310670238, 1.4619417649912032),
        "Faculté de Pharmacie": (43.56803581915576, 1.4645477034498953),
        "Université-Paul-Sabatier": (43.56074285864322, 1.4624382220957395),
        "Ramonville": (43.55571867873419, 1.4757832659295467)
    }
       
    def __init__(self, csv_path, test_size=0.3, random_state=42):
        """Initialise le DataProcessor.
        
        Args:
            csv_path (str): Chemin vers le fichier CSV
            test_size (float): Proportion des données de test (défaut: 0.3)
            random_state (int): Seed pour la reproduction (défaut: 42)
        """
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.numeric_cols = [
        'lot1_surface_carrez', 'nombre_pieces_principales',
        'latitude', 'longitude', 'has_terrain', 'nearest_metro_distance_km'
        ]    
        self.cat_cols = ['nearest_metro_name']
        self.feature_cols = self.numeric_cols + self.cat_cols
    
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Calcule la distance en kilomètres entre deux coordonnées GPS."""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # rayon de la Terre en km
        return c * r
    
    def get_nearest_metro(self, lat, lon):
        """Trouve la station de métro la plus proche et sa distance."""
        min_dist = np.inf
        nearest_name = None
        for name, coords in self.STATIONS_TOULOUSE.items():
            d = self.haversine(lat, lon, coords[0], coords[1])
            if d < min_dist:
                min_dist = d
                nearest_name = name
        return pd.Series([min_dist, nearest_name])
    
    def build_dataframe_gouv(self, df):
        """Nettoie et prépare le DataFrame."""
        df = df.copy()

        # 1) Renommer la colonne valeur_fonciere -> price si besoin
        if 'price' not in df.columns and 'valeur_fonciere' in df.columns:
            df['price'] = df['valeur_fonciere']

        # Ajout d'une colonne indiquant la présence d'un terrain
        df['has_terrain'] = df['surface_terrain'].notna().astype(int)

        # 2) Garder uniquement les colonnes utiles
        cols_to_keep = ['price'] + self.numeric_cols + self.cat_cols
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        df_flat = df[cols_to_keep].copy()

        # 3) Convertir price en numérique (au cas où ce soit une string)
        df_flat['price'] = pd.to_numeric(df_flat['price'], errors='coerce')

        # 4) Drop NaN sur price
        df_flat_clean = df_flat.dropna(subset=['price']).copy()

        # 5) Garder uniquement les prix plausibles
        df_flat_clean = df_flat_clean[
            (df_flat_clean['price'] >= 15000) &
            (df_flat_clean['price'] <= 2500000)
        ]

        # 6) Supprimer les lignes avec NaN dans les colonnes importantes
        cols_for_nan = [c for c in (self.numeric_cols + self.cat_cols) if c in df_flat_clean.columns]

        df_flat_clean = df_flat_clean.dropna(subset=cols_for_nan)

        # 7) Convertir les colonnes numériques
        df_flat_clean["latitude"] = pd.to_numeric(df_flat_clean["latitude"], errors="coerce")
        df_flat_clean["longitude"] = pd.to_numeric(df_flat_clean["longitude"], errors="coerce")
        df_flat_clean["lot1_surface_carrez"] = pd.to_numeric(df_flat_clean["lot1_surface_carrez"], errors="coerce")
        df_flat_clean["nombre_pieces_principales"] = pd.to_numeric(df_flat_clean["nombre_pieces_principales"], errors="coerce")

        # 8) Appliquer les colonnes custom metro sur le DataFrame
        df_flat_clean[["nearest_metro_distance_km", "nearest_metro_name"]] = df_flat_clean.apply(
            lambda row: self.get_nearest_metro(row["latitude"], row["longitude"]),
            axis=1
        )

        print("Taille finale dataframe :", len(df_flat_clean))
        return df_flat_clean
    
    def load_and_split(self):
        """Charge le CSV, prépare les données et retourne le train/test split.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Charger le CSV
        df_raw = pd.read_csv(self.csv_path, low_memory=False)
        
        # Construire le DataFrame nettoyé
        df_clean = self.build_dataframe_gouv(df_raw)
        
        # Préparer X et y
        X = df_clean[self.feature_cols]
        y = df_clean['price']
        
        # Faire le train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
