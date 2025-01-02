import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
"""Variables explicatives
load_forecast : prévision de consommation totale d'éléctricité en France
coal_power_available, gas_power_available, nucelear_power_available : capacité de production totale d'électricité des centrales à charbon, gaz et nucléaire respectivement,
wind_power_forecasts_average, solar_power_forecasts_average : moyenne de différentes prévisions de production totale d'électricité éolienne et solaire (respectivement),
wind_power_forecasts_std, solar_power_forecasts_std : écart-type de ces mêmes prévisions,
predicted_spot_price : prévision du prix SPOT de l'électricité issues d'un modèle interne de Elmy. Ce modèle est lancé chaque jour avant la fermeture des enchères SPOT pour le lendemain.

Variable cible
spot_id_delta : l'écart entre le VWAP des transactions sur le marché infra-journalier (Intraday) et le prix SPOT pour 1MWh d'électricité (spot_id_delta = Intraday - SPOT) : si la valeur est positive, le prix Intraday est supérieur au prix SPOT et inversement."""

x_train = pd.read_csv("X_train_Wwou3IE.csv")
y_train = pd.read_csv("y_train_jJtXgMX.csv")

x_train['DELIVERY_START'] = pd.to_datetime(x_train['DELIVERY_START'], utc="True")

print(x_train['DELIVERY_START'].dtype)

xt_clean1 = x_train.dropna(subset=["coal_power_available", 
                                      "gas_power_available", 
                                      "nuclear_power_available", 
                                      "wind_power_forecasts_average", 
                                      "solar_power_forecasts_average",
                                      "wind_power_forecasts_std",
                                      "solar_power_forecasts_std"])

x_train = x_train.dropna()
x_train = x_train.drop_duplicates()


cols = ['coal_power_available', 'gas_power_available', 'nuclear_power_available', 
        'wind_power_forecasts_average', 'solar_power_forecasts_average', 
        'wind_power_forecasts_std', 'solar_power_forecasts_std', 'predicted_spot_price']

Q1 = x_train['coal_power_available'].quantile(0.25)
Q3 = x_train['coal_power_available'].quantile(0.75)
IQR = Q3 - Q1

# Définir les limites inférieure et supérieure
for col in cols:
    Q1 = x_train[col].quantile(0.25)
    Q3 = x_train[col].quantile(0.75)
    IQR = Q3 - Q1

    # Calcul des bornes inférieure et supérieure
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrer les données pour enlever les outliers
    x_train = x_train[(x_train[col] >= lower_bound) & (x_train[col] <= upper_bound)]
    
print("Valeurs aberrantes supprimées pour toutes les colonnes.")

"""x_train_numeric = x_train.select_dtypes(include=[np.number])  # Sélectionner uniquement les colonnes numériques
x_train_datetime = x_train.select_dtypes(include=['datetime64[ns]'])  # Sélectionner les colonnes datetime (si nécessaire)

# Appliquer la normalisation uniquement sur les données numériques
scaler = StandardScaler()

# Normaliser les données numériques
x_train_scaled = scaler.fit_transform(x_train_numeric)

# Convertir x_train_scaled en DataFrame avec les colonnes d'origine
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train_numeric.columns)

# Concaténer les colonnes normalisées avec les colonnes datetime si nécessaire
x_train_final = pd.concat([x_train_scaled, x_train_datetime], axis=1)"""




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Séparer les variables explicatives (X) et la cible (y)
X = x_train
y = y_train
y = y_train['spot_id_delta']  # Remplace 'spot_id_delta' par la colonne correcte


"""
# Aligner les indices de X et y
X = x_train
y = y.reindex(X.index)  # Assure-toi que y a le même index que X

# Séparer les données en jeu d'entraînement et jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir la colonne DELIVERY_START en caractéristiques temporelles
X_train['year'] = X_train['DELIVERY_START'].dt.year
X_train['month'] = X_train['DELIVERY_START'].dt.month
X_train['day'] = X_train['DELIVERY_START'].dt.day
X_train['hour'] = X_train['DELIVERY_START'].dt.hour

X_test['year'] = X_test['DELIVERY_START'].dt.year
X_test['month'] = X_test['DELIVERY_START'].dt.month
X_test['day'] = X_test['DELIVERY_START'].dt.day
X_test['hour'] = X_test['DELIVERY_START'].dt.hour

# Maintenant tu peux supprimer la colonne originale DELIVERY_START
X_train = X_train.drop(columns=['DELIVERY_START'])
X_test = X_test.drop(columns=['DELIVERY_START'])

y_train['spot_id_delta'] = pd.cut(y_train['spot_id_delta'], 
                                        bins=[-np.inf, 0, np.inf], 
                                        labels=[0, 1])
print(y_train.columns)
y_train.columns = ['spot_id_delta']
print(y_train.columns)

# Créer un modèle Random Forest
rf = RandomForestClassifier()

# Entraîner le modèle
rf.fit(X_train, y_train)

# Afficher l'importance des caractéristiques
importances = rf.feature_importances_

# Créer un DataFrame pour visualiser les importances des caractéristiques
importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

# Afficher les 10 caractéristiques les plus importantes
print(importance_df.head(10))

# Optionnel : afficher un graphique
importance_df.head(10).plot(kind='bar', x='feature', y='importance', legend=False)
plt.title("Importance des caractéristiques")
plt.show()


"""
# Extraire la cible et transformer en classes binaires
y = y_train['spot_id_delta']
y = pd.cut(y, bins=[-np.inf, 0, np.inf], labels=[0, 1]).astype(int)

# Aligner les indices après nettoyage
X = x_train
y = y.loc[X.index]

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir 'DELIVERY_START' en caractéristiques temporelles
X_train['year'] = X_train['DELIVERY_START'].dt.year
X_train['month'] = X_train['DELIVERY_START'].dt.month
X_train['day'] = X_train['DELIVERY_START'].dt.day
X_train['hour'] = X_train['DELIVERY_START'].dt.hour

X_test['year'] = X_test['DELIVERY_START'].dt.year
X_test['month'] = X_test['DELIVERY_START'].dt.month
X_test['day'] = X_test['DELIVERY_START'].dt.day
X_test['hour'] = X_test['DELIVERY_START'].dt.hour

# Supprimer la colonne d'origine
X_train.drop(columns=['DELIVERY_START'], inplace=True)
X_test.drop(columns=['DELIVERY_START'], inplace=True)

# Entraîner le modèle Random Forest
"""rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)"""

# Entraîner avec les meilleurs paramètres trouvés
best_params = {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}

# Créer un modèle Random Forest avec les meilleurs paramètres
rf_optimized = RandomForestClassifier(
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    n_estimators=best_params['n_estimators'],
    random_state=42  # Pour des résultats reproductibles
)

# Entraîner le modèle sur les données d'entraînement
rf_optimized.fit(X_train, y_train)


# Afficher l'importance des caractéristiques
importances = rf_optimized.feature_importances_
importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
print(importance_df.sort_values(by='importance', ascending=False).head(10))


# Optionnel : afficher un graphique
importance_df.head(10).plot(kind='bar', x='feature', y='importance', legend=False)
plt.title("Importance des caractéristiques")
plt.show()

"""# Calculer la matrice de corrélation
corr_matrix = x_train.corr()

# Afficher la matrice de corrélation
print(corr_matrix)

# Tu peux aussi visualiser la matrice de corrélation avec un heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()"""



"""plt.figure(figsize=(10, 6))
sns.boxplot(data=xt_clean, y='coal_power_available')
plt.title('Boxplot de coal_power_available')
plt.show()"""


import seaborn as sns
import matplotlib.pyplot as plt

"""for feature in importance_df.head(5)['feature']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=y_train, y=X_train[feature])
    plt.title(f'Impact de {feature} sur la classe cible')
    plt.show()"""


from sklearn.model_selection import GridSearchCV

"""param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Meilleurs paramètres :", grid_search.best_params_)"""

# Faire des prédictions sur les données de test
y_pred = rf_optimized.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle optimisé : {accuracy:.4f}")

# Afficher le rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Afficher la matrice de confusion
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

yayyy = pd.read_csv("X_test_GgyECq8.csv")

yayyy['DELIVERY_START'] = pd.to_datetime(yayyy['DELIVERY_START'], utc="True")

yayyy['year'] = yayyy['DELIVERY_START'].dt.year
yayyy['month'] = yayyy['DELIVERY_START'].dt.month
yayyy['day'] = yayyy['DELIVERY_START'].dt.day
yayyy['hour'] = yayyy['DELIVERY_START'].dt.hour

yayyy.drop(columns=['DELIVERY_START'], inplace=True)

y_pred = rf_optimized.predict(yayyy)

predictions_df = pd.DataFrame({
    'spot_id_delta': y_pred  # Initialement les prédictions
})

predictions_df['spot_id_delta'] = predictions_df['spot_id_delta'].replace(0, -1)

start_time = pd.to_datetime("2023-04-02 00:00:00+02:00")
predictions_df['DELIVERY_START'] = pd.date_range(start=start_time, periods=len(predictions_df), freq='H')

predictions_df = predictions_df[['DELIVERY_START', 'spot_id_delta']]

# 7. Sauvegarder le DataFrame dans un fichier CSV (y_pred.csv par exemple)
predictions_df.to_csv('y_pred222.csv', index=False)

# Afficher un aperçu des données générées pour vérifier
print(predictions_df.head())

import pandas as pd

# Charger le fichier CSV
df = pd.read_csv('y_random_pt8afo8.csv')

# Convertir la colonne 'DELIVERY_START' en datetime
df['DELIVERY_START'] = pd.to_datetime(df['DELIVERY_START'])

# Vérifier s'il y a des trous dans la séquence de dates
def check_missing_dates(df):
    # Trier les dates par ordre croissant
    df = df.sort_values(by='DELIVERY_START')

    # Créer une plage de dates sans trous, basée sur le premier et le dernier timestamp
    date_range = pd.date_range(start=df['DELIVERY_START'].iloc[0], 
                               end=df['DELIVERY_START'].iloc[-1], 
                               freq='H')  # fréquence horaire

    # Vérifier si toutes les dates attendues sont présentes
    missing_dates = set(date_range) - set(df['DELIVERY_START'])

    if len(missing_dates) > 0:
        print(f"Des trous ont été trouvés dans la séquence de dates ! Voici les dates manquantes :")
        for missing in sorted(missing_dates):
            print(missing)
    else:
        print("Aucun trou trouvé, la séquence de dates est complète.")

# Exécuter la fonction pour vérifier les dates manquantes
check_missing_dates(df)


y_pred = pd.read_csv('y_pred222.csv')
x_test = pd.read_csv('X_test_GgyECq8.csv')

# S'assurer que les indices des deux fichiers sont bien alignés
y_pred = y_pred.set_index(x_test.index)

# Sauvegarder le fichier réindexé
y_pred.to_csv('y_pred_aligned.csv', index=True)
