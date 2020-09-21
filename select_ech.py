import csv
import pandas as pd
from sklearn.model_selection import train_test_split


lyrics = pd.read_csv('/lyrics_final.csv', sep=';', names = ["Artiste", "Titre", "Lien", "Paroles"])
df = pd.DataFrame(lyrics)
lyrics["Artiste"].value_counts() # nombre d'entrées pour chaque chanteur

""" sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state = 0)# 
strat_test_set = []
for ech_index in sss.split(lyrics, lyrics["Artiste"]):
    strat_test_set.append(ech_index) """



ech1, ech2 = train_test_split(lyrics, stratify=lyrics['Artiste'], test_size=500)
ech2.to_csv("ech_a_labeliser.csv")