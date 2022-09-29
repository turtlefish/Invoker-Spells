import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import numpy as np
import requests
import time

INVO_ID = 74
QUAS_ID = 5370
WEX_ID = 5371
EXORT_ID = 5372


def is_exort_build(spells):
    counts = {QUAS_ID: 0, WEX_ID: 0, EXORT_ID: 0}
    
    for spell in spells[:9]:
        if spell in counts:
            counts[spell] += 1
    
    return counts[EXORT_ID] > counts[WEX_ID]

def convert_hero_list_to_ids(hero_list):
    return [heroes_dict[hero] for hero in hero_list]

def parse_json(file_name):
    input_dataframe = pd.read_json(file_name)
    parsed_data = []

    for match_id in input_dataframe["match_id"]:
        print(match_id)
        req = requests.get("https://api.opendota.com/api/matches/{match_id}".format(match_id=match_id))

        players = pd.DataFrame(req.json()["players"])
        hero_ids = players["hero_id"].tolist()
        spells = players.loc[players["hero_id"] == INVO_ID]["ability_upgrades_arr"].tolist()[0]

        if spells:
            if INVO_ID in hero_ids[5:10]:
                hero_ids = hero_ids[5:10] + hero_ids[:5]
            
            hero_ids.remove(INVO_ID)
            parsed_data.append([hero_ids[:4], hero_ids[4:], is_exort_build(spells)])
            print([hero_ids[:4], hero_ids[4:], is_exort_build(spells)])

        time.sleep(1.1)

    return parsed_data


# parsed_data = parse_json("explorer.json")
# df = pd.DataFrame(parsed_data, columns=["ally_heroes", "enemy_heroes", "is_exort_build"])
# df.to_json("parsed_data.json", index=False)

heroes = pd.read_csv("heroes.csv")
heroes_dict = heroes.set_index("name").to_dict()["id"]

df = pd.read_json('parsed_data.json')

mlb_enemy = MultiLabelBinarizer()

mlb_enemy_out = mlb_enemy.fit_transform(df.enemy_heroes)


df = df.join(pd.DataFrame(mlb_enemy_out, columns=mlb_enemy.classes_))

df.drop(["enemy_heroes", "ally_heroes"], axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(df.drop("is_exort_build", axis=1), df["is_exort_build"], test_size=0.2, stratify=df["is_exort_build"])#, random_state=1)

l1model = LogisticRegression(penalty='l2', solver='lbfgs', random_state=0, max_iter=10000, C=1000)
rmodel = RidgeClassifier(alpha=0.1, solver='auto', random_state=0, max_iter=10000)
mlpmodel = MLPClassifier(hidden_layer_sizes=(120, 60), random_state=0, activation='relu', solver='sgd', max_iter=10000)

l1model.fit(x_train, y_train)
y_pred_l1 = l1model.predict(x_test)
print("l1", l1model.score(x_test, y_test), np.unique(y_pred_l1, return_counts=True))

rmodel.fit(x_train, y_train)
y_pred_r1 = rmodel.predict(x_test)
print("r1", rmodel.score(x_test, y_test), np.unique(y_pred_r1, return_counts=True))

mlpmodel.fit(x_train, y_train)
y_pred_mlp = mlpmodel.predict(x_test)
print("mlp", mlpmodel.score(x_test, y_test), np.unique(y_pred_mlp, return_counts=True))

most_common = np.full(len(y_test), False)
print("false", accuracy_score(y_test, most_common))

# print("###### TEST CASE ######")
# test = [convert_hero_list_to_ids(["bloodseeker", "earthshaker", "templar assassin", "undying", "disruptor"])]
# print("l1", l1model.predict(mlb_enemy.transform(test)), l1model.predict_proba(mlb_enemy.transform(test)))
# print("r1", rmodel.predict(mlb_enemy.transform(test)))
# print("l1", mlpmodel.predict(mlb_enemy.transform(test)), mlpmodel.predict_proba(mlb_enemy.transform(test)))
