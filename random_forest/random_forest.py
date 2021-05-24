import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

iris = datasets.load_iris()
features = iris.data
target = iris.target


randomforest = RandomForestClassifier(random_state=0, n_jobs=1)

model = randomforest.fit(features, target)

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]
names = [iris.feature_names[i] for i in indices]

plt.figure()
plt.title("Feature importance")
plt.xticks(range(features.shape[1]), names, rotation = 90)
plt.bar(range(features.shape[1]), importances[indices])
plt.savefig("testfig.png")

features

import pandas as pd
df = pd.read_csv("out-film-temp.csv")
df2 = df.dropna(how='any').dropna(how='all', axis=1)

import pydotplus
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn import tree
df2 = df2
features = df2[0 : 260]
target = df2["out_filmtemp"]

plt.hist(target, bins=50)

decisiontree = DecisionTreeClassifier(random_state=0)
model = decisiontree.fit(features, target)
dot_data = tree.export_graphviz(decisiontree, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.progs = {'dot': u"C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe"}
Image(graph.create_png())

graph.write_png("iris.png")
