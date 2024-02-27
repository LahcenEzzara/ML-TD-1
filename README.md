# ML-TD-1

## Python Code :

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv("train.csv").to_numpy()
clf = DecisionTreeClassifier()
x = data[0:2100, 1:]
label = data[0:2100, 0]
clf.fit(x, label)
xtest = data[2100:, 1:]
actual_label = data[2100:, 0]
p = clf.predict(xtest)

count = 0
for i in range(0, len(p)):
    count += 1 if p[i] == actual_label[i] else 0
print("Accuracy=", (count / len(p)) * 100)

d = xtest[5]
Nombre_de_pixels_errones = 100
for i in range(Nombre_de_pixels_errones):
    position = np.random.randint(0, 784, 1)[0]
    bruit = np.random.randint(-200, 200, 1)[0]
    d[position] += bruit
    d[position] = d[position] % 255
print(clf.predict([d]))
d.shape = (28, 28)
plt.imshow(255 - d, cmap='gray')
plt.show()
```

## Figure :

![Figure](Figure_1.png)


## Results :

Accuracy= 74.7276688453159
[2]