import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, \
    ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import train_test_split

X = np.load("data/svm_x.npy")
print(X.shape)
y = np.load("data/svm_y.npy")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(gamma="auto")
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
cm = confusion_matrix(y_test.reshape(-1), y_predict.reshape(-1), normalize=None)
print(accuracy_score(y_test, y_predict))
print(balanced_accuracy_score(y_test, y_predict))
disp = ConfusionMatrixDisplay(cm, display_labels=["straight", "fwd lean", "bwd lean"])
disp.plot()
plt.savefig("confusion.png")