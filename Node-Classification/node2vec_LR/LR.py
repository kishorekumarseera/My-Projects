from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import node2vec

loaded_model = node2vec.main()

X=[]
T=[]
Y={}

with open("../cora_train.cites", "r") as file:
    # Read the entire contents of the file
    for line in file:
        line = line.strip()
        line = line.split('\t')
        X.append(line[1])

with open("../cora.content", "r") as file:
    # Read the entire contents of the file
    for line in file:
        line = line.strip()
        line = line.split('\t')
        Y[line[0]]=line[-1]

with open("../cora_test.cites", "r") as file:
    # Read the entire contents of the file
    for line in file:
        line = line.strip()
        line = line.split('\t')
        T.append(line[1])

X_train=[]
y_train=[]
X_test=[]
y_test=[]

for i in X:
    X_train.append(loaded_model.wv[i])
    y_train.append(Y[i])

for i in T:
    X_test.append(loaded_model.wv[i])
    y_test.append(Y[i])

# Initializing the logistic regression model with specified parameters
logistic_model = LogisticRegression(C=12, solver="liblinear")

# Fitting the model on the training data
logistic_model.fit(X_train, y_train)

# Making predictions on the testing data
predictions = logistic_model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, predictions)

with open("lr_metrics.txt",'w') as fp:
    fp.write(classification_report(y_test, predictions))

