# %% [markdown]
# # KNN or K-Nearest Neighbours

# %% [markdown]
# KNN is a lazy learning algorithm which is used for classification. In this algorithm, the unknown or test set data will consider 'K' nearest neighbours from training set and predict the class based on count of its 'K' nearest neighbours i.e Among it's K nearest neighbours, which ever class has the highest count, it will be assigned to that particular class. Steps involved in this algorithm are:
#
# 1. Choose the number K of neighbours
# 2. Take the K nearest neighbours of the new data point, according to Euclidian Distance
# 3. Among the data points, count the number of datapoints in each category.
# 4. Assign the new data point to the category where you counted the most neighbours.

# %% [markdown]
# # Setup model

# %%
# imporing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import streamlit as st
from matplotlib.colors import ListedColormap
from Visualize import Visualization
# %%
# feature scaling
from FeatureScaling import FeatureScaling
from LDA import LDA
from KNN import KNN

# %%
st.title("Wine Customer Segmentation App")

st.sidebar.header("Customize Hyperparamer")

K = st.sidebar.slider("Number of neighbors", 1, 10, 5)
train_percent = st.sidebar.slider("Train-test split", 0.0, 1.0, 0.75)
test_percent = 100 - train_percent

# %% [markdown]
# # Preprocess data

# %%
# reading dataset
# Data=pd.read_csv('Social_Network_Ads.csv')

st.header("Data loading")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    Data = pd.read_csv(uploaded_file)
    st.write(Data.describe())

# %%
# Getting features from dataset
Data = Data.sample(frac=1)
# X=Data.iloc[:,[2, 3]].values
# y=Data.iloc[:,4].values
X = Data.iloc[:, :-1].values
y = Data.iloc[:, -1].values
X = X.astype(float)
print(X)

# %%
st.header("Data Preprocessing")

# feature scaling
st.subheader("Feature Scaling")

fs = FeatureScaling(X, y)
X = fs.fit_transform_X()

st.write(X)
# ss=StandardScaler()
# X=ss.fit_transform(X)

# %%
st.subheader("Multi-dimensional data visualization")

clr = ['red', 'blue', 'green']
# create color array based on labels
colors = [clr[int(y[i]) - 1] for i in range(X.shape[0])]

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors)

st.pyplot(fig)

# %%

st.subheader("Reduce dimension with Linear Discriminant Analysis")
lda = LDA(2)
X = lda.transform(X, y)
print(X)

fig = plt.figure(figsize=(6, 6))
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], c=clr[y[i]-1])
st.pyplot(fig)

# %%
# training and testing set size
train_size = int(train_percent*Data.shape[0])
test_size = int(test_percent*Data.shape[0])

print("Training set size : " + str(train_size))
print("Testing set size : "+str(test_size))

# %%
# training set split
X_train = X[0:train_size, :]
Y_train = y[0:train_size]
print(X_train, Y_train)
# %%
# testing set split
X_test = X[train_size:, :]
Y_test = y[train_size:]
print(X_test, Y_test)
# %% [markdown]
# # Training

# %%
st.header('Training and testing model')

if st.button("Run", type="primary"):
    l = time.time()
    knn = KNN(X_train, Y_train, K)
    y_pred = knn.predict(X_test)
    r = time.time()

    KNN_learn_time = (r-l)
    st.write("Training time: ", KNN_learn_time)

    # getting the confusion matrix
    true_predict_1 = len(
        [i for i in range(0, Y_test.shape[0]) if Y_test[i] == 1 and y_pred[i] == 1])
    true_predict_2 = len(
        [i for i in range(0, Y_test.shape[0]) if Y_test[i] == 2 and y_pred[i] == 2])
    true_predict_3 = len(
        [i for i in range(0, Y_test.shape[0]) if Y_test[i] == 3 and y_pred[i] == 3])
    confusion_matrix = np.diag(
        [true_predict_1, true_predict_2, true_predict_3])

    st.write("Testing result:", confusion_matrix)

# # %%
# #Same algorithm using sklearn KNN just for comparsion purpose
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# l=time.time()
# classifier.fit(X_train, Y_train)
# y_pred_sklearn = classifier.predict(X_test)
# r=time.time()
# sklearn_time=(r-l)
# print(sklearn_time)

# # %%
# print("But sklearn time is faster than our implementation by: "+str(KNN_learn_time/sklearn_time)+" times")

# # %%
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(Y_test, y_pred_sklearn)
# print(cm)

# %% [markdown]
# # Visualization

# %%
# Visualising the Training set results for our implementation
st.subheader("Training results visualization")

l = time.time()

fig = Visualization.scatter(X_train, Y_train, knn)
st.pyplot(fig)

r = time.time()

print("Time required for plotting is: "+str(r-l)+" seconds")

# %%
# Visualising the Test set results for our implementation
st.subheader("Testing results visualization")

l = time.time()

fig = Visualization.scatter(X_test, Y_test, knn)
st.pyplot(fig)

r = time.time()

print("Time required for plotting is: "+str(r-l)+" seconds")

# # %%
# # Visualising the Training set results for sklearn class
# l=time.time()
# X_set, y_set = X_train, Y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'blue', 'yellow')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('green', 'orange', 'purple'))(i), label = j,marker='.')
# plt.title('K-NN (training set) using sklearn library')
# plt.xlabel('Wine')
# plt.ylabel('Customer Segment')
# plt.legend()
# plt.show()
# r=time.time()
# print("Time required for plotting is: "+str(r-l)+" seconds")

# # %%
# # Visualising the Test set results for sklearn class
# l=time.time()
# X_set, y_set = X_test, Y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'blue', 'yellow')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('green', 'orange', 'purple'))(i), label = j,marker='.')
# plt.title('K-NN (test set) using sklearn library')
# plt.xlabel('Wine')
# plt.ylabel('Customer Segment')
# plt.legend()
# plt.show()
# r=time.time()
# print("Time required for plotting is: "+str(r-l)+" seconds")

# # %% [markdown]
# # Conclusion is our implementation is slower but still we have achieved similar results compared to sklearn package
