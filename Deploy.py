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
#imporing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import streamlit as st
from matplotlib.colors import ListedColormap

# %%
#feature scaling
class FeatureScaling:
    def __init__(self,X,y):
        self.X=X.copy()
        if y.ndim==1:
            y=np.reshape(y,(y.shape[0],1))
        self.y=y.copy()
        self.minMax_X={}
        self.minMax_y={}
    
    def fit_transform_X(self):
        num_of_features=self.X.shape[1]
        for i in range(num_of_features):
            feature=self.X[:,i]
            Mean=np.mean(feature)
            Min=np.min(feature)
            Max=np.max(feature)
            feature=(feature-Mean)/(Max-Min)
            self.minMax_X[i]=np.array([Mean,Min,Max])
            self.X[:,i]=feature
        return self.X.copy()
    
    def fit_transform_Y(self):
        num_of_features=self.y.shape[1]
        for i in range(num_of_features):
            feature=self.y[:,i]
            Mean=np.mean(feature)
            Min=np.min(feature)
            Max=np.max(feature)
            feature=(feature-Mean)/(Max-Min)
            self.minMax_y[i]=np.array([Mean,Min,Max])
            self.y[:,i]=feature
        return np.reshape(self.y,self.y.shape[0])
    
    def inverse_transform_X(self,X):
        X_transformed=X.copy()
        num_of_features=X_transformed.shape[1]
        for i in range(num_of_features):
            feature=X_transformed[:,i]
            Mean=self.minMax_X[i][0]
            Min=self.minMax_X[i][1]
            Max=self.minMax_X[i][2]
            feature=feature*(Max-Min)+Mean
            X_transformed[:,i]=feature
        return X_transformed
    
    def inverse_transform_Y(self,y):
        y_transformed=y.copy()
        if y_transformed.ndim==1:
            y_transformed=np.reshape(y_transformed,(y_transformed.shape[0],1))
        num_of_features=y_transformed.shape[1]
        for i in range(num_of_features):
            feature=y_transformed[:,i]
            Mean=self.minMax_y[i][0]
            Min=self.minMax_y[i][1]
            Max=self.minMax_y[i][2]
            feature=feature*(Max-Min)+Mean
            y_transformed[:,i]=feature
        return np.reshape(y_transformed,y_transformed.shape[0])
    
    def transform_X(self,X):
        X_transformed=X.copy()
        num_of_features=X_transformed.shape[1]
        for i in range(num_of_features):
            feature=X_transformed[:,i]
            Mean=self.minMax_X[i][0]
            Min=self.minMax_y[i][1]
            Max=self.minMax_y[i][2]
            feature=(feature-Mean)/(Max-Min)
            X_transformed[:,i]=feature
        return X_transformed
    
    def transform_Y(self,y):
        y_transformed=y.copy()
        if y_transformed.ndim==1:
            y_transformed=np.reshape(y_transformed,(y_transformed.shape[0],1))
        num_of_features=y_transformed.shape[1]
        for i in range(num_of_features):
            feature=y_transformed[:,i]
            Mean=self.minMax_y[i][0]
            Min=self.minMax_y[i][1]
            Max=self.minMax_y[i][2]
            feature=(feature-Mean)/(Max-Min)
            y_transformed[:,i]=feature
        return np.reshape(y_transformed,y_transformed.shape[0])
    
    def returnX(self):
        return self.X
    
    def returnY(self):
        return self.y

# %%
class LDA:

    def __init__(self, n_components): # number of dimensions after reduction
        self.n_components = n_components
        self.linear_discriminants = None

    def transform(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within class scatter matrix:
        # SW = sum((X_c - mean_X_c)^2 )

        # Between class scatter:
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            # (4, 1) * (1, 4) = (4,4) -> reshape
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = eigenvectors[0:self.n_components]

        return np.dot(X, self.linear_discriminants.T).astype(np.float64)

# %%
# class LDA:
#     def __init__(self, X, y):
#         self.X, self.y = X, y
#         self.class_dict = {}
#         self.mean_dict = {}

#     def reduce_dimension(self):
#         for i in range(self.y.shape[0]):
#             if self.y[i] not in self.class_dict:
#                 self.class_dict[self.y[i]]=self.X[i].reshape(1,self.X[i].shape[0])
#             else:
#                 self.class_dict[self.y[i]]=np.append(self.class_dict[self.y[i]],self.X[i].reshape(1,self.X[i].shape[0]),axis=0)

#         for each in self.class_dict:
#             self.mean_dict[each]=np.mean(self.class_dict[each],axis=0)

#         #compute With in class scatter matrix
#         n=len(self.class_dict)
#         Sw=np.array([])
#         for i in self.class_dict:
#             x=self.class_dict[i]
#             m=self.mean_dict[i]
#             temp=np.dot((x-m).T,(x-m))
#             if Sw.shape[0]==0:
#                 Sw=np.cov((x-m).T)
#                 #Sw=temp
#             else:
#                 #SW=Sw+temp
#                 SW=Sw+np.cov((x-m).T)


#         #computer between class Scatter matrix
#         Sb=np.array([])
#         Mean=np.mean(self.X,axis=0)
#         Mean=Mean.reshape(Mean.shape[0],1)

#         for i in self.mean_dict:
#             m=self.mean_dict[i].reshape(self.mean_dict[i].shape[0],1)
#             n=len(self.class_dict[i])
#             temp=np.multiply(n,np.dot((m-Mean),(m-Mean).T))
#             if Sb.shape[0]==0:
#                 Sb=temp
#             else:
#                 Sb=Sb+temp
                

#         #Sb=n*np.dot((MeanMatrix-MeanOfMeans).T,(MeanMatrix-MeanOfMeans))
#         #Sb=n*np.cov(MeanMatrix.T)    
#         #computee sw^-1Sb
#         eig_vals,eig_vecs=np.linalg.eigh(np.matmul(np.linalg.pinv(Sw),Sb))
#         eig_pairs=[(np.abs(eig_vals[i]),eig_vecs[i]) for i in range(len(eig_vals))]
#         eig_pairs=sorted(eig_pairs,key=lambda k:k[0],reverse=True)
#         W=np.hstack((eig_pairs[0][1].reshape(13,1),eig_pairs[1][1].reshape(13,1)))

#         X_new=np.dot(self.X,W)

#         return X_new

# %%
class KNN:
    def __init__(self,X_train,Y_train,K):
        self.X_train=X_train
        self.Y_train=Y_train
        self.K=K
        
    def predict(self,X):
        y_pred=np.array([])
        for each in X:
            ed=np.sum((each-self.X_train)**2,axis=1)
            y_ed=np.concatenate((self.Y_train.reshape(self.Y_train.shape[0],1),ed.reshape(ed.shape[0],1)),axis=1)
            y_ed=y_ed[y_ed[:,1].argsort()]
            K_neighbours=y_ed[0:self.K]
            (values,counts) = np.unique(K_neighbours[:,0].astype(int),return_counts=True)
            y_pred=np.append(y_pred,values[np.argmax(counts)])
        return y_pred
            
st.title("Wine Customer Segmentation App")

st.sidebar.header("Customize Hyperparamer")

K = st.sidebar.slider("Number of neighbors", 1, 10, 5)
train_percent = st.sidebar.slider("Train-test split", 0.0, 1.0, 0.75)
test_percent = 100 - train_percent

# %% [markdown]
# # Preprocess data

# %%
#reading dataset
# Data=pd.read_csv('Social_Network_Ads.csv')

st.header("Data loading")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    Data=pd.read_csv(uploaded_file)
    st.write(Data.describe())

# %%
#Getting features from dataset
Data=Data.sample(frac=1)
# X=Data.iloc[:,[2, 3]].values
# y=Data.iloc[:,4].values
X=Data.iloc[:,:-1].values
y=Data.iloc[:,-1].values
X=X.astype(float)
print(X)

# %%
st.header("Data Preprocessing")

#feature scaling
st.subheader("Feature Scaling")
if st.button("Scale", type="primary"):
    fs=FeatureScaling(X,y)
    X=fs.fit_transform_X()

    st.write(X)
# ss=StandardScaler()
# X=ss.fit_transform(X)

# %%
st.subheader("Multi-dimensional data visualization")

clr=['red','blue','green']
colors = [clr[int(y[i]) - 1] for i in range(X.shape[0])] # create color array based on labels

fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors)

st.pyplot(fig)

# %%

st.subheader("Linear Discriminant Analysis")
if st.button("Reduce dimension", type="primary"):
    lda=LDA(2)
    X=lda.transform(X,y)
    print(X)

    fig = plt.figure(figsize=(6,6))
    for i in range(X.shape[0]):
        plt.scatter(X[i,0],X[i,1],c=clr[y[i]-1])
    st.pyplot(fig)

# %%
#training and testing set size
train_size=int(train_percent*Data.shape[0])
test_size=int(test_percent*Data.shape[0])

print("Training set size : "+ str(train_size))
print("Testing set size : "+str(test_size))

# %%
#training set split
X_train=X[0:train_size,:]
Y_train=y[0:train_size]
print(X_train,Y_train)
# %%
#testing set split
X_test=X[train_size:,:]
Y_test=y[train_size:]
print(X_test, Y_test)
# %% [markdown]
# # Training

# %%
st.header('Training and testing model')

if st.button("Run", type="primary"):
    l=time.time()
    knn=KNN(X_train,Y_train,K)
    y_pred=knn.predict(X_test)
    r=time.time()

    KNN_learn_time=(r-l)
    st.write("Training time: ", KNN_learn_time)

    # getting the confusion matrix
    true_predict_1=len([i for i in range(0,Y_test.shape[0]) if Y_test[i]==1 and y_pred[i]==1])
    true_predict_2=len([i for i in range(0,Y_test.shape[0]) if Y_test[i]==2 and y_pred[i]==2])
    true_predict_3=len([i for i in range(0,Y_test.shape[0]) if Y_test[i]==3 and y_pred[i]==3])
    confusion_matrix=np.diag([true_predict_1, true_predict_2, true_predict_3])

    st.write("Testing result:",confusion_matrix)

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
st.subheader("K-NN training results visualization")

l=time.time()

X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

fig = plt.figure(figsize=(6,6))

plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue', 'yellow')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('green', 'orange', 'purple'))(i), label = j,marker='.')
    
plt.title('K-NN (training set) using our implementation')
plt.xlabel('Wine')
plt.ylabel('Customer Segment')
plt.legend()

st.pyplot(fig)

r=time.time()
print("Time required for plotting is: "+str(r-l)+" seconds")
# %%
# Visualising the Test set results for our implementation
st.subheader("K-NN testing results visualization")

l=time.time()

X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

fig = plt.figure(figsize=(6,6))

plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue', 'yellow')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('green', 'orange', 'purple'))(i), label = j,marker='.')
    
plt.title('K-NN (test set) using our implementation')
plt.xlabel('Wine')
plt.ylabel('Customer Segment')
plt.legend()

st.pyplot(fig)

r=time.time()
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