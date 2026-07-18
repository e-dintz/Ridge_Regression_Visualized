# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#%%
"""
How does ridge regression do with polynomial models?

Below, I coded a third degree polynomial and sampled $n$ points from some open 
interval $(a,b)$. I added noise and you can see the difference between the ground 
truth and the data.

The goal of this is to see how ridge regression performs when try to model with 
a polynomial is a higher degree than it needs to be.
"""
# generate some low degree polynomial plus noise
p = lambda x: x**4-5*x**2+4
n = 25
a, b = -2, 2
sigma = 1

# randomly sample in (a,b) and add noise
x  = np.array([random.uniform(a, b) for _ in range(n)])
xi = np.array([random.gauss(0, sigma) for _ in range(n)])
y  = p(x) + xi 
ymin = np.min(np.concat([y,p(x)]))-sigma
ymax = np.max(np.concat([y,p(x)]))+sigma

# plot the polynomial and the noisy data
fig, ax = plt.subplots(1,2,dpi=100,figsize=(8,4))
ax[0].plot(np.linspace(a,b,1000),p(np.linspace(a,b,1000)),c='k')
ax[0].set_ylabel('$p(x)$')
ax[1].scatter(x,y,c='k')
ax[1].set_ylabel('$p(x)+\\xi$')
fig.supxlabel('$x$')
fig.tight_layout()
ax[0].set_ylim(ymin,ymax)
ax[1].set_ylim(ymin,ymax)
plt.show()
#%%
"""
Let's see how ridge regression does on the entire dataset, without a train test 
split. This is just to test to make sure that the code is working so we can make 
an animation later. We can also compare with the least squares estimate ($\alpha=0$).
"""
# initialize the ridge regression model: kth degree polynomial
k = 10
X = np.zeros((len(x),k))
for i in range(1,k):
    X[:,i] = x**i

# calculate least squares
alpha = 0
clf = Ridge(alpha=alpha,fit_intercept=True)
clf.fit(X, y)

# generate prediction on finely sampled data
domain = np.linspace(a,b,1000)
Xdomain = np.zeros((len(domain),k))
for i in range(1,k):
    Xdomain[:,i] = domain**i
yhat = clf.predict(Xdomain)

# plot the data vs. the ridge model and ground truth
plt.figure(dpi=100)
plt.scatter(x,y,c='k',label='$y+\\xi$')
plt.plot(domain,p(domain),label='$p(x)$',c='k')
plt.ylim(ymin,ymax)
plt.plot(np.linspace(a,b,1000),yhat,c='r',label='$\\hat{y}$')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Least Squares, $\\alpha='+str(alpha)+'$')
plt.show()

# calculate ridge regression, no train test split
alpha = 10
clf = Ridge(alpha=alpha,fit_intercept=True)
clf.fit(X, y)

# generate prediction on finely sampled data
domain = np.linspace(a,b,1000)
Xdomain = np.zeros((len(domain),k))
for i in range(1,k):
    Xdomain[:,i] = domain**i
yhat = clf.predict(Xdomain)

# plot the data vs. the ridge model and ground truth
plt.figure(dpi=100)
plt.scatter(x,y,c='k',label='$y+\\xi$')
plt.plot(domain,p(domain),label='$p(x)$',c='k')
plt.ylim(ymin,ymax)
plt.plot(np.linspace(a,b,1000),yhat,c='r',label='$\\hat{y}$')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Ridge Regression, $\\alpha='+str(alpha)+'$')
plt.show()
#%%
"""
Now try a train-test split and calculate the $R^2$. 
"""
# train test split chunk

# initialize the ridge regression model: kth degree polynomial
k = 7
X = np.zeros((len(x),k))
for i in range(1,k):
    X[:,i] = x**i

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=67)

# form ridge regression on the train data
alpha = 1
clf = Ridge(alpha=alpha,fit_intercept=True)
clf.fit(X_train, y_train)


# calculate the train and test accuracy
yhat_train = clf.predict(X_train)
R2_train = r2_score(y_train, yhat_train)
yhat_test = clf.predict(X_test)
R2_test = r2_score(y_test, yhat_test)
print('train score: '+str(np.round(R2_train,4))+'\ntest score: '+str(np.round(R2_test,4)))

# generate prediction on finely sampled data
domain = np.linspace(a,b,1000)
Xdomain = np.zeros((len(domain),k))
for i in range(1,k):
    Xdomain[:,i] = domain**i
yhat = clf.predict(Xdomain)

# plot the data vs. the ridge model and ground truth
plt.figure(dpi=100)
plt.scatter(X_train[:,1],y_train,c='k',label='$y$ train')
plt.scatter(X_test[:,1],y_test,label='$y$ test',facecolors='none', edgecolors='k')
plt.plot(domain,p(domain),label='$p(x)$',c='k')
plt.ylim(ymin,ymax)
plt.plot(np.linspace(a,b,1000),yhat,c='r',label='$\\hat{y}$')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
#%%
"""Usually the test MSE is quite bad if you choose an $\alpha$ hyperparameter 
value that's too low. Let's try computing this for a bunch of $\alpha$s and see 
how it performs.
"""
alphas = np.logspace(-2,2,100)
R2s_train = np.zeros(len(alphas))
R2s_test = np.zeros(len(alphas))
for j in range(len(alphas)):
    # form ridge regression on the train data
    alpha = 1
    clf = Ridge(alpha=alphas[j],fit_intercept=True)
    clf.fit(X_train, y_train)
    
    
    # calculate the train and test accuracy
    yhat_train = clf.predict(X_train)
    R2s_train[j] = r2_score(y_train, yhat_train)
    yhat_test = clf.predict(X_test)
    R2s_test[j] = r2_score(y_test, yhat_test)

plt.figure(dpi=100)
plt.plot(alphas, R2s_train,c='k',label='train $R^2$')
plt.plot(alphas, R2s_test,c='r',label='test $R^2$')
plt.legend()
plt.ylim([0,1])
plt.ylabel('$R^2$')
plt.xlabel('$\\alpha$')
plt.show()
#%%
"""
Let's code here one frame of this animation I want to make: vary $\alpha$, and 
produce a graph that cumulatively plots the train test error on the right and 
shows how the fit is doing on the left.
"""
# generate prediction on finely sampled data
domain = np.linspace(a,b,1000)
Xdomain = np.zeros((len(domain),k))
for i in range(1,k):
    Xdomain[:,i] = domain**i
yhat = clf.predict(Xdomain)

# plot the data vs. the ridge model and ground truth
fig, ax = plt.subplots(1,2,dpi=100,figsize=(10,4))
ax[0].scatter(X_train[:,1],y_train,c='k',label='$y$ train')
ax[0].scatter(X_test[:,1],y_test,label='$y$ test',facecolors='none', edgecolors='k')
ax[0].plot(domain,p(domain),label='$p(x)$',c='k')
ax[0].set_ylim(ymin,ymax)
ax[0].plot(np.linspace(a,b,1000),yhat,c='r',label='$\\hat{y}$')
ax[0].legend(loc='center right', bbox_to_anchor=(-0.15, 0.8)) 
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$y$')
# subplot2: the train test accuracy curves
ax[1].scatter(alphas[0], R2s_train[0],c='k',label='train $R^2$')
ax[1].scatter(alphas[0], R2s_test[0],c='r',label='test $R^2$')
ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.85))
ax[1].set_ylim([-0.1,1.1])
ax[1].set_ylabel('$R^2$')
ax[1].set_xlabel('$\\alpha$')
ax[1].set_xlim(min(alphas)-1,max(alphas))
fig.tight_layout()
#%%
"""
Now for the animation!
"""
# set output dir
output_dir = # your output directory here
savefig = True

# define alphas
alphas = np.logspace(-2,2,100)
R2s_train = np.zeros(len(alphas))
R2s_test = np.zeros(len(alphas))
for j in range(len(alphas)):
    # form ridge regression on the train data
    clf = Ridge(alpha=alphas[j],fit_intercept=True)
    clf.fit(X_train, y_train)
    
    # calculate the train and test accuracy
    yhat_train = clf.predict(X_train)
    R2s_train[j] = r2_score(y_train, yhat_train)
    yhat_test = clf.predict(X_test)
    R2s_test[j] = r2_score(y_test, yhat_test)

    # generate prediction on finely sampled data
    domain = np.linspace(a,b,1000)
    Xdomain = np.zeros((len(domain),k))
    for i in range(1,k):
        Xdomain[:,i] = domain**i
    yhat = clf.predict(Xdomain)
    
    # plot the data vs. the ridge model and ground truth
    fig, ax = plt.subplots(1,2,dpi=100,figsize=(10,4))
    ax[0].scatter(X_train[:,1],y_train,c='k',label='$y$ train')
    ax[0].scatter(X_test[:,1],y_test,label='$y$ test',facecolors='none', edgecolors='k')
    ax[0].plot(domain,p(domain),label='$p(x)$',c='k')
    ax[0].set_ylim(ymin,ymax)
    ax[0].plot(np.linspace(a,b,1000),yhat,c='r',label='$\\hat{y}$')
    ax[0].legend(loc='center right', bbox_to_anchor=(-0.15, 0.8)) 
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$y$')
    # subplot2: the train test accuracy curves
    ax[1].plot(alphas[0:j], R2s_train[0:j],c='k',label='train $R^2$')
    ax[1].plot(alphas[0:j], R2s_test[0:j],c='r',label='test $R^2$')
    ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.85))
    ax[1].set_ylim([-0.1,1.1])
    ax[1].set_ylabel('$R^2$')
    ax[1].set_xlabel('$\\alpha$')
    ax[1].set_xscale('log')
    # ax[1].set_xlim(min(alphas)-1,max(alphas))
    ax[1].set_xlim(min(alphas),max(alphas))
    fig.tight_layout()
    if savefig:
        plt.savefig(output_dir + '/frame'+str(j)+'.png', dpi=200, bbox_inches='tight')
    plt.close()








