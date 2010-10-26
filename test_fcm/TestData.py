#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt

small = 5
medium = small*5
large = small*10

## case 1
case1c1f1 = np.random.normal(3,0.5,large)
case1c1f2 = np.random.normal(10,0.5,large)
case1c1 = np.vstack((case1c1f1,case1c1f2)).T

case1c2f1 = np.random.normal(10,1.0,large)
case1c2f2 = np.random.normal(10,0.5,large)
case1c2 = np.vstack((case1c2f1,case1c2f2)).T

case1c3f1 = np.random.normal(10,0.5,small)
case1c3f2 = np.random.normal(3,0.5,small)
case1c3 = np.vstack((case1c3f1,case1c3f2)).T

case1 = np.vstack((case1c1,case1c2))
case1 = np.vstack((case1,case1c3))

case1Labels = np.hstack((np.array([0]).repeat(np.shape(case1c1)[0]),np.array([1]).repeat(np.shape(case1c2)[0])))
case1Labels = np.hstack((case1Labels,np.array([2]).repeat(np.shape(case1c3)[0])))

## case 2 (missing cluster 1)
case2c2f1 = np.random.normal(10,1.0,large)
case2c2f2 = np.random.normal(10,0.5,large)
case2c2 = np.vstack((case2c2f1,case2c2f2)).T

case2c3f1 = np.random.normal(10,0.5,small)
case2c3f2 = np.random.normal(3,0.5,small)
case2c3 = np.vstack((case2c3f1,case2c3f2)).T

case2 = np.vstack((case2c2,case2c3))
#case2Labels = np.hstack((np.array([1]).repeat(np.shape(case2c3)[0]),np.array([2]).repeat(np.shape(case2c3)[0])))
case2Labels = np.hstack((np.array([1]).repeat(np.shape(case2c2)[0]),np.array([2]).repeat(np.shape(case1c3)[0])))

## case 3 (shifted cluster 1)
case3c1f1 = np.random.normal(3,0.5,large)
case3c1f2 = np.random.normal(8.5,0.5,large)
case3c1 = np.vstack((case3c1f1,case3c1f2)).T

case3c2f1 = np.random.normal(10,1.0,large)
case3c2f2 = np.random.normal(10,0.5,large)
case3c2 = np.vstack((case3c2f1,case3c2f2)).T

case3c3f1 = np.random.normal(10,0.5,small)
case3c3f2 = np.random.normal(3,0.5,small)
case3c3 = np.vstack((case3c3f1,case3c3f2)).T

case3 = np.vstack((case3c1,case3c2))
case3 = np.vstack((case3,case3c3))

case3Labels = np.hstack((np.array([0]).repeat(np.shape(case3c1)[0]),np.array([1]).repeat(np.shape(case3c2)[0])))
case3Labels = np.hstack((case3Labels,np.array([2]).repeat(np.shape(case3c3)[0])))

## case 4 (split cluster 2)
case4c1f1 = np.random.normal(3,0.5,large)
case4c1f2 = np.random.normal(10,0.5,large)
case4c1 = np.vstack((case4c1f1,case4c1f2)).T

case4c2f1 = np.random.normal(9,0.25,medium)
case4c2f2 = np.random.normal(10,0.25,medium)
case4c2 = np.vstack((case4c2f1,case4c2f2)).T

case4c4f1 = np.random.normal(11,0.25,medium)
case4c4f2 = np.random.normal(10,0.25,medium)
case4c4 = np.vstack((case4c4f1,case4c4f2)).T

case4c3f1 = np.random.normal(10,0.5,small)
case4c3f2 = np.random.normal(3,0.5,small)
case4c3 = np.vstack((case4c3f1,case4c3f2)).T

case4 = np.vstack((case4c1,case4c2))
case4 = np.vstack((case4,case4c3))
case4 = np.vstack((case4,case4c4))

case4Labels = np.hstack((np.array([0]).repeat(np.shape(case4c1)[0]),np.array([1]).repeat(np.shape(case4c2)[0])))
case4Labels = np.hstack((case4Labels,np.array([2]).repeat(np.shape(case4c3)[0])))
case4Labels = np.hstack((case4Labels,np.array([3]).repeat(np.shape(case4c4)[0])))


if __name__=='__main__':
    
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.scatter(case1[:,0], case1[:,1])
    ax.set_title("Case 1")
    ax.set_xlim([0,14])
    ax.set_ylim([0,14])
    
    ax = fig.add_subplot(222)
    ax.scatter(case2[:,0], case2[:,1])
    ax.set_title("Case 2")
    ax.set_xlim([0,14])
    ax.set_ylim([0,14])

    ax = fig.add_subplot(223)
    ax.scatter(case3[:,0], case3[:,1])
    ax.set_title("Case 3")
    ax.set_xlim([0,14])
    ax.set_ylim([0,14])

    ax = fig.add_subplot(224)
    ax.scatter(case4[:,0], case4[:,1])
    ax.set_title("Case 4")
    ax.set_xlim([0,14])
    ax.set_ylim([0,14])

    plt.show()
