#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 
#%%
#create a equation: log(x)
dest_X = np.arange(0.1,10.1,0.2)
dest_y = np.log(dest_X)
plt.plot(dest_X, dest_y)
plt.show()
#%% 
#Create a polynomial equation that degree equals 3
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(dest_X.flatten().reshape(50,1))

#%%
#set up learning rates and the number of iterations
lr_list = [0.0001, 0.001,0.01]
iter_list = [100,50,10]
err_list = []
A_list = []
for iter in iter_list:
    # the optimizer allows us to apply gradients to update variables
    for lr in lr_list:
        optimizer = tf.keras.optimizers.Adam(lr)
        err = []
        #initial x,A,b
        x = tf.convert_to_tensor(X_poly,dtype='float32') #(50,4)
        A = tf.Variable(tf.zeros([4, 1]))
        b = tf.convert_to_tensor(dest_y.reshape(50,1),dtype='float32')
        for step in range(iter):
            #print("Iteration", step)
            with tf.GradientTape() as tape:
                # Calculate A*x
                product = tf.matmul(x, A)
                # calculat the loss value we want to minimize
                # what happens if we don't use the square here?
                difference_sq = tf.math.square(product - b)
                sqe = tf.norm(tf.math.sqrt(difference_sq).numpy())
                err.append(sqe.numpy())
                # print("Squared error:", sqe)
                # calculate the gradient
                grad = tape.gradient(difference_sq, [A])
                # update A
                optimizer.apply_gradients(zip(grad, [A]))
        A_list.append(A.numpy())
        err_list.append(err)
        # Check the final values
        print('learning rate: %.4f, the number of iterations: %d' %(lr, iter))
        #print("Optimized x", x.numpy())
        print("sqe: ", sqe.numpy()) # Should be close to the value of b
#%%
plt.plot(np.arange(100), err_list[0], color = 'red',label='learning rate = 0.0001, iteration = 100')
plt.plot(np.arange(100), err_list[1], color = 'blue',label='learning rate = 0.001, iteration = 100')
plt.plot(np.arange(100), err_list[2], color = 'black',label='learning rate = 0.01, iteration = 100')
plt.xlabel("iteration")
plt.ylabel("error")
plt.legend()
plt.show()
# %%
#Compare the fitted function with sin(x)
fig, ax = plt.subplots(1)

ax.plot(dest_X.flatten().reshape(50,1),(x@A_list[2]).numpy(), label='predicted value')
ax.plot(dest_X.flatten(),dest_y, label='log(x)')

ax.legend()
ax.set_title('Predicted result')
ax.set_xlabel('x')

ax.set_ylabel('y')

plt.show()