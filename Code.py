import turtle as trtl
import math
import numpy as np
from keras import models
from keras import layers
import keras
import matplotlib.pyplot as plt

#PARAMETERS#

t = 500

r = 0.5

a1 = 0.3
a2 = 0.2

samples_positions = np.zeros((361,2))
samples_angles = np.zeros((361,2))

#Network Creation

network = models.Sequential()
network.add(layers.Dense(100, activation='relu', input_shape=(2,)))
network.add(layers.Dense(2, activation='linear'))
    
sgd_optimizer = keras.optimizers.SGD(learning_rate=0.01)
adam_optimizer = keras.optimizers.Adam()
    
network.compile(optimizer ='rmsprop',loss='mean_absolute_error',metrics=['mean_absolute_error']) 

#TENSORS PREPARATION
def GetSamples():
    for theta in range(361):
        x = r * math.cos(math.radians(theta))
        y = r * math.sin(math.radians(theta))
        
        atheta2 = math.acos(math.radians((x*x + y*y - a1*a1 - a2*a2)/(2*a1*a2)))
        atheta1 = math.atan2(y,x) - math.atan2((a2*math.sin(atheta2)),(a1+a2*math.cos(atheta2)))
        
        atheta1 = math.degrees(atheta1) / 361
        atheta2 = math.degrees(atheta2) / 361
            
        samples_positions[theta] = np.array([x,y])
        samples_angles[theta] = np.array([atheta1, atheta2])    

def GetTensors():
    global train_positions
    global train_angles
    global test_positions
    global test_angles
    global validation_positions
    global validation_angles

    random_indexes = np.random.choice(361, size=110, replace=False)
      
    train_positions = np.delete(samples_positions, random_indexes,axis=0)
    train_angles = np.delete(samples_angles, random_indexes,axis=0)
    
    test_positions = samples_positions[random_indexes]
    test_angles = samples_angles[random_indexes]
    
    validation_positions = test_positions[0:10]
    validation_angles = test_angles[0:10]
    
    test_positions = test_positions[10:]
    test_angles = test_angles[10:]

# Standarization (increased the error)
#mean = np.divide(np.sum(train_positions, axis=1), 110)
#mean = mean.reshape((251,1))
#mean = np.broadcast_to(mean, (251,2))

#train_positions = np.subtract(train_positions, mean)

#sigma_sqrd = np.divide(np.sum(np.power(train_positions,2), axis=1), 110)
#sigma_sqrd = sigma_sqrd.reshape((251,1))
#sigma_sqrd = np.broadcast_to(sigma_sqrd, (251,2))

#train_positions = np.divide(train_positions, sigma_sqrd)

#DRAWING
def DrawArms(index, tensor):
    trtl.speed(0)
    trtl.goto(0,0)
    trtl.setheading(0)
    trtl.left(tensor[index][0] * 360)
    trtl.forward(t*r*a1)
    trtl.left(tensor[index][1] * 360)
    trtl.forward(t*r*a2)
    trtl.Screen().exitonclick()
    
def DrawTest(tensor, step):
    for x in range(0, tensor.shape[0], step):
    	DrawArms(x, tensor)
    

    
def TrainNetwork():
    history = network.fit(train_positions, train_angles, epochs=500, batch_size=1)
    
    print('Mean absolute error:', history.history['mean_absolute_error'][-1])
    
    plt.plot(history.history['loss'])
    plt.title('loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    
    #evaluating
    test_loss = network.evaluate(test_positions, test_angles)
    test_loss = test_loss[0]
    print('test loss:', test_loss)
    
    global predictions_angles
    
    #predicting
    predictions_angles = network.predict(validation_positions)


#DRAWING2 (without multiplying by 360)
def DrawArms2(index, tensor):
    trtl.speed(0)
    trtl.goto(0,0)
    trtl.setheading(0)
    trtl.left(tensor[index][0])
    trtl.forward(t*r*a1)
    trtl.left(tensor[index][1])
    trtl.forward(t*r*a2)
    trtl.Screen().exitonclick()
    
def DrawTest2(tensor, step):
    for x in range(0, tensor.shape[0], step):
    	DrawArms2(x, tensor)
    
    
def GetAnglesByPos(pos):
    x = pos[0]
    y = pos[1]
    atheta2 = math.acos(math.radians((x*x + y*y - a1*a1 - a2*a2)/(2*a1*a2)))
    atheta1 = math.atan2(y,x) - math.atan2((a2*math.sin(atheta2)),(a1+a2*math.cos(atheta2)))
    atheta1 = math.degrees(atheta1)
    atheta2 = math.degrees(atheta2)
    return (atheta1, atheta2)

def DrawPrediction(index):
    DrawTest(np.array([predictions_angles[index]]), 1)
    
    
def DrawReal(index):    
    true_angle = GetAnglesByPos(validation_positions[index])
    true_angle = np.array([true_angle])
        
    DrawTest2(true_angle, 1)

#Training on a circle (radius = 1)
r = 0.5
GetSamples()
GetTensors()
TrainNetwork()
#DrawPrediction(0)
#DrawReal(0)

#TRAINING WITH RADIUS = 0.1
r = 0.1
GetSamples()
GetTensors()
TrainNetwork()
#DrawPrediction(0)
#DrawReal(0)

#TRAINING WITH RADIUS = 0.25
r = 0.25
GetSamples()
GetTensors()
TrainNetwork()
#DrawPrediction(0)
#DrawReal(0)

#Testing on the whole space
random_positions = np.random.rand(50,2)
random_angles = np.zeros((50,2))

for i in range(50):
    random_angles[i] = GetAnglesByPos(random_positions[i])
    
random_angles = np.divide(random_angles, 360)

#evaluating
test_loss = network.evaluate(random_positions, random_angles)

plt.plot(test_loss)
plt.title('loss per sample')
plt.ylabel('loss')
plt.xlabel('samples')
plt.show()
