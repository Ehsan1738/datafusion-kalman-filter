
#Ehsan Rahimi

import numpy as np
import csv
import matplotlib.pyplot as plt

#Variables for the kalman filter
Tot_sample = 20000
dt = 0.01
H = np.identity(4)
xhat_zero = [1, 0, 0, 0]
p_zero = np.identity(4)
Q = [[10**-4, 0, 0, 0],
     [0, 10**-4, 0, 0],
     [0, 0, 10**-4, 0],
     [0, 0, 0, 10**-4]]

R = [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0,],
     [0, 0, 0, 1]]

rolls, pitches = [], []
rolls_accel, pitches_accel = [], []
rolls_gyro, pitches_gyro = [], []

# at first I change the raw accelerometer data into the eulers angles. 
# this function takes three values and gives the roll and pitch angles.  
def accel_to_eul(x, y, z):
    g = 9.81 
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    pitch = np.arcsin(x / g) 
    roll = np.arcsin(-y / (g * np.cos(pitch)))
    
    return roll, pitch

#in this function i change the euler angles into quaternion to make it easy for the kalman filter, sinds the matrix from euler 
# gives problems when the angle is on the +/- 90 degress. 
def eul_to_quater(roll, pitch):
    return [
        np.cos(roll / 2) * np.cos(pitch / 2) + np.sin(roll / 2) * np.sin(pitch / 2),
        np.sin(roll / 2) * np.cos(pitch / 2) - np.cos(roll / 2) * np.sin(pitch / 2),
        np.cos(roll / 2) * np.sin(pitch / 2) + np.sin(roll / 2) * np.cos(pitch / 2),
        np.cos(roll / 2) * np.cos(pitch / 2) - np.sin(roll / 2) * np.sin(pitch / 2)
    ]
    

#when the filter is done it gives us a quaternion as answer. these matrices should be converted back into euler angels. 
def quater_to_euler(lijst):
    a, b, c, d = np.array(lijst)
    roll = np.arctan2((a**2 - b**2 - c**2 + d**2), 2*(a*b + c*d))
    pitch = np.arcsin(2 * (a * c - d * b))
    return roll, pitch 


# this is the kalman filter that takes data from gyro and accelerometer and gives a prediction of the next angle 
# and the it updates the eror. 
def kalman_filter(p, q, r, x, y, z, xhat_zero, p_zero, Q, R):

    A = H @ ((dt/2) * np.array([[0, -p, -q, -r],
                                [p, 0, r, -q],
                                [q, -r, 0, p],
                                [r, q, -p, 0]]))
    
    xhat_predict = A @ xhat_zero
    xhat_predict /= np.linalg.norm(xhat_predict)
    Pk_predict = np.linalg.inv((A @ p_zero @ A.T) + Q)
    k_gain = Pk_predict @  np.linalg.inv(Pk_predict + R)
    roll, pitch = accel_to_eul(x, y, z)  
    z = eul_to_quater(roll, pitch)
    xhat_new = xhat_predict + k_gain @ (z - xhat_predict)

    xhat_new /= np.linalg.norm(xhat_new)
    PK = Pk_predict - k_gain @ Pk_predict
    #PK /= np.linalg.norm(PK)
    return [xhat_new, PK]


#this is my main loop that reads the csv file. after unpacking the data I use the while loop to calculate the 
#last result filtered by the kalmanfilter. 
with open("Assignment_gyroaccel.csv", mode='r', newline='', encoding='utf-8-sig') as file: 
    data = csv.reader(file, delimiter=";") 
    j = 0 
    p, q, r, x, y, z = [], [], [], [], [], []   
    gyro_roll, gyro_pitch = 0, 0
    next(data)
    while j < Tot_sample:
        row = next(data)  # Read the next row of data
        row = [val.replace(",", ".") for val in row]
        
        # getting the samples from csv
        p.append(float(row[0]))
        q.append(float(row[1]))
        r.append(float(row[2]))
        x.append(float(row[3]))
        y.append(float(row[4]))
        z.append(float(row[5]))

        #calling the kalmanfilter and update the error and position
        result = kalman_filter(p[j], q[j], r[j], x[j], y[j], z[j], xhat_zero, p_zero, Q, R)
        p_zero = result[1]
        xhat_zero = result[0]

        # the result of the filter is set back into euler angles
        angles = quater_to_euler(xhat_zero)
        rolls.append(angles[0])
        pitches.append(angles[1])
        
        #roll and pitch for accelerometer. these are used to showcase the differance. 
        accel_roll, accel_pitch = accel_to_eul(x[j], y[j], z[j])
        rolls_accel.append(accel_roll)
        pitches_accel.append(accel_pitch)

        # roll and pitch for gyroscopen 
        gyro_roll += p[j] * dt
        gyro_pitch += q[j] * dt
        rolls_gyro.append(gyro_roll)
        pitches_gyro.append(gyro_pitch)

        j += 1  # Increment j for the next iteration


# Create stacked plots for Roll and Pitch angles
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot Roll angles from Kalman filter, accelerometer, and gyroscope
axs[0].plot(np.degrees(rolls), label='Roll (Kalman)', color='blue')
#axs[0].plot(np.degrees(rolls_accel), label='Roll (Accelerometer)', color='cyan', linestyle='--')
#axs[0].plot(np.degrees(rolls_gyro), label='Roll (Gyroscope)', color='green', linestyle='-.')
axs[0].set_title('Roll Angle Comparison')
axs[0].set_ylabel('Roll Angle (degrees)')
axs[0].grid()
axs[0].legend()

# Plot Pitch angles from Kalman filter, accelerometer, and gyroscope
axs[1].plot(np.degrees(pitches), label='Pitch (Kalman)', color='orange')
#axs[1].plot(np.degrees(pitches_accel), label='Pitch (Accelerometer)', color='red', linestyle='--')
#axs[1].plot(np.degrees(pitches_gyro), label='Pitch (Gyroscope)', color='purple', linestyle='-.')
axs[1].set_title('Pitch Angle Comparison')
axs[1].set_xlabel('Sample Number')
axs[1].set_ylabel('Pitch Angle (degrees)')
axs[1].grid()
axs[1].legend()

plt.tight_layout()
plt.show()
      
        
        

