import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import messagebox, simpledialog
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors


def unit_step(x, epsilon=1e-5):# a threshold value added to the unit step for faulty computer float arithmetic
    # => might calculate 0.3 - 0.3 => -1.0e-17 instead of 0.0
    if abs(x) < epsilon:
        x = 0.0
    return 1 if x >= 0 else 0


class PerceptronPassFail:
    def __init__(self, csvFile, split_rate):
        self.tested = False
        self.trained = False
        self.learning_rate = 0.01
        self.epochs = 250
        self.threshold = 200
        self.splitRate = split_rate
        self.df = None
        self.X = None
        self.Yd = None
        self.Ya = None
        self.MSE_history = []
        self.weights = None
        self.X_train = None
        self.X_test = None
        self.Yd_train = None
        self.Yd_test = None
        self.Ya_train = None
        self.Ya_test = None
        self.splitDataFrame(csvFile, split_rate)
        print("weights: ", self.weights)
        print("Xtrain: ", self.X_train)
        print("Xtest: ", self.X_test)
        print("Yd: ", self.Yd)
        print(self.X.shape[0])

    def splitDataFrame(self, csvFile, split_rate):#this function splits the data and initializes the dataframe arrays
        # for learning
        self.df = pd.read_csv(csvFile)
        self.df['pass_fail'] = self.df['pass_fail'].map({'pass': 1, 'fail': 0})
        self.X = self.df[['english', 'math', 'science']].values
        self.Yd = self.df['pass_fail'].values
        self.Ya = [0] * len(self.Yd)
        self.X = np.hstack((self.X, np.ones((self.X.shape[0], 1))))  # Add bias term to X
        split_index = int(self.splitRate * len(self.df))#split the data

        self.X_train = self.df[['english', 'math', 'science']].values[:split_index]
        self.X_test = self.df[['english', 'math', 'science']].values[split_index:]

        self.Yd_train = self.df['pass_fail'].values[:split_index]
        self.Yd_test = self.df['pass_fail'].values[split_index:]

        self.Ya_train = [0] * len(self.Yd_train)
        self.Ya_test = [0] * len(self.Yd_test)

        self.X_train = np.hstack((self.X_train, np.ones((self.X_train.shape[0], 1))))  #add 1 to data to update the threshold
        self.X_test = np.hstack((self.X_test, np.ones((self.X_test.shape[0], 1))))  #add 1 to data to update the threshold

    def train(self, epoch_val, threshold_val, learning_rate_val, goal_val):
        self.trained = True
        self.MSE_history.clear()
        count_goal = 0
        self.splitDataFrame('passfail.csv', split_rate=self.splitRate)
        self.epochs = epoch_val
        self.threshold = threshold_val
        self.learning_rate = learning_rate_val
        #self.weights = np.array([0.1, 0.4, 0.3, -self.threshold])
        random_weights = np.random.uniform(0.3, 0.8, 3)
        self.weights = np.concatenate((random_weights, [-self.threshold]))  #threshold is updated with the weights as a -ve value; this worked better
        epoch = 0
        while True:
            total_error = 0
            for j in range(self.X_train.shape[0]):
                bigX = np.dot(self.X_train[j], self.weights)
                self.Ya_train[j] = unit_step(bigX)
                error = self.Yd_train[j] - self.Ya_train[j]
                total_error += error ** 2
                delta_w = error * self.learning_rate * self.X_train[j]
                self.weights += delta_w

            mse = total_error / self.X_train.shape[0]
            self.MSE_history.append(mse)
            print(f"Epoch {epoch}, MSE: {mse}, Weights: {self.weights}")

            epoch += 1
            if goal_val != 0:
                if mse <= goal_val:
                    count_goal = count_goal + 1
                    if count_goal >= 10:
                        break
            else:
                if epoch >= self.epochs:
                    break

        print("Final weights: ", self.weights)
        print("MSE history: ", self.MSE_history)

    def plotMSE(self):
        #plotting the mean square error to show performance
        if not self.trained:
            messagebox.showerror("Plot Error",
                     "Please train the Perceptron model first.")
            return
        else:
            window_size = 10
            smoothed_MSE = np.convolve(self.MSE_history, np.ones(window_size) / window_size, mode='valid')

            plt.plot(range(len(smoothed_MSE)), smoothed_MSE, linestyle='-', linewidth=0.5, color='b')
            plt.title('Learning Performance')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Square Error')
            plt.grid(True)
            plt.show()
            self.MSE_history.clear()


    def test(self):
        #test the perceptron for the split data
        if not self.trained:
            messagebox.showerror("Test Error",
                     "Please train the Perceptron model first.")
            return
        else:
            self.tested = True
            self.splitDataFrame('passfail.csv', split_rate=self.splitRate)
            correct_predictions = 0
            total_samples = len(self.X_test)

            for i in range(total_samples):
                bigX = np.dot(self.X_test[i], self.weights)
                prediction = unit_step(bigX)
                self.Ya_test[i] = prediction
                if prediction == self.Yd_test[i]:
                    correct_predictions += 1

            return (correct_predictions / total_samples) * 100.0

    def plot(self):
        #this method plots two planes and compares them
        #the first is a plane generated by the learning of the perceptron
        #the second is the ideal plane that separates the data into fail and pass
        #the ideal plane has the formula x + y + z = 180
        #since x + y + x >= 180 is classified as passed and failed otherwise
        #this comes from the average formula, if the students average is above 60, then they pass
        #note that if we were to implement a learning for a case of when all marks are greater than 60 to pass
        #this will need a multi layer perceptron model and that is not the "basic model" referenced in the HOMEWORK
        if not self.trained:
            messagebox.showerror("Plot Error",
                     "Please train the Perceptron model first.")
            return
        else:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection='3d')
            for i in range(self.X_train.shape[0]):
                if self.Yd_train[i] == 1:
                    ax1.scatter(self.X_train[i, 0], self.X_train[i, 1], self.X_train[i, 2], c='b', marker='o')
                else:
                    ax1.scatter(self.X_train[i, 0], self.X_train[i, 1], self.X_train[i, 2], c='r', marker='x')
            x = np.linspace(self.X_train[:, 0].min(), self.X_train[:, 0].max(), 10)
            y = np.linspace(self.X_train[:, 1].min(), self.X_train[:, 1].max(), 10)
            X, Y = np.meshgrid(x, y)
            Z = (-self.weights[3] - self.weights[0] * X - self.weights[1] * Y) / self.weights[2]
            ax1.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100, color='g')
            ax1.set_xlabel('English')
            ax1.set_ylabel('Math')
            ax1.set_zlabel('Science')
            ax1.set_title('Perceptron Decision Boundary')
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, projection='3d')
            for i in range(self.X_train.shape[0]):
                if self.Yd_train[i] == 1:
                    ax2.scatter(self.X_train[i, 0], self.X_train[i, 1], self.X_train[i, 2], c='b', marker='o')
                else:
                    ax2.scatter(self.X_train[i, 0], self.X_train[i, 1], self.X_train[i, 2], c='r', marker='x')
            Z2 = 180 - X - Y
            ax2.plot_surface(X, Y, Z2, alpha=0.5, rstride=100, cstride=100, color='y')
            ax2.set_xlabel('English')
            ax2.set_ylabel('Math')
            ax2.set_zlabel('Science')
            ax2.set_title('Ideal Plane: x + y + z = 180')
            plt.show()

    def test_once(self, englishGrade, mathGrade, scienceGrade):
        #test for single data entry by the user
        if int(englishGrade) > 100 or int(mathGrade) > 100 or int(scienceGrade) > 100:
            messagebox.showerror("Data Entry Error",
                                 "Data must be less than 100.")
            return

        if not self.trained:
            messagebox.showerror("Test Error",
                     "Please train the Perceptron model first.")
            return
        else:
            grades = np.array([int(englishGrade), int(mathGrade), int(scienceGrade), 1])
            bigX = np.dot(grades, self.weights)
            result = unit_step(bigX)
            return "failed" if result == 0 else "passed", (grades[0]+grades[1]+grades[2])/3

    def printReport(self):
        if not self.tested:
            messagebox.showerror("Print Error",
                     "Please test the Perceptron model first.")
            return
        else:
            #create pdf file
            doc = SimpleDocTemplate("report.pdf", pagesize=letter)

            #column names
            data = [["x1", "x2", "x3", "ya_test", "yd_test"]]
            for i in range(len(self.X_test)):
                data.append(
                    [self.X_test[i][0], self.X_test[i][1], self.X_test[i][2], self.Ya_test[i], self.Yd_test[i]])


            table = Table(data)

            #style the table
            style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black)])
            table.setStyle(style)

            elements = []
            elements.append(table)
            doc.build(elements)

