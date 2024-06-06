import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import messagebox, simpledialog
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors


def unit_step(x, epsilon=1e-5):
    if abs(x) < epsilon:
        x = 0.0
    return 1 if x >= 0 else 0


class PerceptronPassFail:
    def __init__(self, csvFile, split_rate):
        self.learning_rate = 0.01  # Adjust learning rate if needed
        self.epochs = 250
        self.threshold = 200
        self.splitRate = split_rate
        self.df = None
        self.X = None
        self.Yd = None
        self.Ya = None
        self.weights = None  # Include threshold as part of weights
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

    def splitDataFrame(self, csvFile, split_rate):
        self.df = pd.read_csv(csvFile)
        self.df['pass_fail'] = self.df['pass_fail'].map({'pass': 1, 'fail': 0})
        self.X = self.df[['english', 'math', 'science']].values
        self.Yd = self.df['pass_fail'].values
        self.Ya = [0] * len(self.Yd)
        self.X = np.hstack((self.X, np.ones((self.X.shape[0], 1))))  # Add bias term to X
        # Splitting data into training and testing sets
        split_index = int(self.splitRate * len(self.df))

        self.X_train = self.df[['english', 'math', 'science']].values[:split_index]
        self.X_test = self.df[['english', 'math', 'science']].values[split_index:]

        self.Yd_train = self.df['pass_fail'].values[:split_index]
        self.Yd_test = self.df['pass_fail'].values[split_index:]

        self.Ya_train = [0] * len(self.Yd_train)
        self.Ya_test = [0] * len(self.Yd_test)

        self.X_train = np.hstack((self.X_train, np.ones((self.X_train.shape[0], 1))))  # Add bias term to X
        self.X_test = np.hstack((self.X_test, np.ones((self.X_test.shape[0], 1))))  # Add bias term to X

    def train(self, epoch_val, threshold_val, learning_rate_val):
        self.splitDataFrame('passfail.csv', split_rate=self.splitRate)
        self.epochs = epoch_val
        self.threshold = threshold_val
        self.learning_rate = learning_rate_val
        self.weights = np.array([0.1, 0.4, 0.3, -self.threshold])  # Include threshold as part of weights
        for i in range(self.epochs):
            for j in range(self.X_train.shape[0]):
                bigX = np.dot(self.X_train[j], self.weights)
                self.Ya_train[j] = unit_step(bigX)
                error = self.Yd_train[j] - self.Ya_train[j]
                delta_w = error * self.learning_rate * self.X_train[j]
                self.weights += delta_w
                print(f"Epoch {i}, Sample {j}, Weights: {self.weights}, Error: {error}")

    def test(self):
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
        grades = np.array([int(englishGrade), int(mathGrade), int(scienceGrade), 1])
        bigX = np.dot(grades, self.weights)
        result = unit_step(bigX)
        return "failed" if result == 0 else "passed"

    def printReport(self):
        # Create a PDF document
        doc = SimpleDocTemplate("report.pdf", pagesize=letter)

        # Data for the table
        data = [["x1", "x2", "x3", "ya_train", "yd_train"]]
        for i in range(len(self.X_train)):
            data.append(
                [self.X_train[i][0], self.X_train[i][1], self.X_train[i][2], self.Ya_train[i], self.Yd_train[i]])

        # Create a table
        table = Table(data)

        # Style for the table
        style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)])
        table.setStyle(style)

        # Add the table to the document
        elements = []
        elements.append(table)
        doc.build(elements)

