import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd

from perceptron import PerceptronPassFail

import perceptron


class PerceptronGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        #initialize perceptron object
        self.threshold = -200
        self.epoch = 100
        self.learningRate = 0.01
        self.perceptron = PerceptronPassFail('passfail.csv', 0.8)
        self.title("Perceptron GUI")
        self.geometry("300x200")

        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(pady=20)

        # Button 1
        self.trainPerceptronButton = ttk.Button(self.main_frame, text="Train Perceptron",
                                                command=self.showTrainingFrame)
        self.trainPerceptronButton.pack(pady=5)

        # Button 2
        self.button2 = ttk.Button(self.main_frame, text="Test Perceptron", command=self.showTestingFrame)
        self.button2.pack(pady=5)

        # Button 3
        self.button3 = ttk.Button(self.main_frame, text="Enter Data", command=self.showDataFrame)
        self.button3.pack(pady=5)

        # Button 4
        self.button4 = ttk.Button(self.main_frame, text="Print Data Report", command=self.printReport)
        self.button4.pack(pady=5)

    def showTrainingFrame(self):
        trainFrame = tk.Toplevel(self)
        trainFrame.title("Training the Perceptron")
        trainFrame.geometry("400x400")

        self.epoch = tk.IntVar()
        self.learningRate = tk.DoubleVar()
        self.threshold = tk.DoubleVar()

        tk.Label(trainFrame, text="Enter epochs (250 recommended) : ").pack(pady=5)
        tk.Entry(trainFrame, textvariable=self.epoch).pack(pady=5)

        tk.Label(trainFrame, text="Enter learning rate (0.01 recommended) :").pack(pady=5)
        tk.Entry(trainFrame, textvariable=self.learningRate).pack(pady=5)

        tk.Label(trainFrame, text="Enter threshold (200 recommended) :").pack(pady=5)
        tk.Entry(trainFrame, textvariable=self.threshold).pack(pady=5)

        ttk.Button(trainFrame, text="Train", command=self.getReadyToTrain).pack(pady=5)
        ttk.Button(trainFrame, text="Plot Perceptron Boundary and ideal Boundary", command=self.perceptron.plot).pack(
            pady=5)
        ttk.Button(trainFrame, text="Plot Performance", command=self.perceptron.plotMSE).pack(pady=5)

    def getReadyToTrain(self):
        # Check if values have been set by the user
        epoch_val = self.epoch.get()
        learning_rate_val = self.learningRate.get()
        threshold_val = self.threshold.get()

        # Check if any values have been changed before training
        if epoch_val != 0 or threshold_val != 0 or learning_rate_val != 0:
            self.perceptron.train(epoch_val, threshold_val, learning_rate_val)

    def showTestingFrame(self):
        trainFrame = tk.Toplevel(self)
        trainFrame.title("Testing Perceptron")
        trainFrame.geometry("400x400")

        tk.Label(trainFrame, text="Enter English Grade:").pack(pady=5)
        english_entry = tk.Entry(trainFrame)
        english_entry.pack(pady=5)

        tk.Label(trainFrame, text="Enter Math Grade:").pack(pady=5)
        math_entry = tk.Entry(trainFrame)
        math_entry.pack(pady=5)

        tk.Label(trainFrame, text="Enter Science Grade:").pack(pady=5)
        science_entry = tk.Entry(trainFrame)
        science_entry.pack(pady=5)

        def test_using_split_data():
            accuracy = self.perceptron.test()
            accuracy_label.config(text=f"Accuracy of testing is: {accuracy}")

        def test_entered_data():
            result = self.perceptron.test_once(english_entry.get(), math_entry.get(), science_entry.get())
            result_label.config(text=f"Result of Entered Grades is: {result}")

        ttk.Button(trainFrame, text="Test Entered Data", command=test_entered_data).pack(pady=5)
        ttk.Button(trainFrame, text="Test Using Split Data", command=test_using_split_data).pack(pady=5)

        accuracy_label = tk.Label(trainFrame, text="")
        accuracy_label.pack(pady=5)
        result_label = tk.Label(trainFrame, text="")
        result_label.pack(pady=5)

    def add_data_to_csv(self, english, math, science, result):
        # Convert result to 'pass' or 'fail'
        result = 'pass' if result.lower() == 'pass' else 'fail'

        # Create a DataFrame with the new data
        new_data = pd.DataFrame({
            'english': [english],
            'math': [math],
            'science': [science],
            'pass_fail': [result]
        })

        # Append the new data to the CSV file
        with open('passfail.csv', mode='a', newline='') as file:
            new_data.to_csv(file, index=False, header=False)

    def showDataFrame(self):
        dataFrame = tk.Toplevel(self)
        dataFrame.title("Data Frame")
        dataFrame.geometry("400x400")

        tk.Label(dataFrame, text="Add English Grade:").pack(pady=5)
        english_entry = tk.Entry(dataFrame)
        english_entry.pack(pady=5)

        tk.Label(dataFrame, text="Add Math Grade:").pack(pady=5)
        math_entry = tk.Entry(dataFrame)
        math_entry.pack(pady=5)

        tk.Label(dataFrame, text="Add Science Grade:").pack(pady=5)
        science_entry = tk.Entry(dataFrame)
        science_entry.pack(pady=5)

        tk.Label(dataFrame, text="Add result (pass/fail):").pack(pady=5)
        result_entry = tk.Entry(dataFrame)
        result_entry.pack(pady=5)

        def add_data_and_close():
            english = english_entry.get()
            math = math_entry.get()
            science = science_entry.get()
            result = result_entry.get()
            try:
                # Convert entries to integers and validate result
                english = int(english)
                math = int(math)
                science = int(science)
                if result.lower() not in ['pass', 'fail']:
                    raise ValueError("Result must be 'pass' or 'fail'")
                self.add_data_to_csv(english, math, science, result)
                dataFrame.destroy()
            except ValueError as e:
                tk.messagebox.showerror("Input Error", str(e))

        ttk.Button(dataFrame, text="Add data to csv file", command=add_data_and_close).pack(pady=5)

    def printReport(self):
        self.perceptron.printReport()


if __name__ == "__main__":
    app = PerceptronGUI()
    app.mainloop()
