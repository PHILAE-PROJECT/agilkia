import random
import sys
import agilkia
import numpy as np
import pandas as pd

from pathlib import Path
from agilkia import TraceSet
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt


class RandomNumberGenerator:
    """
    Generator that generates random number for each event's target column in the traceset.

    Attributes
    ----------
    generate_order : dict
        A dictionary with input columns name as key and type as value in order of generation
    current_index : int
        Specify the index of which column that is being generated
    target_column_name : str
        Name of the column that is being generated
    low : scalar
        The minimum number in the target column
    high : scalar
        The maximum number in the target column

    Methods
    -------
    fit(training)
        Fit the generator to training data
    transform(action_sequence)
        Transform action sequence using fitted generator
    """

    def __init__(self, generate_order, current_index):
        """
            Parameters
            ----------
            generate_order : dict
                A dictionary with input columns name as key and type as value in order of generation
            current_index : int
                Specify the index of which column that is being generated
        """
        self.generate_order = generate_order
        self.current_index = current_index

    def fit(self, training):
        """Fit the generator to training data.

        Store the minimum and maximum number in the target column in the training data.

        Parameters
        ----------
        training : TraceSet
            Data used to train the model
        """
        self.target_column_name = list(self.generate_order.keys())[self.current_index]
        # Specified column's data in Pandas DataFrame
        column_data = training.to_pandas()[self.target_column_name].dropna()
        self.low = column_data.min()
        self.high = column_data.max()

    def transform(self, action_sequence):
        """Transform action sequence using fitted generator.

        Replace the target column in action sequence with generated data.

        Parameters
        ----------
        action_sequence : TraceSet
            Data that will be replaced with generated data
        """
        for tr in action_sequence:
            for ev in tr:
                # TODO: Is this a bug?
                if isinstance(self.low, int) and isinstance(self.low, int):
                    ev.inputs[self.target_column_name] = random.randint(self.low, self.high)
                else:
                    ev.inputs[self.target_column_name] = random.uniform(self.low, self.high)


class RandomCategoryGenerator:
    """
    Generator that generates random Category based on distribution for each event's
    target column in the traceset.

    Attributes
    ----------
    generate_order : dict
        A dictionary with input columns name as key and type as value in oder of generation
    current_index : int
        Specify the index of which column that is being generated
    target_column_name : str
        Name of the column that is being generated
    category : list
        List of distinct category in the target column in the training data
    weights : list
        List of the possibility for each category.

    Methods
    -------
    fit(training)
        Fit the generator to training data
    transform(action_sequence)
        Transform action sequence using fitted generator
    """

    def __init__(self, generate_order, current_index):
        """
        Parameters
        ----------
        generate_order : dict
            A dictionary with input columns name as key and type as value in oder of generation
        current_index : int
            Specify the index of which column that is being generated
        """
        self.generate_order = generate_order
        self.current_index = current_index

    def fit(self, training):
        """Fit the generator to training data.

        Store the distinct categories and probabilities in the target column in the training data

        Parameters
        ----------
        training : TraceSet
            Data used for training the model
        """
        self.target_column_name = list(self.generate_order.keys())[self.current_index]
        # Specified column's data in Pandas DataFrame
        column_data = training.to_pandas()[self.target_column_name].dropna()
        # Get the numbers of samples of each distinct items
        value_count = column_data.value_counts(normalize=True)
        self.category = list(value_count.index.values)
        self.weights = value_count.to_list()

    def transform(self, action_sequence):
        """Transform action sequence using fitted generator.

        Replace the target column in action sequence with generated data.

        Parameters
        ----------
        action_sequence : TraceSet
            Data that will be replaced with generated data
        """
        for tr in action_sequence:
            for ev in tr:
                ev.inputs[self.target_column_name] = random.choices(population=self.category, weights=self.weights)[0]


class SessionGenerator:
    """
    Generator that generates session number with prefix.

    Attributes
    ----------
    generate_order : dict
        A dictionary with input columns name as key and type as value in oder of generation
    current_index : int
        Specify the index of which column that is being generated
    prefix : str
        prefix of the session number ex: "prefix0", "prefix1" ... (default is "session")

    Methods
    -------
    fit(training)
        Fit the generator to training data
    transform(action_sequence)
        Transform action sequence using fitted generator
    """

    def __init__(self, generate_order, current_index, prefix=""):
        """
        Parameters
        ----------
        generate_order : dict
            A dictionary with input columns name as key and type as value in oder of generation
        current_index : int
            Specify the index of which column that is being generated
        prefix : str
            prefix of the session number ex: "prefix0", "prefix1" ... (default is "session")
        """
        self.generate_order = generate_order
        self.current_index = current_index
        self.prefix = prefix

    def fit(self, training):
        """Fit the generator to training data.

        Store the target_column_name.

        Parameters
        ----------
        training : TraceSet
            Not used
        """
        names = list(self.generate_order.keys())
        self.target_column_name = names[self.current_index]

    def transform(self, action_sequence):
        """Transform action sequence using fitted generator.

        Replace the target column in action sequence with generated data.

        Parameters
        ----------
        action_sequence : TraceSet
            Data that will be replaced with generated data
        """
        traceset_count = 0
        for tr in action_sequence:
            for ev in tr:
                ev.inputs[self.target_column_name] = self.prefix + str(traceset_count)
            traceset_count += 1


class NumericalGenerator:
    """
    Generator that generates numerical data using the LinearRegression algorithm.

    Attributes
    ----------
    generate_order : dict
        A dictionary with input columns name as key and type as value in oder of generation
    current_index : int
        Specify the index of which column that is being generated
    decimal : bool
        A boolean indicates if the numerical data in the target column is decimal (default is False)
    metrics : bool
        A boolean indicates whether to show evaluation metrics and graphs
    train_column_names : list
        List contains name of columns that will be used as training inputs
    train_column_types : list
        List contains data type of columns that will be used as training inputs
    encoders : list
        List of encoders used to encode categorical data in training columns
    target_column_name : str
        Name of the column that is being generated
    model : Estimator
        Linear Regression model

    Methods
    -------
    fit(training)
        Fit the generator to training data
    transform(action_sequence)
        Transform action sequence using fitted generator
    """

    def __init__(self, generate_order: dict, current_index, decimal=False, metrics=False):
        """
        Parameters
        ----------
        generate_order : dict
            A dictionary with input columns name as key and type as value in oder of generation
        current_index : int
            Specify the index of which column that is being generated
        decimal : bool
            A boolean indicates if the numerical data in the target column is decimal (default is False)
        metrics : bool
            A boolean indicates whether to show evaluation information
        """
        self.generate_order = generate_order
        self.current_index = current_index
        self.decimal = decimal
        self.metrics = metrics

    def fit(self, training):
        """Fit the generator to training data.

        Encode any categorical data in the training data then fit the Linear
        Regression model with encoded training data.

        Parameters
        ----------
        training : TraceSet
            Data used for training the model
        """
        names = list(self.generate_order.keys())
        types = list(self.generate_order.values())
        self.train_column_names = names[:self.current_index]
        self.train_column_types = types[:self.current_index]
        self.target_column_name = names[self.current_index]
        training_pandas = training.to_pandas().dropna(subset=[self.target_column_name])

        train_x = training_pandas.loc[:, self.train_column_names].fillna("NaN")
        # Encode categorical columns in training inputs
        self.encoders = []
        for i in range(len(self.train_column_names)):
            if self.train_column_types[i] == "categorical":
                encoder = OrdinalEncoder()
                encoder.fit(train_x[self.train_column_names[i]].values.reshape(-1, 1))
                self.encoders.append(encoder)
                train_x[self.train_column_names[i]] = pd.Series(list(encoder.transform(
                    train_x[self.train_column_names[i]].values.reshape(-1, 1))))

        train_y = training_pandas[self.target_column_name]
        if self.metrics:
            train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3)

        reg = LinearRegression()
        self.model = reg.fit(train_x, train_y)
        # Evaluation
        if self.metrics:
            predict = self.model.predict(test_x)
            if not self.decimal:
                predict = np.around(predict, 0)
                predict.astype(int)
            plt.title(self.target_column_name + " column model")
            plt.hist([predict, test_y], label=["Predict", "Real"])
            plt.legend(loc='upper right')
            plt.show()
            print("Coefficient of Determination (R2) :", reg.score(train_x, train_y))

    def transform(self, action_sequence):
        """Transform action sequence using fitted generator.

        Encode any categorical data in the action_sequence then generate data using encoded data.
        Replace the target column in action_sequence with generated data.

        Parameters
        ----------
        action_sequence : TraceSet
            Data that will be replaced with generated data
        """
        actions_pandas = action_sequence.to_pandas()
        predict_x = actions_pandas[self.train_column_names].copy(deep=True)
        # Encode action_sequence columns
        categorical_count = 0
        for i in range(len(self.train_column_names)):
            if self.train_column_types[i] == "categorical":
                predict_x.loc[:, self.train_column_names[i]] = \
                    pd.Series(list(self.encoders[categorical_count].transform(
                        predict_x[self.train_column_names[i]].values.reshape(-1, 1))))
                categorical_count += 1
        row_count = 0
        for tr in action_sequence:
            for ev in tr:
                predict = self.model.predict(predict_x.iloc[[row_count]])
                if self.decimal:
                    ev.inputs[self.target_column_name] = predict[0]
                else:
                    ev.inputs[self.target_column_name] = int(predict[0].round())
                row_count += 1


class CategoricalGenerator:
    """
    Generator that generates categorical data using the DecisionTreeClassifier algorithm.

    Attributes
    ----------
    generate_order : dict
        A dictionary with input columns name as key and type as value in oder of generation
    current_index : int
        Specify the index of which column that is being generated
    metrics : bool
        A boolean indicates whether to show evaluation information
    train_column_names : list
        List contains name of columns that will be used as training inputs
    train_column_types : list
        List contains data type of columns that will be used as training inputs
    encoders : list
        List of encoders used to encode categorical data in training columns
    target_column_name : str
        Name of the column that is being generated
    model : Estimator
        Linear Regression model

    Methods
    -------
    fit(training)
        Fit the generator to training data
    transform(action_sequence)
        Transform action sequence using fitted generator
    """

    def __init__(self, generate_order: dict, current_index, metrics=False):
        """
        Parameters
        ----------
        generate_order : dict
            A dictionary with input columns name as key and type as value in order of generation
        current_index : int
            Specify the index of the column that is being generated
        metrics : bool
            A boolean indicates whether to show evaluation metrics and graphs
        """
        self.generate_order = generate_order
        self.current_index = current_index
        self.metrics = metrics

    def fit(self, training):
        """Fit the generator to training data.

        Encode any categorical data in the training data then fit the
        DecisionTreeClassifier model with encoded training data.

        Parameters
        ----------
        training : TraceSet
            Data used for training the model
        """
        names = list(self.generate_order.keys())
        types = list(self.generate_order.values())
        self.train_column_names = names[:self.current_index + 1]
        self.train_column_types = types[:self.current_index + 1]
        self.target_column_name = names[self.current_index]
        training_pandas = training.to_pandas()

        train_x = training_pandas.loc[:, self.train_column_names[:-1]].fillna("NaN")

        # Add previous output to train_x
        output = training_pandas.loc[:, self.train_column_names[-1]].fillna("NaN")
        output[-1] = "NaN"
        output.index = output.index + 1
        output.sort_index(inplace=True)
        output = output.iloc[:-1]
        train_x[self.target_column_name] = output

        # Encode categorical columns in training inputs
        # all categorical columns are transformed at once, with one OrdinalEncoder.
        self.category_cols = [self.train_column_names[i] 
                            for i in range(self.current_index + 1) 
                            if self.train_column_types[i] == "categorical"]
        # print(train_x.head(20))
        # print("encoding columns:", self.category_cols, "out of", self.train_column_names)
        self.transformer = ColumnTransformer([("categ", OrdinalEncoder(), self.category_cols)], remainder='passthrough')
        train_x = self.transformer.fit_transform(train_x)
        train_y = training_pandas[self.target_column_name].fillna("NaN")
        if self.metrics:
            train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3)

        clf = DecisionTreeClassifier()
        self.model = clf.fit(train_x, train_y)
        # Evaluation
        if self.metrics:
            predict = self.model.predict(test_x)
            predict_category = pd.Series(predict).value_counts()
            plt.title(self.target_column_name + " predict")
            plt.bar(predict_category.index, predict_category.values)
            for index, value in enumerate(predict_category):
                plt.text(index, value, str(value), ha="center")
            plt.show()

            real_category = pd.Series(test_y).value_counts()
            plt.title(self.target_column_name + " real")
            plt.bar(real_category.index, real_category.values)
            for index, value in enumerate(real_category):
                plt.text(index, value, str(value), ha="center")
            plt.show()

            confusion = confusion_matrix(test_y, predict)
            print(confusion)
            print("Accuracy score =", metrics.accuracy_score(test_y, predict))

    def transform(self, action_sequence):
        """Transform action sequence using fitted generator.

        Encode any categorical data in the action_sequence then generate data using encoded data.
        Replace the target column in action_sequence with generated data.

        Parameters
        ----------
        action_sequence : TraceSet
            Data that will be replaced with generated data
        """
        actions_pandas = action_sequence.to_pandas()
        predict_x = actions_pandas.loc[:, self.train_column_names[:-1]]
        predict_x[self.target_column_name] = "NaN"  # add final column for previous predicted value.

        row_count = 0
        previous_predict = "NaN"
        for tr in action_sequence:
            for ev in tr:
                # Add previous predict to inputs
                predict_x.iloc[[row_count], -1] = previous_predict
                inputs = predict_x.iloc[[row_count], :]
                input_vec = self.transformer.transform(inputs)
                predict = self.model.predict(input_vec)
                # Record the predict data
                ev.inputs[self.target_column_name] = predict[0]
                previous_predict = predict[0]
                row_count += 1


def main(args):
    """
    User will need to define some parameters before running the program.
    Parameters that needed to be defined are marked with TODO:.
    """
    random.seed(3)
    np.random.seed(3)
    is_smart = False
    pd.set_option('mode.chained_assignment', None)

    # TODO: Step1. Choose Training and Action Sequence data
    TRAIN = Path("../test/fixtures/1026-steps.json")
    ACTIONS = Path("../test/fixtures/scanner.json")

    training_data = agilkia.TraceSet.load_from_json(TRAIN)
    actions_sequence0 = agilkia.TraceSet.load_from_json(ACTIONS)
    action_sequence = agilkia.TraceSet([])
    action_sequence.set_event_chars(actions_sequence0.get_event_chars())
    for tr in actions_sequence0:
        events = []
        for ev in tr:
            outputs = {"Status": ev.status}
            events.append(agilkia.Event(ev.action, {}, outputs))
        action_sequence.append(agilkia.Trace(events))

    # TODO: Step2. Define Columns Order
    # Inputs dictionary with columns' name and datatype in oder of generation
    # "Action" and "Status"(Output) columns should not be modified
    generate_order = {"Action": "categorical", "Status": "numerical",
                      "sessionID": "categorical",
                      "object": "categorical",
                      "param": "categorical"}

    # TODO: Step3. Choose Generators
    # Estimators in oder of generation
    if is_smart:  # smart
        input_generators = [SessionGenerator(generate_order, current_index=2, prefix="client"),
                            CategoricalGenerator(generate_order, current_index=3),
                            CategoricalGenerator(generate_order, current_index=4)]
    else:  # not-smart
        input_generators = [SessionGenerator(generate_order, current_index=2, prefix="client"),
                            RandomCategoryGenerator(generate_order, current_index=3),
                            RandomCategoryGenerator(generate_order, current_index=4)]

    columns_name = list(generate_order.keys())
    # All generators fit on training data
    for i, generator in enumerate(input_generators):
        print("Fitting on \"" + columns_name[i + 2] + "\" begin.")
        generator.fit(training_data)
        print("Fitting on \"" + columns_name[i + 2] + "\" finished.")

    # All generators transform action sequence
    for i, generator in enumerate(input_generators):
        print("Transforming on \"" + columns_name[i + 2] + "\" begin.")
        generator.transform(action_sequence)
        print("Transforming on \"" + columns_name[i + 2] + "\" finished.")

    if is_smart:
        action_sequence.save_to_json(Path("../test/fixtures/smart_scanner.json"))
    else:
        action_sequence.save_to_json(Path("../test/fixtures/not_smart_scanner.json"))


if __name__ == "__main__":
    main(sys.argv)
