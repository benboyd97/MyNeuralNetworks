import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from numpy.random import default_rng
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, input_size*3),
            nn.ReLU(),
            nn.Linear(input_size*3, input_size*3),
            nn.ReLU(),
            nn.Linear(input_size*3, output_size)
        )

    def forward(self, x):
        return self.main(x)


class Regressor():

    def __init__(self, x, nb_epoch = 1000, learning_rate=0.01):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.nb_epoch = nb_epoch

        # Preprocess dataset
        X, _ = self._preprocessor(x, training = True)

        self.model = NeuralNetwork(input_size=X.shape[1], output_size=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False, norm_method = 'min_max'):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """


        x=x.fillna(-1) #replace all NaNs with -1

        if training ==True: 
            self.one_hot=preprocessing.LabelBinarizer().fit(x['ocean_proximity']) #fit one-hot encoding
            x['ocean_proximity']=np.argmax(self.one_hot.transform(x['ocean_proximity']),axis=1) #one-hot tranform
            self.norm_x_min=x.min() #save x columnwise normalisation constants
            self.norm_x_max=x.max() 
            self.norm_x_mean=x.mean() 
            self.norm_x_std=x.std() 


        else:
            x['ocean_proximity']=np.argmax(self.one_hot.transform(x['ocean_proximity']),axis=1) #perform one-hot tranform     


        if norm_method=='standard': #if normalisation method is the standardised norm

            x=(x-self.norm_x_mean)/self.norm_x_std #calculate using saved normalisation constants from training


        if norm_method=='min_max': #if normalisation method is min max

            x=(x-self.norm_x_min)/(self.norm_x_max-self.norm_x_min) #calculate using saved normalisation constants from training

        x = x.to_numpy()

        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        return x, y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def fit(self, x, y, batch_size=1, shuffle=True):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True)

        # load numpy array into a data loader
        # transform to torch tensor
        tensor_x = torch.Tensor(X)
        tensor_y = torch.Tensor(Y)

        dataset = TensorDataset(tensor_x, tensor_y)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

        for _ in range(self.nb_epoch):

            self.optimizer.zero_grad()

            for x, y in dataloader:

                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                self.optimizer.step()

        return self.model

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x, pre_proc=True):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if pre_proc:
            X, _ = self._preprocessor(x, training = False)
            tensor_x = torch.Tensor(X)
        else:
            tensor_x = torch.Tensor(x)
        return self.model(tensor_x).detach().numpy()


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y, mse=True):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y, training = False)
        preds = self.predict(X, pre_proc=False)

        if mse:
            return mean_squared_error(Y, preds)
        else:
            return mean_absolute_error(Y, preds)
        

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    train, test = train_test_split(data, test_size=0.2, shuffle=True)
    test, val = train_test_split(test, test_size=0.5)

    # Spliting input and output
    x_train = train.loc[:, data.columns != output_label]
    y_train = train.loc[:, [output_label]]
    x_val = val.loc[:, data.columns != output_label]
    y_val = val.loc[:, [output_label]]
    x_test = test.loc[:, data.columns != output_label]
    y_test = test.loc[:, [output_label]]


    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

