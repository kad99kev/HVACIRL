import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from gymnasium.wrappers.normalize import RunningMeanStd
import torchvision.transforms as transforms
from tqdm import tqdm
from termcolor import colored

def prepare_loaders(X, y, batch_size, seed):
    """
    Prepare dataloaders for imitation learning.
    """

    normaliser = ImitationNormalisation(obs_shape=X.shape[1])
    # Ensure that all actions are in the range.
    # There is nothing learnt here, the ranges are known.
    # So we can normalise them together.
    y = normaliser.transform_actions(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=seed # shuffle = True important
    )

    # Perform fit and transform here so that
    # we do not need to perform it during
    # training and testing iterations.
    X_train = normaliser.fit(X_train)
    X_test = normaliser.transform(X_test)

    # Convert to tensors.
    X_train = torch.tensor(X_train.values.astype(np.float32))
    y_train = torch.tensor(y_train.values.astype(np.float32))
    X_test = torch.tensor(X_test.values.astype(np.float32))
    y_test = torch.tensor(y_test.values.astype(np.float32))

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    return normaliser, (train_loader, test_loader)


class ImitationNormalisation:
    def __init__(self, obs_shape, epsilon: float = 1e-8):
        """
        Performs normalisation via RunningMeanStd from gymnaisum.
        """

        self._update_running_mean = True
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.epsilon = epsilon

    def update_running_mean(self, setting):
        """Whether to update running mean stats."""
        self._update_running_mean = setting

    def step(self, obs):
        """Steps through the environment and normalizes the observation."""
        # Normalize observation and return
        return self.normalize(np.array([obs]))[0]

    def normalize(self, obs):
        """Normalizes the observation using the running mean and variance of the observations."""
        if self._update_running_mean:
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
    
    def fit(self, X):
        """Fit to save the RMS stats."""
        # Initialize an empty DataFrame to store normalized values
        normalized_df = pd.DataFrame(columns=X.columns, dtype=np.float64)

        # Normalize each row and store the normalized values in the new DataFrame
        print(colored("Fitting onto data...", "light_blue"))
        for index, row in tqdm(X.iterrows(), total=X.shape[0]):
            normalized_row = self.step(row.values)  # Assuming step method returns the normalized row
            normalized_df.loc[index] = normalized_row

        print(colored("Fitting complete!", "light_blue"))
        return normalized_df
    
    def transform(self, X):
        """Fit to save the RMS stats."""
        # Initialize an empty DataFrame to store normalized values
        normalized_df = pd.DataFrame(columns=X.columns, dtype=np.float64)

        # Set updates to false.
        org_update_running_mean = self._update_running_mean
        self.update_running_mean(False)

        # Normalize each row and store the normalized values in the new DataFrame
        print(colored("Transforming data...", "light_blue"))
        for index, row in tqdm(X.iterrows(), total=X.shape[0]):
            normalized_row = self.step(row.values)  # Assuming step method returns the normalized row
            normalized_df.loc[index] = normalized_row
        print(colored("Transform complete!", "light_blue"))

        # Reset to original value.
        self.update_running_mean(org_update_running_mean)
        return normalized_df
    
    def transform_actions(self, y, setpoint_range=None):
        """Transform actions between the defined setpoint ranges."""
        if setpoint_range is None:
            setpoint_range = {
                "Heating_Setpoint_RL": [15.0, 23.0],
                "Cooling_Setpoint_RL": [23.0, 30.0],
            }

        normalized_df = pd.DataFrame(columns=y.columns, dtype=np.float64)
        for column in y.columns:
            # Get data.
            values = y[column]
            v_range = setpoint_range[column]
            # Transform data.
            values = (values - v_range[0]) / (v_range[1] - v_range[0])
            values = 2 * values - 1
            # Ensure that all values like within [-1, 1] for IL/RL training.
            normalized_df.loc[:, column] = np.clip(values, -1, 1)
        return normalized_df


    def save_rms(self, save_path):
        """Save the RMS stats into a pickle file."""
        with open(save_path, "wb") as file_handler:
            pickle.dump(self.obs_rms, file_handler)
