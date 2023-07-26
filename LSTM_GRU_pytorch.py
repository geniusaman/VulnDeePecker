import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

class LSTMGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, use_lstm=True):
        super(LSTMGRUModel, self).__init__()
        self.use_lstm = use_lstm
        self.hidden_size = hidden_size

        if use_lstm:
            print("* Using PyTorCh-LSTM Network *")
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            print("* Using PyTorCh-GRU Network *")
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 300)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(300, 300)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(300, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        # Output shape: (batch_size, sequence_length, hidden_size)
        rnn_output, _ = self.rnn(x)

        # Take the last hidden state (corresponding to the last time step)
        # Output shape: (batch_size, hidden_size)
        last_hidden_state = rnn_output[:, -1, :]

        x = self.fc1(last_hidden_state)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = self.softmax(x)
        return output

class LSTMGRU:
    def __init__(self, data, name="", batch_size=64, use_lstm=True):
        vectors = np.stack(data["vector"].values)
        labels = data["val"].values

        # Perform a stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, stratify=labels)

    
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.y_train = torch.LongTensor(y_train)  # Changed to use integer labels instead of one-hot encoded
        self.y_test = torch.LongTensor(y_test)
        self.name = name
        self.batch_size = batch_size

        input_size = vectors.shape[2]
        hidden_size = 300
        num_classes = 2
        self.model = LSTMGRUModel(input_size, hidden_size, num_classes, use_lstm=use_lstm)

        # Compute class weights for imbalanced classes
        y_integers = y_train
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
        self.class_weight = torch.FloatTensor(class_weights)

        # Lower learning rate to prevent divergence (equivalent to Adamax with learning rate 0.002)
        self.optimizer = optim.Adamax(self.model.parameters(), lr=0.002)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weight)

    def train(self):
        print("[+] traning process started.............")
        for epoch in range(4):
            print("[+] Epoch", '(',epoch,')',"====================->")
            self.model.train()
            for i in range(0, len(self.X_train), self.batch_size):
                inputs = self.X_train[i:i + self.batch_size]
                targets = self.y_train[i:i + self.batch_size]

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

        torch.save(self.model.state_dict(), self.name + "_model.pt")

    def test(self):
        self.model.load_state_dict(torch.load(self.name + "_model.pt"))
        self.model.eval()

        # Directly evaluate the model on the test data
        with torch.no_grad():
            outputs = self.model(self.X_test)
            _, predictions = torch.max(outputs, dim=1)
            predictions = predictions.cpu().numpy()

        print("Predictions:", predictions)

        tn, fp, fn, tp = confusion_matrix(self.y_test, predictions).ravel()
        print('False positive rate is...', fp / (fp + tn))
        print('False negative rate is...', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('True positive rate is...', recall)
        precision = tp / (tp + fp)
        print('Precision is...', precision)
        print('F1 score is...', (2 * precision * recall) / (precision + recall))

        # Assuming 'predictions' contains the output of the model
        vulnerability_type_predictions = predictions
        vulnerability_location_predictions = predictions  # This is just a placeholder for demonstration purposes

        # ... (rest of the code) ...


        # Assuming the output is (batch_size, 2) where each element contains start and end positions
        print("Shape of vulnerability_location_prediction:", vulnerability_location_predictions.shape)
        print("Content of vulnerability_location_prediction:", vulnerability_location_predictions)

        # Iterate over individual gadgets and extract their vulnerability locations
        vulnerability_locations = []
        for location_prediction in vulnerability_location_predictions:
            # Assuming 'location_prediction' contains a single binary prediction for the gadget
            # Convert binary prediction to integer (0 or 1)
            location_int = int(location_prediction)
            if location_int == 1:
                # If vulnerability exists, append the gadget's location (start, end) to the list
                start_position = 0  # Replace with the actual start position
                end_position = 0  # Replace with the actual end position
                vulnerability_locations.append((start_position, end_position))

        # Print the individual vulnerability location predictions
        # Print the vulnerability predictions
        print("Vulnerability Type Prediction:", vulnerability_type_predictions)
        print("Vulnerability Location Prediction:", vulnerability_location_predictions)
        print("Vulnerability Locations:", vulnerability_locations)



