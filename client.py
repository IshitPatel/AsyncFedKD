
import tensorflow as tf
import flwr as fl
from models import *
from helpers import *
import keras
import torch

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

#import tensorflow_addons as tfa
import pandas as pd

# Load the trained teacher model.
#teacher = tf.keras.models.load_model('teacher_final')
#teacher = keras.layers.TFSMLayer('teacher_final', call_endpoint='serving_default')
teacher = tf.keras.applications.MobileNetV3Small(input_shape=(48,48,3), include_top=False, weights='imagenet')

base_learning_rate = 0.001
# Define the student model.
#student = get_model(trainable=True, arch='mobilenetv3small', dense_layers=[64, 32], dropouts=[], name='Student', compiled=True, lr=base_learning_rate)
student = FedKDModel(trainable=True, arch='mobilenetv3small', dense_layers=[64, 32], dropouts=[], name='Student', compiled=True, lr=base_learning_rate).get_model()

teacher.summary()
student.summary()


class MammographyClient(fl.client.NumPyClient):

    def __init__(self, trained_teacher, student, cid=0, epochs=5):
        super(MammographyClient, self).__init__()

        self.trained_teacher = trained_teacher
        self.student = student

        train_data, test_data = get_data(int(cid),'./calcifications.csv')
        #train_data, test_data = get_data('./calcifications_new.csv')


        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs

        self.model = DistillerModel(self.trained_teacher, self.student)
        self.model.compile(tf.keras.optimizers.Adam(learning_rate=base_learning_rate))
        print('model compiled')

    def get_parameters(self, config):
        """
        Returns the model weights.
        """

        return self.model.get_weights()

    def fit(self, parameters, config):
        """
        Start training on the client side.
        """
        print("Training Started")
        self.model.set_weights(parameters)
        history = self.model.fit(self.train_data, epochs=self.epochs)

        return self.model.get_weights(), 2871, {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model.
        """

        self.model.set_weights(parameters)
        # loss, accuracy = self.model.evaluate(self.test_data)
        accuracy, loss = self.model.evaluate(self.test_data)
        print('Eval accuracy:', accuracy)
        print('Eval loss:', loss)

        return loss, 957, {"accuracy": accuracy}


# history = compressed_model.fit(train_data, validation_data=test_data, epochs=2)

# # print(compressed_model.evaluate(train_data))
# print(compressed_model.evaluate(test_data))
# save_history('history_student.json', history)
"""
def client_fn(cid: str) -> MammographyClient:
    ""Create a Flower client representing a single organization.""

    # Load model
    #net = DistillerModel(teacher,student).to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data

    # Create a  single Flower client representing a single organization
    return MammographyClient(teacher,student,cid,epochs=5).to_client()

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=4,  # Never sample less than 10 clients for training
    min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
    min_available_clients=4,  # Wait until all 10 clients are available
)

# Specify the resources each of your clients need. By default, each
# client will be allocated 1x CPU and 0x GPUs
client_resources = {"num_cpus": 1, "num_gpus": 0.0}
if DEVICE.type == "cuda":
    # here we are assigning an entire GPU for each client.
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}
    # Refer to our documentation for more details about Flower Simulations
    # and how to setup these `client_resources`.

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=4,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)
"""

fl.client.start_client(server_address="localhost:8080", client=MammographyClient(teacher, student, epochs=5).to_client())
