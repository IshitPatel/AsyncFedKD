
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from time import time
import tensorflow as tf
from models import *
from helpers import *
import flwr as fl
import random

base_learning_rate = 0.001

student = FedKDModel(trainable=True, arch='mobilenetv3small', dense_layers=[64, 32], dropouts=[], name='Student', compiled=True, lr=base_learning_rate)

class WaitClientManager(SimpleClientManager):

    def __init__(self, wait_time_after_client=60) -> None:
        """
        When the server receives the trained weights then it starts the timer
        and waits for the `wait_time_after_client` seconds and then aggregates the weights.
        """

        super().__init__()
        self.wait_time_after_client = wait_time_after_client
        self.start_time = None
        self.processed = set()

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Returns:
            bool: Indicating if registration was successful. False if ClientProxy is
                already registered or can not be registered for any reason
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client

        if self.start_time is None:
            self.start_time = time()

        with self._cv:
            self._cv.notify_all()

        return True

    def diff(self, ct):

        if self.start_time is not None:
            diff = ct - self.start_time

            print(diff)
            print(self.clients)
            print(self.wait_time_after_client)
        else:
            print('start_time is none')
        return True

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Block until at least `num_clients` are available or until a timeout
        is reached.

        Current timeout default: 1 day.
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: self.diff(time()) and len(self.clients) >= 1 and self.start_time is not None and (time() - self.start_time) >= self.wait_time_after_client, timeout=timeout
            )

    def sample(
        self,
        num_clients: int,
        min_num_clients = None,
        criterion = None,
    ):
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients

        with self._cv:
            while True:
                print('Waiting', len(self.clients))

                if len(self.clients) > 0:
                    break

                self._cv.wait()

        # self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        num_clients = len(self.clients)
        # if criterion is not None:
        #     available_cids = [
        #         cid for cid in available_cids if criterion.select(self.clients[cid])
        #     ]

        # if num_clients > len(available_cids):
        #     log(
        #         INFO,
        #         "Sampling failed: number of available clients"
        #         " (%s) is less than number of requested clients (%s).",
        #         len(available_cids),
        #         num_clients,
        #     )
        #     return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]


class CustomStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, rnd, results, failures):

        aggregate_weights = super().aggregate_fit(rnd, results, failures)


        if aggregate_weights is not None:

            # print(aggregate_weights)
            # print('Aggregated Weights:')
            print(self.evaluate(2, aggregate_weights))

        return aggregate_weights

def get_evaluate_fn(distilled):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    train_data, test_data = get_data('../calcifications_new.csv')

    # The `evaluate` function will be called after every round
    def evaluate(current_round, weights, config):
        distilled.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = distilled.evaluate(test_data)
        return loss, {"accuracy": accuracy}

    return evaluate

if __name__ == '__main__':

    compressed = DistillerModel(None, student)
    compressed.compile(tf.keras.optimizers.Adam(learning_rate=base_learning_rate))

    history = fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=5),
        # strategy=CustomStrategy(evaluate_fn=get_evaluate_fn(compressed)),
        strategy=CustomStrategy(),
        client_manager=WaitClientManager(wait_time_after_client=10)
    )

    print(history)

    # Starts the server and continues the training for 20 rounds.
    # fl.server.start_server(
    #     config=fl.server.ServerConfig(num_rounds=10),
    #     strategy=CustomStrategy(min_fit_clients=2, min_evaluate_clients=1, min_available_clients=1)
    # )
