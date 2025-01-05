import argparse
import flwr as fl

from helper.load_data import load_data
from model.model import train, test, set_parameters, get_model

parser = argparse.ArgumentParser(description="Flower client")
parser.add_argument("--server_address", type=str, default="server:8080", help="Server address")
args = parser.parse_args()

net = get_model()
trainloader, testloader = load_data()

class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, trainloader)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
    
def start_fl_client():
    try:
        client = Client().to_client()
        fl.client.start_client(
            server_address=args.server_address,
            client=client,
        )
    except Exception as e:
        print(e)

if __name__ == "__main__":
    start_fl_client()