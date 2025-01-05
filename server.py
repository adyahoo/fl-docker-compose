import argparse
import flwr as fl

parser = argparse.ArgumentParser(description="Flower server")
parser.add_argument("--num_rounds", type=int, default=5, help="Number of rounds")
args = parser.parse_args()

def weighted_average(metrics):
    """Multiply accuracy of each client by number of examples used"""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = sum(num_examples for num_examples, _ in metrics)

    """ Aggregate and return the weighted average """
    return {"ini accuracy": sum(accuracies) / examples}

config = fl.server.ServerConfig(num_rounds=args.num_rounds)

strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average
    )

def start_fl_server():
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=config,
            strategy=strategy,
        )
    except Exception as e:
        print(e)

if __name__ == "__main__":
    start_fl_server()
