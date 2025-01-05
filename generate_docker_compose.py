import argparse

parser = argparse.ArgumentParser(description="Flower Generate Compose")
parser.add_argument(
    "--total_clients", type=int, default=2, help="Number of clients (default 2 clients)"
)

parser.add_argument(
    "--num_rounds", type=int, default=10, help="Number of Rounds"
)

args = parser.parse_args()

def create_docker_compose(args):
    client_configs = [
        {"mem_limit": "3g", "batch_size": 128, "cpus": 2, "learning_rate": 0.001},
        {"mem_limit": "4g", "batch_size": 128, "cpus": 2, "learning_rate": 0.02},
    ]

    content = f"""
services:
    server:
        container_name: server
        build:
            context: .
            dockerfile: Dockerfile
        command: python server.py --num_rounds={args.num_rounds}
        environment:
            FLASK_RUN_PORT: 6000
            DOCKER_HOST_IP: host.docker.internal
        volumes:
            - .:/app
            - /var/run/docker.sock:/var/run/docker.sock
        ports:
            - "6000:6000"
            # - "8265:8265"
            # - "8000:8000"
        stop_signal: SIGINT
    # data-loader:
    #     container_name: data-loader
    #     build:
    #         context: .
    #         dockerfile: Dockerfile
    #     command: python helper/split_dataset.py --num_clients={args.total_clients}
    #     environment:
    #         FLASK_RUN_PORT: 6100
    #         DOCKER_HOST_IP: host.docker.internal
    #     volumes:
    #         - .:/app
    #         - /var/run/docker.sock:/var/run/docker.sock
    #     ports:
    #         - "6100:6100"
    #     stop_signal: SIGINT
"""

    for i in range(1, args.total_clients + 1):
        config = client_configs[(i - 1) % len(client_configs)]
        content += f"""
    client{i}:
        container_name: client{i}
        build:
            context: .
            dockerfile: Dockerfile
        command: python client.py --server_address=server:8080
        deploy:
            resources:
                limits:
                    cpus: "{(config['cpus'])}"
                    memory: "{config['mem_limit']}"
        volumes:
            - .:/app
            - ./data/mnist_splits/client_{i}:/app/data
            - /var/run/docker.sock:/var/run/docker.sock
        ports:
            - "{6000 + i}:{6000 + i}"
        depends_on:
            - server
            # - data-loader
        environment:
            FLASK_RUN_PORT: {6000 + i}
            container_name: client{i}
            DOCKER_HOST_IP: host.docker.internal
        stop_signal: SIGINT  
"""
    content += """
volumes:
    data:
"""

    with open("docker-compose.yml", "w") as f:
        f.write(content)

if __name__ == "__main__":
    create_docker_compose(args)