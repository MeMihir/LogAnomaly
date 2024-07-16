# Load and Parse the dataset

from drain3 import TemplateMiner, template_miner_config
from os import path
import time
import logging
import json
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


def load_dataset(file_path):
    """
    Load the dataset from the file
    :param file_path: Path to the dataset file
    :return: List of log messages
    """
    with open(file_path, 'r') as file:
        return file.readlines()

def initialize_drain():
    """
    Initialize the Drain3 model
    :return: Drain3 model
    """
    # Drain3 model
    config = template_miner_config.TemplateMinerConfig()
    config.load(path.join(path.dirname(__file__), 'configs', 'drain3.ini'))
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)
    return template_miner

def parse_dataset(dataset, batch_size=1000, line_count=0, batch_start_time=0):
    """
    Parse the dataset using Drain3 model
    :param template_miner: Drain3 model
    :param dataset: List of log messages
    :return: List of parsed log messages
    """
    start_time = time.time()
    batch_start_time = start_time
    template_miner = initialize_drain()
    results = []
    for line in dataset:
        line = line.rstrip()
        line = line.partition(": ")[2]
        result = template_miner.add_log_message(line)
        params = template_miner.extract_parameters(
            result["template_mined"], line, exact_matching=True)
        
        if params is None:
            print("Parameters extraction failed.")
            params = []
            
        result["parameters"] = params
        results.append(result)
        
        line_count += 1
        if line_count % batch_size == 0:
            time_took = time.time() - batch_start_time
            rate = batch_size / time_took
            logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                        f"{len(template_miner.drain.clusters)} clusters so far.")
            batch_start_time = time.time()
        # if result["change_type"] != "none":
        #     result_json = json.dumps(result)
        #     logger.info(f"Input ({line_count}): {line}")
        #     logger.info(f"Result: {result_json}")

    time_took = time.time() - start_time
    rate = line_count / time_took
    logger.info(f"Total lines: {line_count}, rate {rate:.1f} lines/sec, "
                f"{len(template_miner.drain.clusters)} clusters.")
    return results, template_miner

def log_cluster(cluster):
    """
    Log the cluster
    :param cluster: Cluster
    """
    logger.info(f"Cluster: {cluster.id}, {cluster.size} messages, "
                f"template: {cluster.template}, depth: {cluster.depth}, children: {len(cluster.children)}")
    for child in cluster.children:
        log_cluster(child)

if __name__ == "__main__":
    dataset = load_dataset("loghub/Linux/Linux_2k.log")
    results, template_miner = parse_dataset(dataset)
    # for result in results:
    #     result_json = json.dumps(result)
    #     logger.info(f"Result: {result_json}")
    # logger.info(f"Clusters: {len(template_miner.drain.clusters)}")
    