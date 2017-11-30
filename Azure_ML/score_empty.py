# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
import numpy as np
import os

try:
    from azureml.datacollector import ModelDataCollector
except ImportError:
    print("Data collection is currently only supported in docker mode. May be disabled for local mode.")
    # Mocking out model data collector functionality
    class ModelDataCollector(object):
        def nop(*args, **kw): pass
        def __getattr__(self, _): return self.nop
        def __init__(self, *args, **kw): return None
    pass
    

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

model = None

def init():
    pass
    

def run(input_array):
    return int(0)

def generate_api_schema():
    import os
    print("create schema")
    sample_input = np.zeros((1,784))
    inputs = {"input_array":SampleDefinition(DataTypes.NUMPY, sample_input)}
    
    os.makedirs('./outputs', exist_ok=True)
    generate_schema(run_func=run, inputs=inputs, filepath='./outputs/service_schema.json')

# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    # Import the logger only for Workbench runs
    from azureml.logging import get_azureml_logger
    logger = get_azureml_logger()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate Schema')
    args = parser.parse_args()

    if args.generate:
        generate_api_schema()

    init()
    input = np.zeros((1,784))
    result = run(input)
    print(result)
    logger.log("Result",result)
