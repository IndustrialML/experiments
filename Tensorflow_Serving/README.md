# Tensorflow Serving
Keep in mind, that if you want to use a Python client, using the pip package "tensorflow-serving-api", without the need to install Bazel, you will have to use python 2.7 as the pip package is not yet avlaiable to Python 3.x.
Additionally, for Tensorflow Serving to work, you will need to have Tensorflow installed.
## Installation
1. Install the Python Client package and gRPC package
To install Tensorflow Serving dependencies, execute the following:
```bash
	sudo apt-get update && sudo apt-get install -y \
		build-essential \
		curl \
		libcurl3-dev \
		git \
		libfreetype6-dev \
		libpng12-dev \
		libzmq3-dev \
		pkg-config \
		python-dev \
		python-numpy \
		python-pip \
		software-properties-common \
		swig \
		zip \
		zlib1g-dev
```
Afterwards install the Python packages, using `pip`:
```python
	pip install tensorflow-serving-api grpcio
```


2. Install the prebuilt TensorFlow Serving ModelServer binary with:
```bash
	echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list

	curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

	sudo apt-get update && sudo apt-get install tensorflow-model-server
```
Once installed, the binary can be invoked using the command `tensorflow_model_server`.
More details can be found in the official [documentation](https://www.tensorflow.org/serving/setup)

## Usage
1. (*optional:*) retrain the model, executing the `train.py` script. This will add a new model version in a folder under the `export` directory. Notice, that the `tensorflow_model_server` will always host the model in the newest subdirectory, which is the one with the highest version number.
2. Run the server with:
```bash
	tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=home/matze/Documents/TensorflowServe/Tensorflow_Serving/export
```
**Note**: You will need to change the `--model_base_path` path to the export directory, according to your local directory. But the path needs to be an absolute path.

3. Run the client: Execute the `client.py` script, to send a request to the server and get it's prediction. The output should look something like:
```
Extracting ../../Data/MNIST_data/train-images-idx3-ubyte.gz
Extracting ../../Data/MNIST_data/train-labels-idx1-ubyte.gz
Extracting ../../Data/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ../../Data/MNIST_data/t10k-labels-idx1-ubyte.gz
Server response prediction: 4
Correct label is : 4
```

