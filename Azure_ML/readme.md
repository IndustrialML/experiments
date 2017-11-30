# Getting Started

## Tutorial
To get information on how Azure Machine Learning Services work, please follow these [installation instructions](https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation) and this [tutorial](https://docs.microsoft.com/en-us/azure/machine-learning/preview/tutorial-classifying-iris-part-1).

## Pull example images
If you just want to use my example applications, I hosted the images on my azure docker hub:


Login to my account:
```bash
	docker login mlcrpacr71c0115362c8.azurecr.io -u mlcrpacr71c0115362c8 -p eTC28ULo/KNb9xT2gJGVo7HfUN+p9V+n
```

Pull the images you want to use:
```bash
	docker pull mlcrpacr71c0115362c8.azurecr.io/irisapp:1
	docker pull mlcrpacr71c0115362c8.azurecr.io/mnistapp50:1
```


If you are wondering, where the credential info comes to log into my container registry, you can find these about your own azure container registry with:
```python
	az ml env get-credential -n <your environment name> -r <ressource group name of the env>
```
It will be listed in the "containerRegistry" section. Notice, that this will get the informations about the currently set `env`.

## Run example images
They will launch a REST API on port 5001 per default, but you can define on which port of your machine they should run with `-p`. For example:
```bash
	docker run -d -p 5002:5001 <image-id>
```