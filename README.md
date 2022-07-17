# User Recommendation

This project has goal to predict the most likely user to answer a question, given its title & text.
It's based on a dataset from public Stack Overflow posts.

___
## requirements

This project could be run with:
- Your laptop environment
- Docker container
### Docker

Install Docker following official documentation https://docs.docker.com/get-docker/

### Laptop environment

You need to install:
- Python >= 3.9 https://www.python.org/downloads/
- Poetry https://python-poetry.org/docs/#installation
_____

## Installation
After having followed **requirements** section you are ready to install the project.
In order to develop and run this project you must follow these steps.
Check relevant section according to your need, i.e if you use your **laptop environment** or **docker**
### Laptop environment

1. Clone the project on your laptop using:
```
git clone git@github.com:BarthelemyPavy/user-recommendation.git
```
2. Create a virtual environment, to do so make sure to be at the root directory of the project (same level than pyproject.toml) and run:
```
poetry install
source .venv/bin/activate
```
3. Download nltk resources
```
poetry run nltk_resources
```
or
```
./post_install.sh
```
4. Download Stack Overflow data
```
poetry run download_data
```

### Docker
1. Clone the project on your laptop using:
```
git clone git@github.com:BarthelemyPavy/user-recommendation.git
```
2. Build Docker Image
```
docker build -f Dockerfile -t user_recommendation .
```
Or
```
sudo docker build -f Dockerfile -t user_recommendation .
```
Depend on your install.

3. Docker run
```
docker run -it --name user_recommendation -p 8085:8085 --mount type=bind,src=$(pwd),dst=/home/user/user_recommendation/ --entrypoint bash user_recommendation
```
Or
```
sudo docker run -it --name user_recommendation -p 8085:8085 --mount type=bind,src=$(pwd),dst=/home/user/user_recommendation/ --entrypoint bash user_recommendation
```
Now, you should be in a terminal into the container.

4. Install dependencies

Make sure to be at the root directory of the project (same level than pyproject.toml) and run:

```
poetry install
source .venv/bin/activate
```
5. Download nltk resources
```
poetry run nltk_resources
```
or
```
./post_install.sh
```

6. Download Stack Overflow data

```
poetry run download_data
```
___
## Execution

For information about job executions, please check [this page](./doc/source/content/execution.rst)

___
### Code style

We use:
-  [black](https://black.readthedocs.io/en/stable/) as code formatter
- [pylint](https://pylint.pycqa.org/en/latest/) as linter
- [mypy](https://mypy.readthedocs.io/en/stable/) as type checker


Also some shortcuts are available to check your code quality:

```
# Format all your project
poetry run fmt

# Check linter, formatter and typing for all your project
poetry run lint

# Run your unit test
poetry run test
```
