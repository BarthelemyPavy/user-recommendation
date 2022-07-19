# Algorithm

## Data Exploration

My first step was to take a look to the 3 datasets, some classical check such as:

- An overview of the data
- Type of the columns
- Missing Data
- Distribution of the data
- Quality of textual data

Take a look to **[this notebook](../../../notebooks/data_exploration.ipynb)** to deep dive into my exploration

## Divide and Rule

One of the most important thing training a machine learning model is to be able to evaluate it, to do so we need to keep a subsample of data (unseen during training) to use for evaluation.
Something I like to do is to have a **validation set** to follow the performance of a model during the **training** compared to performance from **training set**. This allow us to see if the model converge, learn and don't overfit.
So I did my choice! I wanted a Train a Test and a Validation (Yep I could be demanding).
But split a dataset for a recommendation use case is not trivial.
1. From my exploration I saw that **cold start** was an important thing to take care in this use case.

Ok good, but what else? Easy.. After Cold start comes **Warm start**!
It's true but a little bit more complicated, to split our data for warm start case we may take care to the temporality, so I choose to use:

2. Leave One Last Item Strategy but maybe more than one!


Less word and more code ? ok -> [click](../../../notebooks/generate_train_test.ipynb)

Reference:
[Exploring Data Splitting Strategies for the Evaluation
of Recommendation Models](https://arxiv.org/pdf/2007.13237.pdf)

## Text Processing

Textual data are a good resource to add information about users or questions, these give us the possibility to find similar questions or users and help the model to handle **cold start case** or **high sparsity**

[Test on text processing](../../../notebooks/text_processing.ipynb)

## Recommendation Algorithm

I thought that a pure collaborative filtering model will perform poorly due to the high sparsity of the data and cold start problem.
After some research I found [this algorithm](https://making.lyst.com/lightfm/docs/home.html) used by lyft.

The advantage of this solution is that could be work as a pure Collaborative Filtering algorithm but we could add information about users and items in order to tackle the cold start problem.
It's also a good solution to benchmark Pure Collaborative Filtering VS Hybrid model with the same package.

[Notebook for LightFM train](../../../notebooks/train_recsys.ipynb)

[Notebook for LightFM sandbox](../../../notebooks/sandbox_lightFM.ipynb)
