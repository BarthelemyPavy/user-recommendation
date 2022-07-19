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

Textual data are a good resource to add information about users or questions, these give us the possibility to find similar questions or users and help the model to handle **cold start case** and **high sparsity**

Reading the paper of LightFM I see this part:

[Section 6.3: Tag embeddings](https://arxiv.org/pdf/1507.08439.pdf)
>In this respect, LightFM is similar to recent word embedding approaches like word2vec and GloVe

Means that to take the best of these data we have to keep the most relevant part of text to qualify a user or a question.

In this part my goal was clear:

- Finding keywords from textual data

To do so I choosed to benchmark 4 solutions:

- [Tf-Idf](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
- [KeyBERT](https://github.com/MaartenGr/KeyBERT)
- [LDA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
- [BERTopic](https://github.com/MaartenGr/BERTopic)

### LDA and BERTopic

These solutions are more for topics discovery accross documents and link the most relevant keywords to it. It's also possible to link a document to its closest topic and by transitivity to keywords. The problem is that require a good setting of number of topics to define:

- If you define too few topics you could have each topic covering a huge scope and assign not accurate keywords to your document

- On the contrary, with too much topics you will have too small corpus to find similiraty inside it.

---
**NOTE**

BERTopic could find the best number of topic alone and give to us an outliers topic with keywords to ignore. It could be interesting to combine these list with KeyBERT | Tf-Idf approach to better filter unrelevant words or keyphrase.

---

### KeyBERT and Tf-Idf

These two seems to better fit with our case as they give you directly most relevant keywords per document.
KeyBERT should be more accurate (and its possible to combine with [KeyphraseVectorizers](https://github.com/TimSchopf/KeyphraseVectorizers) an improved count vectorizer.) as it is using the power of transformers model to better understand the importance of each word or keyphrase. More other we could use pre trained models from [SentenceTransformers](https://www.sbert.net/) trained on siamese network and fitted to perform similarity between texts. But this solution is also more time consumming and could require to have GPU to process a huge dataset.

**NB: Unfortunately I was not the time to test the performance of my model using keywords from KeyBERT but I implemented the solution in my TextProcessingFlow**

--> To show more about my [tests on text processing](../../../notebooks/text_processing.ipynb)

## Recommendation Algorithm

I thought that a pure collaborative filtering model will perform poorly due to the high sparsity of the data and cold start problem.
After some research I found [this algorithm](https://making.lyst.com/lightfm/docs/home.html) used by lyft.

The advantage of this solution is that could be work as a pure Collaborative Filtering algorithm but we could add information about users and items in order to tackle the cold start problem.
It's also a good solution to benchmark Pure Collaborative Filtering VS Hybrid model with the same package.

[Notebook for LightFM train](../../../notebooks/train_recsys.ipynb)

[Notebook for LightFM sandbox](../../../notebooks/sandbox_lightFM.ipynb)


## Results

At the end I benchmarked LightFM model as pure Collaborative Filtering model and as an Hybrid one.

You could find these result [this Notebook](../../../notebooks/train_recsys.ipynb)

Or run running the flows describe [here](execution.md) and see the report generated by Metaflow.
