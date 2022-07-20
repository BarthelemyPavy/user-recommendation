# Next Steps

## Approach

### With LightFM

- Testing textual feature extraction from:
    - KeyBERT
    - KeyBERT and filtering with BERTopic outliers

- Using other textual data:

    - Text of the question to qualify a question
    - Answers of users to fill the missing about me section and qualify a user

- Grid Search to find bests hyperparameters

### With other methods

Try something more state of the art with maybe autoencoders.

[Here](https://github.com/microsoft/recommenders) we could find some ideas



## Code


In globality my code missing checks.

I have to improve my error catching adding more try catch and testing all possibility as I make the choice to give some flexibility to the user.

I added some Unit Tests, but a lot are missing

I could also implement more solution to benchmark through the training flow. It could be text processing or new models.


## Orchestration

Select the best model based on other metrics such as the run time in train and inference stages, used resources (CPU/GPU, RAM).

Add more flexibility on my Flows using foreach operator on training step to have the choice to add more combination of training.

Test Metaflow on a cloud with the UI.
