# Execution


This project use Metaflow, a framework developed by Netflix to help data science project management.

---
**NOTE**

To know more about Metaflow, please check [the documentation](https://docs.metaflow.org/)

---

## Project Flows

This project is organised in "Flow". Each Flow is composed by one or several Tasks.
These Flows are defined in a logically way defining steps of a data science project lifecyle.

> To have more details about flows composing this project please check [this page](./flows.rst)


### Train Test Split

If you want to generate Train, Test and Validation sets for training

Example:

Code:
```
source .venv/bin/activate
python user_recommendation/flows.py run --tag split_dataset
```

### Training Model


If you want to train your model.

---
**NOTE**

Require that Train Test Split flow was run previously.

---

Example:

Code:

```
source .venv/bin/activate
python user_recommendation/flows.py run --tag training
```
