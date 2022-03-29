![Image](/banner.jpg)

# Hierarchical Multilabel Classification

Hierarchical Multi-Level Classification is a classification, where a given input is classified in multiple levels, with a hierarchy amongst them. It is easier to explain it by walking through progressively more complex tasks: binary classification, multi-class classification, multi-level classification, and finally hierarchical multi-level class classification.

Binary Classification
In a binary classification the task is to classify a given input into one of two classes. Spam filters are a good example, which classify every email into "junk" or "not-junk". Many classifiers used in medicine have a similar structure: classify the given input into "disease detected" or "no disease detected". Fraud detection models and credit risk models (whether to extend credit or not) are other examples.

Multi-Class Classification
In multi-class classification, the number of classes are more than two. The Iris dataset (https://archive.ics.uci.edu/ml/datasets/Iris) where based on four features (sepal length, sepal width, petal length and petal width), each flower needs to be classified into one of three classes of Iris plants. Most classification problems are usually multi-class classification problems. Examples includes character recognition models, image classification models, topic or document classification models and segmenting customers into different cohorts.

Single Level vs. Multiple Levels
It is important to distinguish between 'class' and 'level'. The above-mentioned multi-class classification has a single level. In some cases, multiple labels may be attached to a given input. For instance, a given article might be categorized as belonging to 'Sports', 'Afghanistan', and 'Feminism'. These problems are Multi-Level Classification or Multi-Label Classification problems. Another example is detecting objects in a photograph, which could have a 'Dog', 'Cat', and a 'Car'. Sometimes, the number of levels are limited; at other times, the number of levels are fixed. It is also seen that to accommodate an unknown number of levels, a large number of levels are provisioned but many of them could be simply nulls. This is seen in the Brazilian legislation dataset available in this accelerator.

Hierarchy among levels
It is also possible that there is a hierarchy among levels. This is a special case of Multi-Class Classification. Five examples are given in the next section, which make these problems clear.

# Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
