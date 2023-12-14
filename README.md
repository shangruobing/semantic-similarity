# Semantic Similarity

This repository contains an example of how to fine-tune a pretrained BERT model for semantic similarity.

## Background

### Cross Model

Cross-models are commonly employed in text classification tasks where it is necessary to determine whether two input text segments belong to the same category. In this model structure, after encoding two text segments, they are cross-combined and then subjected to subsequent classification predictions. Cross-models are relatively common in classification tasks and typically only require a single BERT encoder.

![](doc/Cross.png)

### Siamese Model

Dual-tower Siamese models are commonly used in text similarity tasks, such as question answering and retrieval. This model structure consists of two identical towers, each of which is a BERT encoder. They encode two text segments separately. The encoded text representations are then compared using distance measurement methods (such as cosine similarity) to calculate the similarity between the texts.

![](doc/Siamese.png)

## Baseline Model

You can download baseline model from [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese).

