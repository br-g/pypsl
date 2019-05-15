# PyPSL

PyPSL is a new library for building PSL models in Python.    
*This is still a prototype.*

![Version](http://img.shields.io/badge/version-0.0.1-blue.svg)     
[![Build status](http://ec2-54-93-95-13.eu-central-1.compute.amazonaws.com/jenkins/buildStatus/icon?job=pypsl%2Fmaster)](http://ec2-54-93-95-13.eu-central-1.compute.amazonaws.com/jenkins/job/pypsl/)     
![Tests coverage](https://s3.eu-central-1.amazonaws.com/pypsl-public/cov_master.svg)

## Probabilistic Soft Logic

> Probabilistic soft logic (PSL) is a machine learning framework for developing probabilistic models. PSL models are easy to use and fast. You can define models using a straightforward logical syntax and solve them with fast convex optimization. PSL has produced state-of-the-art results in many areas spanning natural language processing, social-network analysis, knowledge graphs, recommender system, and computational biology.

To learn more about PSL, see this paper: [Hinge-Loss Markov Random Fields
and Probabilistic Soft Logic](http://www.jmlr.org/papers/volume18/15-631/15-631.pdf).

## Guiding principles

- __User friendliness.__ PyPSL offers a consistent and user-friendly API.

- __Easy extensibility.__ The modules composing the library are simple to extend.

- __Work with Python__. No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.

## Installation

```
git clone git@github.com:br-g/pypsl.git
cd pypsl
make install
```

## Usage example
To get started, please follow these [examples](https://github.com/br-g/pypsl/tree/master/examples).
