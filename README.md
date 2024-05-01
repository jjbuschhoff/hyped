# :boom: Hyped

[![Tests](https://github.com/open-hyped/hyped/actions/workflows/tests.yml/badge.svg?branch=hyped-v2)](https://github.com/open-hyped/hyped/actions/workflows/tests.yml)
[![Linting](https://github.com/open-hyped/hyped/actions/workflows/linting.yml/badge.svg?branch=hyped-v2)](https://github.com/open-hyped/hyped/actions/workflows/linting.yml)
[![Coverage Status](https://coveralls.io/repos/github/open-hyped/hyped/badge.svg?branch=hyped-v2)](https://coveralls.io/github/open-hyped/hyped?branch=hyped-v2)
[![PyPi version](https://badgen.net/pypi/v/hyped/)](https://pypi.org/project/hyped)
[![PyPi license](https://badgen.net/pypi/license/hyped/)](https://pypi.org/project/hyped/)

Hyped is a versatile framework built on top of Hugging Face Datasets, designed to simplify the management and execution of data pipelines. With Hyped, you can define data pipelines as sequences of data processors, leveraging the rich ecosystem of Hugging Face datasets while also providing the flexibility to implement custom processors when needed.

## Features

- **Seamless Integration with Hugging Face Datasets**: Utilize the extensive collection of datasets available through [HuggingFace](https://huggingface.co/docs/datasets/en/index) with ease. Hyped handles data loading and preprocessing using HuggingFace's powerful tools.
- **Flexible Data Processing**: Define complex data processing workflows using a sequence of data processors. Hyped comes with a set of general-purpose processors out of the box, allowing for a wide range of transformations and manipulations on your data.
- **Configurable Data Processors**: Each data processor in Hyped is fully configurable, allowing users to fine-tune their behavior according to specific requirements. This flexibility enables users to customize data processing workflows and adapt them to different use cases seamlessly.
- **Custom Processor Support**: Implement custom data processors tailored to your specific requirements. Whether you need to apply domain-specific transformations or integrate with external libraries, Hyped provides the flexibility to extend its functionality as needed.
- **Efficient Execution**: Execute your data pipelines efficiently, whether you're working with small datasets or processing large volumes of data. Hyped supports multiprocessing and data streaming out of the box, enabling efficient utilization of computational resources and avoiding memory limitations when processing large datasets.
- **Scalability**: Hyped provides scalability to handle diverse workload demands, allowing you to seamlessly scale your data processing tasks as needed. Whether you're processing small datasets on a single machine or dealing with large volumes of data across distributed computing environments, Hyped adapts to your workload requirements, ensuring efficient execution and resource utilization.

## Getting Started

Get up and running with Hyped in no time! Follow these simple steps to install the framework and start defining and executing your data pipelines effortlessly.

For detailed documentation, please refer to the [Hyped Documentation](https://open-hyped.github.io/hyped/index.html)

### Installation

Hyped is available on PyPI and can be installed using pip:

```bash
pip install hyped
```

Alternatively, you can install Hyped directly from the source code repository:

```bash
# Clone the Hyped repository from GitHub
git clone https://github.com/open-hyped/hyped.git

# Navigate to the cloned repository
cd hyped

# Install the package including optional developer dependencies
pip install -e .[linting, tests]
```

Now you're ready to start using Hyped for managing and executing your data pipelines!

### Usage

Start by importing the necessary modules and classes:
```python
import datasets
from hyped.data.pipe import DataPipe
from hyped.data.processors.tokenizers.hf import (
    HuggingFaceTokenizer,
    HuggingFaceTokenizerConfig
)
```

Next, load your dataset using the datasets library. In this example, we load the IMDb dataset:

```python
ds = datasets.load_dataset("imdb")
```

Then, define your data pipeline using the `DataPipe` class from Hyped. Add data processors to the pipeline to specify the desired data transformations. For instance, the following code applies a HuggingFace tokenizer to tokenize the text feature of the dataset using the BERT tokenizer:

```python
pipe = DataPipe([
    HuggingFaceTokenizer(
        HuggingFaceTokenizerConfig(
            tokenizer="bert-base-uncased",
            text="text"
        )
    )
])
```

Finally, apply the data pipeline to your dataset using the apply method:

```python
ds = pipe.apply(ds)
```

Now, your dataset has been processed according to the defined pipeline, and you can proceed with further analysis or downstream tasks in your application.

For more examples and advanced usage scenarios, check out the [Hyped examples](https://github.com/open-hyped/examples) repository.

## Configuration

Hyped provides various configuration options that allow users to customize the behavior of the framework. Below are some of the key configuration options and how you can use them:

### 1. Processor Configuration

Each data processor in Hyped can be configured with specific parameters to tailor its behavior. For example, when using the HuggingFaceTokenizer, you can specify the tokenizer model to use, the maximum sequence length, and other tokenizer-specific settings.

```python
tokenizer_config = HuggingFaceTokenizerConfig(
    tokenizer="bert-base-uncased",
    max_length=128,
    padding=True,
    truncation=True
)
```

### 2. Multiprocessing and Batch Processing

Hyped supports data parallel multiprocessing to utilize multiple CPU cores for faster data processing. You can configure the number of processes to use and other multiprocessing options based on your system's specifications. Additionally, batch processing allows you to process data in batches, which can further improve performance and memory efficiency.

```python
ds = pipe.apply(ds, num_proc=4, batch_size=32)
```

### 3. Data Streaming

Hyped supports streaming data directly from and to disk, enabling efficient processing of large datasets that may not fit into memory. You can stream datasets using lazy processing, where examples are only processed when accessed.

```python
from hyped.data.io.writers.json import JsonDatasetWriter

# Load dataset with streaming enabled
ds = datasets.load_dataset("imdb", split="train", streaming=True)

# Apply data pipeline (lazy processing for streamed datasets)
ds = pipe.apply(ds)

# Write processed examples to disk using 4 worker processes
JsonDatasetWriter("dump/", num_proc=4).consume(ds)
```

## Running Tests

Hyped includes a suite of tests to ensure its functionality. You can run these tests using pytest:

```bash
pytest tests
```

Ensure that you have pytest installed in your environment. You can install it via pip:

```bash
pip install pytest
```

Running the tests will execute various test cases to validate the behavior of Hyped.

## Contribution Guidelines

We welcome contributions from the community to help improve and expand Hyped. Before contributing, please review our [Contribution Guidelines](/CONTRIBUTING.md) for instructions on reporting bugs, suggesting features, and submitting pull requests.

## License

Hyped is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See the [`LICENSE`](/LICENSE) file for details.
