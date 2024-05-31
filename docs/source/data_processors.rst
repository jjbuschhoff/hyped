Data Processors
===============

Data processors are a fundamental component of the data flow architecture, serving as one of the three primary types of nodes within a data flow. Specifically, a data processor nodes implement data transformations, such as tokenization, normalization, or any other feature enrichment operations. Data processors focus on transforming individual examples by taking input features, applying the specified transformation, and producing enriched output features.

:code:`Hyped` comes equipped with a wide range of pre-built, general-purpose data processors designed to handle various types of data and tasks. These processors cover different modalities, including text, image, audio, and more, offering a comprehensive toolkit for data preprocessing and feature engineering. These pre-built processors provide robust solutions for common data processing challenges, enabling users to quickly and effectively prepare their data for downstream tasks.

Design Principles
-----------------

- **Modularity**: Data processors should be designed as modular components that can be easily integrated into various data flows. Each processor should focus on a specific transformation task, making it reusable across different workflows.
- **Isolation**: Ensure that data processors operate on individual examples independently, avoiding dependencies on other examples in the dataset. This promotes parallelism and simplifies the processing logic.
- **Configurable**: Data processors should be highly configurable, allowing users to customize their behavior through well-defined configuration objects. This enhances flexibility and adaptability to different tasks.
- **Clear and Type-Safe Input/Output Definitions**: Define the required input keys and expected output features clearly while enforcing type safety. This helps in catching errors early, maintaining consistency throughout the data flow, and ensuring that data processors can be easily integrated and composed within a data flow.


Working with Data Processors
----------------------------

This section covers essential aspects such as configuring data processors and invoking them through the call method. Understanding these functionalities is crucial for effective data processing workflows.

Configuring Data Processors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data processors in the Hyped library are highly configurable, allowing users to tailor their behavior to specific tasks or requirements. This section explores the various configuration options available for data processors and provides examples to illustrate their usage.

**Configuration Options**

Each data processor comes with a set of configuration options that define its behavior. These options are encapsulated within a :class:`Pydantic` model, providing a structured and type-safe way to specify the processor's parameters. Common configuration options may include:

- **Model Selection**: Specify the pre-trained model or algorithm to be used by the processor.
- **Parameter Settings**: Define parameters such as maximum sequence length, padding strategy, truncation strategy, etc., depending on the processor's functionality.
- **Other Customization Options**: Additional options specific to each processor, allowing fine-grained control over its behavior.

**Example: Configuring a Transformers Tokenizer**

Let's consider the example of configuring a Transformers Tokenizer data processor. This processor is responsible for tokenizing text inputs using pre-trained models from the Hugging Face Transformers library.

.. code-block:: python

    from hyped.data.flow.processors.tokenizers.transformers import (
        TransformersTokenizer, TransformersTokenizerConfig
    )

    # Define the configuration for the Transformers Tokenizer
    tokenizer_config = TransformersTokenizerConfig(
        tokenizer="bert-base-uncased",  # Specify the pre-trained model to use
        max_length=128,                 # Maximum sequence length for tokenization
        padding="max_length",           # Padding strategy for sequences
        truncation=True                 # Truncation strategy for sequences
    )

    # create the tokenizer processor instance
    tokenizer = TransformersTokenizer(tokenizer_config)

In this example, we define a configuration for the :class:`TransformersTokenizer`, specifying parameters such as the pre-trained model to use (:code:`tokenizer`), the maximum sequence length for tokenization (:code:`max_length`), the padding strategy for sequences (:code:`padding`), and the truncation strategy for sequences (:code:`truncation`). These parameters can be adjusted based on the specific requirements of the tokenization task.

Alternatively, users can initialize data processors using keyword arguments to override specific configuration values:

.. code-block:: python

    tokenizer = TransformersTokenizer(
        tokenizer_config,
        tokenizer="roberta-base"  # Override the pre-trained model to use
    )

In this example, the :class:`TransformersTokenizer` is initialized with a configuration instance :code:`tokenizer_config`. However, we use a keyword argument to specify a different pre-trained model (:code:`roberta-base`) compared to the one defined in the configuration. This approach allows for flexible customization of data processors without modifying the original configuration instance.

It's also possible to initialize data processors directly with keyword arguments, without using a configuration class:

.. code-block:: python

    tokenizer = TransformersTokenizer(
        tokenizer="roberta-base",
        max_length=256,
        padding="longest",
        truncation=True
    )

In this example, we directly specify the configuration values as keyword arguments during initialization. This approach provides a convenient way to configure data processors on-the-fly without the need for a separate configuration instance.


Invoking Data Processors
~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`call` method serves as the gateway for invoking data processors. It plays a crucial role in applying the specified transformations to input features, ultimately enriching the dataset with new or modified features.

The primary purpose of a data processor's :code:`call` method is to integrate it into the data flow graph. This method accepts input features as arguments, which can come from either the outputs of other processors within the data flow or directly from the source features of the dataset. By calling this method, users can seamlessly apply data transformations, facilitating the creation of complex data processing pipelines.

**Internals of the call method**

The :code:`call` method of a data processor serves as the core mechanism for integrating the processor into the data flow graph. This method orchestrates the processing of input features and constructs the necessary connections within the data flow. Here's an in-depth look at how the call method operates:

1. **Input Verification**: The call method rigorously verifies the feature types of the input feature references. This step ensures the consistency and type safety of the provided inputs. By validating inputs early in the process, potential errors can be caught and addressed during the data flow construction phase.
2. **Data Flow Construction**: Following successful input verification, the call method constructs a new node within the data flow graph. This node represents the data processor and establishes connections with the provided input features, defining the processing dependencies within the data flow.
3. **Output Generation**: The :code:`call` method outputs feature references that represent the enriched features after applying the specified data transformation. These feature references can be used for further processing as inputs to other processors, therby building more complex data flows.

**Example: Invoking a Transformers Tokenizer**

Let's illustrate the usage of the :code:`call` method with a practical example. Consider a scenario where we want to tokenize text inputs using a :class:`TransformersTokenizer` data processor within a data flow. Here's how we can achieve this using the :code:`call` method:

.. code-block:: python

    # using the imdb dataset as an example
    ds = datasets.load_dataset("imdb", split="train")

    # create a data flow with the features from the dataset
    flow = DataFlow(ds.features)

    # Call the tokenizer processor with input features
    tokenized_features = tokenizer.call(text=flow.src_features.text)

Implementing Custom Data Processors
-----------------------------------

Custom data processors provide a way to extend the functionality of the data flow framework by implementing custom data transformation operations tailored to specific use cases. Here's a step-by-step guide on how to implement a custom data processor:

1. Define Input and Output References
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by defining input and output reference classes (:code:`InputRefs` and :code:`OutputRefs`). These classes specify the structure and types of input and output features expected by the data processor. Ensure that input references match the features required for processing and output references define the features generated by the processor.

.. code-block:: python

    import datasets
    from hyped.data.flow.refs.ref import FeatureRef
    from hyped.data.flow.refs.inputs import InputRefs, CheckFeatureEquals
    from hyped.data.flow.refs.outputs import OutputRefs, OutputFeature

    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, CheckFeatureEquals(datasets.Value("string"))]

    class CustomOutputRefs(OutputRefs):
        y: Annotated[FeatureRef, OutputFeature(datasets.Value("string"))]

For more information on specifying input and output references, please refer to the :doc:`InputRefs <api/data.flow.refs.inputs>` and :doc:`OutputRefs <api/data.flow.refs.outputs>` documentation, respectively.


2. Define Configuration
~~~~~~~~~~~~~~~~~~~~~~~
    
If your custom data processor requires configurable parameters, define a configuration class (`CustomConfig`) inheriting from :code:`BaseDataProcessorConfig`. This class allows users to customize the behavior of the processor by adjusting configuration parameters.

.. code-block:: python

    from hyped.data.flow.processors.base import BaseDataProcessorConfig

    class CustomConfig(BaseDataProcessorConfig):
        val: float = 1.0

3. Implement Custom Processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a custom processor class (:code:`CustomProcessor`) inheriting from :class:`BaseDataProcessor`. Override the :code:`process` method to define the processing logic for individual input samples. Access configuration values and input features within the `process` method to perform custom transformations.

.. code-block:: python

    from hyped.data.flow.processors.base import Sample, BaseDataProcessor
    
    class CustomProcessor(BaseDataProcessor[CustomConfig, CustomInputRefs, CustomOutputRefs]):
        def process(self, inputs: Sample, index: int, rank: int) -> Sample:
            # Access configuration values
            val = self.config.val
            # Custom process function combining index and input feature
            return Sample(y=f"Index {index} has content {inputs['x']}")

**Asynchronous Processing Example:**

Asynchronous processing is particularly useful for IO-bound tasks or operations that involve waiting for external resources. It allows the processor to execute other tasks while waiting, thus improving overall efficiency. Hyped supports asynchronous processing, enabling seamless integration of asynchronous operations into your data processing pipeline.

.. code-block:: python

    from asyncio import sleep
    from hyped.data.flow.processors.base import Sample, BaseDataProcessor

    class CustomAsyncProcessor(BaseDataProcessor[CustomConfig, CustomInputRefs, CustomOutputRefs]):
        async def process(self, inputs: Sample, index: int, rank: int) -> Sample:
            # Simulate asynchronous processing
            await sleep(1)
            return Sample(y=f"Index {index} has content {inputs['x']}")

**Batch Processing Example:**

Implementing the :code:`batch_process` function allows for batch processing, which can significantly improve the efficiency of data processing tasks, especially for operations that can be vectorized. By overriding this function, you can define custom batch processing logic tailored to your specific requirements.

.. code-block:: python

    class CustomBatchProcessor(BaseDataProcessor[CustomConfig, CustomInputRefs, CustomOutputRefs]):
        async def batch_process(self, inputs: Batch, index: list[int], rank: int) -> Batch:
            # Custom batch processing logic
            processed_batch = {
                key: [
                    f"Index {i} has content {value}" for value in values
                ]
                for key, values in inputs.items()
            }
            return processed_batch

4. Instantiate and Apply the Custom Processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instantiate the custom processor with optional configuration parameters. Use the :code:`call` method to apply the processor to input features within the data flow. Provide input features as arguments to the :code:`call` method, and retrieve the processed output features for further analysis or processing.

.. code-block:: python

    # Instantiate the custom processor
    custom_processor = CustomProcessor(CustomConfig(val=2.0))
    # Apply the custom processor to input features
    processed_features = custom_processor.call(x=flow.src_features.text)
