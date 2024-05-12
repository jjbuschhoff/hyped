Data Pipe
=========

Built on top of the `HuggingFace datasets <https://huggingface.co/docs/datasets/en/index>`_ library, the :code:`DataPipe` class offers seamless integration with its extensive functionality for efficient dataset management. A :code:`DataPipe` is defined as a sequence of data processors, transforming input data through customizable processing stages. It provides comprehensive functionality for feature management, ensuring compatibility and consistency throughout the pipeline. Additionally, the :code:`DataPipe` class includes advanced features such as multiprocessing support, enhancing performance in computationally intensive tasks, and data streaming capabilities for handling large datasets efficiently.

Key Features
------------

- **Customizability**: Chain together a series of data processors to create a customizable data processing pipeline. This allows for flexible and modular data transformation workflows tailored to specific use cases.

- **Feature Handling**: Manage input and output dataset features with ease, ensuring compatibility and consistency throughout the processing pipeline.

- **Batch Processing**: Easily process batches of examples through the pipeline using convenient methods provided by the :code:`DataPipe` class. Batch processing ensures efficient handling of large datasets and enables parallelized execution for improved performance.

- **Multiprocessing Support**: Harness the power of multiprocessing to accelerate data processing tasks, improving efficiency and scalability. The :code:`DataPipe` class includes built-in support for multiprocessing, allowing for parallel execution of data processing steps across multiple CPU cores.

- **Data Streaming**: Efficiently handle large datasets by streaming data through the pipeline, reducing memory usage and enabling processing of datasets that exceed available memory constraints.

- **Integration with Hugging Face datasets library**: Heavily relies on the Hugging Face datasets library for efficient data processing tasks, leveraging its extensive functionality and dataset management capabilities. This seamless integration enables users to leverage the rich ecosystem of datasets available in the Hugging Face library within the :code:`DataPipe` framework.


Main Functionalities
--------------------

The :code:`DataPipe` class provides essential functionality for preparing, processing batches of examples, and applying the data pipeline to datasets.

init
~~~~

The :code:`__init__` function initializes a :code:`DataPipe` object with a sequence of data processors. It allows users to specify an initial pipeline of processors to be applied to the data. If no processors are provided, the data pipe is initialized with an empty list of processors.

.. code-block:: python

   # Initialize a DataPipe object with a sequence of data processors
   pipe = DataPipe(processors=[processor1, processor2, ...])
   
   # If no processors are provided, initialize the DataPipe with an empty list
   pipe = DataPipe()

Please refer to the :doc:`Data Processors Documentation <data_processors>` for more information about customizing data processors.

prepare
~~~~~~~

The :code:`prepare()` function prepares all data processors within the data pipeline for a specific input dataset. It ensures that each processor is ready to execute by passing the input dataset features through the pipeline and collecting the output features. This step enables seamless data processing by establishing compatibility and consistency throughout the pipeline.

.. code-block:: python

   features = datasets.Features(
	   {
		   "featureA": datasets.Value("string"),
		   "featureB": datasets.Value("string"),
	   }
   )
   # prepare the data pipe for the input-features
   out_features = pipe.prepare(features)

After preparing the data pipeline, the following attributes are set

.. code-block:: python

   # the input features
   pipe.in_features
   # the new features computed by the data pipe that are not part
   # of the input features
   pipe.new_features
   # the output features of the data pipe, can also include input
   # features when they are not filtered out
   pipe.out_features


batch_process
~~~~~~~~~~~~~

The :code:`batch_process()` function takes in a batch of examples and passes it through each data processor of the data pipe. This sequential processing enables the application of various transformations, such as feature extraction, normalization, and filtering, to the input data. The function iterates over the data processors, applying each processor's transformation to the batch of examples and yielding the output of the last data processor.

.. code-block:: python

   # create a sample batch in the huggingface datasets format,
   # i.e. dict of lists
   batch = {
       "featureA": ["x", "y", "z", ...],
       "featureB": ["a", "b", "c", ...]
   }
   # apply the data pipe to the batch of examples
   out = pipe.batch_process(batch, index=[...], rank=0)


apply
~~~~~

The :code:`apply()` function automates the application of the data pipeline to a HuggingFace dataset, returning the transformed dataset. It seamlessly integrates with the HuggingFace datasets library, allowing users to leverage its extensive functionality and dataset management capabilities.

.. code-block:: python

   # Load a dataset using the Hugging Face datasets library
   ds = datasets.load_dataset(...)

   # Apply the data pipeline to the dataset
   out_ds = pipe.apply(ds, batch_size=32)

The :code:`apply()` function offers additional functionalities:

**Multiprocessing**

The :code:`apply()` function supports multiprocessing to accelerate data processing tasks. Users can specify the number of processes to utilize for parallel execution.

.. code-block:: python

   # Apply the data pipeline to the dataset using multiprocessing
   out_ds = pipe.apply(ds, num_proc=16)

**Data Streaming**

When applied to a dataset in streaming mode, the :code:`apply()` function operates in a lazy manner, allowing for data streaming. The data pipe is applied as items are accessed, minimizing memory usage and enabling processing of datasets larger than available memory.

.. code-block:: python

   # Load the dataset in stream mode
   ds = datasets.load_dataset(..., stream=True)
   
   # Apply the data pipeline to the dataset
   out_ds = pipe.apply(ds, batch_size=32)
   
   # Iterate over the transformed dataset
   for item in out_ds:
       pass

   # Write the transformed dataset to disk
   # Note: Multiprocessing settings need to be defined in the writer
   JsonDatasetWriter("dump/", num_proc=1).consume(out_ds)

