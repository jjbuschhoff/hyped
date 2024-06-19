Data Flow
=========

Data Flows provides a structured way to organize and execute data processing tasks. It represents the sequence of operations applied to input data to produce an output. Data flows are modeled as Directed Acyclic Graphs (DAGs), where each node represents a processing step, and the edges denote the flow of data between these steps. This structure ensures that data flows are organized, manageable, and easy to visualize.

Structure of Data Flows
-----------------------

In Hyped, data flows are structured representations of the processing pipeline, consisting of nodes and edges that dictate the flow of data.

Directed Acyclic Graph (DAG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A Directed Acyclic Graph (DAG) is a graph with directed edges where no cycles exist. This ensures that data flows in a single direction from input to output without any loops, making the process straightforward and preventing infinite loops. Each node in the DAG represents a processing step, and the edges denote the flow of data between these steps.

In Hyped, the DAG structure ensures that data processing is organized and manageable. It allows for clear visualization of the processing pipeline, making it easier to understand and debug. The absence of cycles guarantees that each piece of data is processed exactly once in a defined order, enhancing the reliability and predictability of the data flow.

.. image:: _static/DAG.svg

Nodes and Edges
~~~~~~~~~~~~~~~
Nodes serve as modular and configurable processing steps within a data flow, encapsulating specific operations or transformations applied to input data. Designed for flexibility and reusability, nodes are interconnected to form a DAG, enabling the creation of complex data processing pipelines. Each node operates as a coroutine, allowing for asynchronous execution and efficient utilization of computational resources. This design promotes code maintainability and scalability, facilitating the development of robust and scalable data processing workflows that optimize performance and throughput.

Edges in a data flow represent the pathways along which data moves between nodes. They denote the dependencies and sequence of operations, ensuring that data flows in a logical and structured manner. The edges also define the order in which nodes are executed, maintaining the integrity of the data processing pipeline.


Types of Nodes
~~~~~~~~~~~~~~
Nodes in a data flow can be of various types, each performing a specific function:

- **Source Node**: A Data Flow always has exactly one source node, representing the entry points of data into the flow. The data going into this node is provided from the dataset to be processed.
- **Data Processor Nodes**: Processor nodes apply transformations to the features of an isolated example in the dataset. This might include tokenization or normalization.
- **Data Aggregator Nodes**: Aggregator nodes perform dataset-wide statistical operations on the features. This might include summation or averaging.
- **Data Augmentation Nodes**: Coming Soon


Execution Model
---------------

The execution model employed by the data flow architecture plays a crucial role in efficiently processing data and optimizing performance. This section provides an overview of the execution model and explores how asynchronous execution and parallelization are leveraged to enhance efficiency.

Batch Processing
~~~~~~~~~~~~~~~~

Batch processing is a fundamental aspect of the data flow execution model, facilitating the efficient handling of large datasets. In batch processing, a batch of samples flows through the data flow graph, rather than individual samples, enhancing computational efficiency and resource utilization.

Within the data flow architecture, each processor node operates on a batch of samples, enabling parallelized computation and optimization of processing throughput. This batching mechanism streamlines data flow execution, minimizing the overhead associated with processing individual instances and maximizing computational parallelism.

By processing data in batches, the data flow architecture achieves improved throughput and scalability, making it well-suited for handling large-scale datasets and high-throughput processing tasks. Batch processing also enhances memory efficiency by minimizing redundant computations and optimizing data access patterns.

Synchronization of Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~

Synchronization of inputs is achieved by enforcing a mechanism where each node in the DAG waits until all its parent nodes have completed their execution. This synchronization ensures that the inputs required by a node are fully prepared and available before the node begins processing. By waiting for its parent nodes to finish execution, a node guarantees that all necessary data dependencies are satisfied, maintaining the integrity and consistency of the data flow.

The synchronization approach is crucial, especially as Hyped utilizes an asynchronous execution model. Challenges in managing data dependencies and ensuring synchronization are pronounced in these models. However, strict input synchronization within the DAG effectively addresses these challenges, ensuring coordinated concurrent processing while maintaining data integrity.

Emphasizing input synchronization underscores the necessity of employing Directed Acyclic Graphs (DAGs) in data flow architectures. DAGs provide a structured approach, ensuring orderly data processing without encountering cycles. This structure prevents potential deadlocks or race conditions, promoting reliable execution of data processing tasks.

Asynchronous Execution
~~~~~~~~~~~~~~~~~~~~~~
Asynchronous execution is a key feature of the data flow architecture, allowing tasks to run concurrently without blocking the execution of other tasks. By utilizing asynchronous programming techniques, the data flow can execute multiple tasks concurrently, thereby reducing idle time and improving overall throughput.

**Benefits of Asynchronous Execution:**

- **Improved Concurrency**: Asynchronous execution enables the data flow to perform multiple tasks simultaneously, maximizing resource utilization and minimizing latency.
- **Non-Blocking Operations**: Asynchronous tasks can execute independently, allowing the data flow to proceed with other operations while waiting for I/O-bound tasks to complete.
- **Efficient Resource Management**: Asynchronous execution optimizes resource usage by avoiding unnecessary waiting periods, resulting in better scalability and responsiveness.

.. image:: _static/AsyncExecution.svg

Parallelization
~~~~~~~~~~~~~~~
Parallelization is another key aspect of the data flow execution model, enabling the simultaneous execution of tasks across multiple processing units or cores. By distributing workloads and leveraging parallel processing capabilities, the data flow can accelerate data processing tasks and improve overall performance.

**Techniques for Parallelization:**

- **Data Parallelism**: Data parallelism involves partitioning data into smaller chunks and processing them in parallel across multiple processing units. This approach enhances throughput and scalability, particularly for large-scale data processing tasks.
- **Pipeline Parallelism**: Coming Soon

.. figure:: _static/PipelineParallel.svg

   Pipeline Parallelism

Optimizing Performance
~~~~~~~~~~~~~~~~~~~~~~
By combining asynchronous execution and parallelization techniques, the data flow architecture optimizes performance and enhances the efficiency of data processing tasks. This approach enables the data flow to handle large volumes of data, meet stringent processing requirements, and deliver timely results.

**Best Practices for Performance Optimization:**

- **Fine-Grained Task Management**: Breaking down tasks into smaller, more granular units facilitates finer control over execution and resource allocation, leading to better load balancing and improved performance.
- **Batch Size Tuning**: Optimizing the batch size parameter based on memory constraints, computational resources, and processing requirements is essential for achieving efficient batch processing.
- **Parallelism Tuning**: Adjusting the degree of parallelism based on workload characteristics and system resources helps achieve optimal performance and scalability across different environments.


Working With Data Flows
-----------------------

The :class:`DataFlow` class in Hyped provides a powerful toolset for organizing and executing data processing tasks. In this section, we'll walk through practical examples of how to work with :class:`DataFlow` instances to build and apply data processing pipelines.

Initializing a DataFlow
~~~~~~~~~~~~~~~~~~~~~~~
To get started, you'll first need to initialize a :class:`DataFlow` object with the features of your dataset. Here's how you can do it:

.. code-block:: python

   import datasets
   from hyped.data.flow import DataFlow

   # Load dataset
   ds = datasets.load_dataset("imdb", split="test")

   # Initialize a DataFlow object with the dataset features
   flow = DataFlow(features=ds.features)

Adding Nodes
~~~~~~~~~~~~

Once you have initialized a :class:`DataFlow` instance, you can enhance it by adding processor nodes to perform specific data processing tasks. These nodes encapsulate operations or transformations applied to input data. Here's an example of how to add a processor node to tokenize the input text features:

.. code-block:: python

   from hyped.data.flow.processors.tokenizers.transformers import TransformersTokenizer

   # Define a tokenizer processor
   tokenizer = TransformersTokenizer(model_name="bert-base-uncased")

   # Apply the tokenizer to the text feature in the data flow
   tokenized_features = tokenizer.call(text=flow.src_features.text)

The :code:`call` method of the tokenizer processor creates the node and adds it to the :class:`DataFlow` instance with the correct dependencies. It ensures that the tokenizer node waits for the completion of its parent nodes before starting execution. The output of the :code:`call` method is the features generated by the processor. These tokenized features can be used for further processing, similar to the source features (:code:`src_features`).

For more information please refer to the :doc:`Data Processor Documentation <data_processors>`

Building a Data Flow
~~~~~~~~~~~~~~~~~~~~

Building a data flow is an optional step in the data processing workflow. This step involves constructing the data flow graph necessary to compute the specified output features. By specifying the output features as the collect argument, you can build a sub-flow that contains only the nodes required to compute these features. This approach optimizes the execution process by excluding unnecessary nodes, resulting in improved efficiency and performance.

To build a data flow, you can use the build method of the DataFlow class:

.. code-block:: python

   # Build a sub-data flow to compute the requested output features
   sub_flow, aggregates = flow.build(collect=tokenized_features, aggregators={...})

The :code:`build` method takes the desired output features as the collect argument and returns a new :class:`DataFlow` instance containing only the nodes necessary to compute these features. This new flow represents a subset of the original data flow, tailored specifically to the computation of the specified outputs.

In addition, the build function takes an optional argument :code:`aggregators` specifying all the aggregator nodes to be computed and returns a proxy object to the aggregate values (:code:`aggregates`). This proxy object is a dictionary mirroring the given :code:`aggregators` argument but contains the up-to-date aggregate values instead of the aggregator nodes. 

Executing Data Processing Tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After building the flow, you can execute data processing tasks within the DataFlow instance. The :code:`batch_process` method is used to process batches of data. Here's how you can execute the tokenizer processor:

.. code-block:: python

   # Execute the tokenizer processor on a batch of data
   processed_batch = sub_flow.batch_process(batch=batch_data, index=batch_index)

It's important to note that the :code:`batch_process` function assumes that the flow has been built before execution. Therefore, it's essential to use the :code:`sub_flow` instance, which is already built.

Applying a Data Flow to a Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Afterwards, you can apply the data flow to your dataset for processing. The apply method integrates the data flow processing with the dataset. Here's how you can apply the tokenizer flow to the dataset:

.. code-block:: python

   # Apply the data flow to the dataset
   processed_dataset, aggregates = flow.apply(ds, collect=tokenized_features, aggreagtes={...})

The apply method takes the dataset (:code:`ds`) as input and optional arguments such as the :code:`collect` argument. The :code:`collect` argument specifies which features the output dataset should contain. If the data flow has not been built before the collect argument becomes crucial. In such cases, the apply function internally builds the flow with this argument. This means that only the sub-flow required to compute the output features will be executed, and any unnecessary nodes will be pruned before execution.

The :code:`apply` method returns a tuple containing the processed dataset (:code:`processed_dataset`) and a snapshot of the aggregated values after processing the dataset (:code:`aggregates`). The :code:`aggregates` output of the function has the following edge-cases:

 - If the data flow doesn't contain any aggregator nodes, then :code:`aggregates` output is :code:`None`.
 - If the dataset is not an iterable dataset, the :code:`aggregates` output is a snapshot of the aggregated values.
 - If the dataset is iterable, the :code:`aggregates` output is a proxy object of the aggregated values instead of a snapshot. This proxy always links to up-to-date aggregate values.

**Batch Processing**

Batch processing optimizes data flow execution by processing data in batches rather than individual instances. The batch_process method facilitates efficient batch processing within the DataFlow instance.

.. code-block:: python

   # Process data in batches
   processed_dataset, _ = flow.apply(ds, collect=tokenized_features, batch_size=32)

The :code:`batch_size` parameter specifies the number of instances to process in each batch. Adjusting this parameter allows fine-tuning of processing efficiency based on memory constraints and computational resources.

**Data Parallelism**

Data parallelism enhances data processing throughput by concurrently processing multiple data instances across multiple processing units or cores. By partitioning data into smaller chunks and processing them simultaneously, it accelerates large-scale data processing tasks.

.. code-block:: python

   # Apply the data flow with data parallelism
   processed_dataset, _ = flow.apply(ds, collect=tokenized_features, num_proc=4)

Adjust the :code:`num_proc` parameter to optimize parallelism based on available resources and workload characteristics.


Streaming Data for Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For handling datasets larger than available memory, Hyped provides support for streaming data processing. This allows you to efficiently process datasets in a streaming fashion, reading and processing data in smaller, manageable chunks without loading the entire dataset into memory at once.

.. code-block:: python

   from hyped.data.io.writers.json import JsonDatasetWriter

   # Load dataset with streaming enabled
   ds = datasets.load_dataset("imdb", split="train", streaming=True)

   # Apply data pipeline (lazy processing for streamed datasets)
   ds = flow.apply(ds, collect=tokenized_features)

   # Write processed examples to disk using 4 worker processes
   JsonDatasetWriter("dump/", num_proc=4).consume(ds)

In this example, the :code:`load_dataset` function is called with the :code:`streaming=True` argument, enabling streaming mode for dataset loading. The apply method is then used to apply the data flow pipeline to the streamed dataset. Since the dataset is streamed, the processing is performed lazily as the data is read, allowing for efficient memory usage.

Finally, the processed examples are written to disk using the :class:`JsonDatasetWriter`, which supports parallel processing with the specified number of worker processes (:code:`num_proc=4`). This enables efficient writing of processed data to disk while leveraging multiple CPU cores for faster execution.

Visualizing Data Flows
~~~~~~~~~~~~~~~~~~~~~~

Before diving into data flow visualization, let's first create a more complex data flow for demonstration purposes. In the following example, we'll construct a data flow that involves template application and tokenization:

.. code-block:: python

   import datasets
   from hyped.data.flow import DataFlow
   from hyped.data.flow.ops import collect
   from hyped.data.flow.processors.tokenizers.transformers import TransformersTokenizer
   from hyped.data.flow.processors.templates.jinja2 import Jinja2

   # Create a more complex data flow for visualization
   complex_flow = DataFlow(features=ds.features)

   # Separate the input features
   text = collect({"text": complex_flow.src_features.text})
   label = collect({"label": complex_flow.src_features.label})

   # Apply template
   input_tmpl = Jinja2(template="Input: {{ inputs.text }}").call(features=text)
   label_tmpl = Jinja2(template="Label: {{ inputs.label }}").call(features=label)

   # Apply tokenizer
   tokenizer = TransformersTokenizer(tokenizer="gpt2")
   encoding = tokenizer.call(
      text=input_tmpl.rendered,
      text_pair=label_tmpl.rendered
   )

Once the data flow is constructed, you can visualize it using the :code:`plot()` method. This method generates a visualization of the data flow graph using matplotlib:

.. code-block::

   # Visualize the data flow
   complex_flow.plot()

The resulting visualization provides insights into the structure of the data flow, including the sequence of operations and data dependencies. This visualization aids in understanding and debugging complex data processing pipelines.

.. image:: _static/flow.pdf