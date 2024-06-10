Data Aggregators
================

Data aggregators are a crucial component of the data flow architecture, serving as one of the three primary types of nodes within a data flow. Specifically, data aggregator nodes implement dataset-wide computations, such as calculating the sum, mean, or standard deviation of features across all examples in a dataset. Unlike data processors that operate on individual examples, data aggregators focus on aggregating values from multiple examples to compute summary statistics or other collective metrics.

:code:`Hyped` includes a variety of pre-built, general-purpose data aggregators designed to perform common statistical operations. These aggregators provide robust solutions for calculating dataset-wide metrics, enabling users to easily integrate these computations into their data processing pipelines.

Design Principles
-----------------

- **Thread-Safety**: Data aggregators ensure thread-safety to handle concurrent access and updates to shared resources. This prevents race conditions and maintains the integrity of aggregated data, enabling reliable operation in multi-threaded environments.
- **Efficient Computation**: To achieve computational efficiency, data aggregators separate the extraction and update phases of aggregation. Asynchronous and parallel extraction maximizes resource utilization, while synchronous and thread-safe updates ensure data integrity. This approach balances performance optimization with concurrency control, enabling scalable and responsive aggregation processes.

Data aggregators also adhere to the design principles for data processors, ensuring modularity, isolation, configurability, and clear, type-safe input/output definitions to facilitate seamless integration into diverse data flow architectures.

Execution Model
---------------

Data aggregators operate through a series of well-defined steps that ensure efficient and accurate aggregation of values across the dataset.

The figure below illustrates the workings of the extraction and update phase. It highlights the concurrent execution of the extraction phase alongside the synchronized processing of the update phase which guarantees the integrity of the aggregation outcome.

.. image:: _static/AggregationExecutionModel.svg
    :width: 600

This logic is implemented by the :doc:`DataAggregationManager <api/data.flow.aggregators.base>`.

Initialization Phase
~~~~~~~~~~~~~~~~~~~~

Before the aggregation process begins, data aggregators undergo an initialization phase to set up their internal state. During this phase, the aggregator initializes its state, which includes any parameters or configurations specified by the user. This phase ensures that the aggregator is ready to start processing data and can maintain consistent behavior throughout the aggregation process.

Extraction Phase
~~~~~~~~~~~~~~~~

The extraction phase is where data aggregators retrieve relevant information from the input data batch. Operating asynchronously and often in parallel, this phase maximizes resource utilization and computational efficiency, especially in multi-process environments. Aggregators extract data from each example in the batch, focusing on the features specified for aggregation. For instance, in the case of calculating the sum of numerical features, the extraction phase precomputes the sum of the given input batch.

Update Phase
~~~~~~~~~~~~

Once the relevant information is extracted, the update phase triggers to incorporate this information into the aggregator's current state. The update phase is responsible for aggregating values accumulated during the extraction phase, updating the aggregator's internal state accordingly. Thread-safe mechanisms ensure that updates occur reliably, even in concurrent or multi-threaded settings, preventing data corruption or inconsistency.


Working with Data Aggregators
-----------------------------

This section covers essential aspects such as configuring data aggregators and invoking them through the call method. Understanding these functionalities is crucial for effective data aggregation workflows.

Configuring Data Aggregators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data aggregators in the Hyped library are highly configurable, allowing users to tailor their behavior to specific tasks or requirements. This section explores the various configuration options available for data aggregators and provides examples to illustrate their usage.

**Configuration Options**

Each data aggregator comes with a set of configuration options that define its behavior. These options are encapsulated within a :class:`Pydantic` model, providing a structured and type-safe way to specify the aggregator's parameters.

**Example: Configuring a Sum Aggregator**

Let's consider the example of configuring a Sum Aggregator. This aggregator is responsible for calculating the sum of specified input features across all examples in a dataset.

.. code-block:: python

    from hyped.data.flow.aggregators.ops import (
        SumAggregator, SumAggregatorConfig
    )

    # Define the configuration for the Sum Aggregator
    sum_config = SumAggregatorConfig(
        start=0.0  # Initial value for the sum calculation
    )

    # Create the sum aggregator instance
    sum_aggregator = SumAggregator(sum_config)

In this example, we define a configuration for the :class:`SumAggregator`, specifying the initial value for the sum calculation (:code:`start`). This parameter can be adjusted based on the specific requirements of the aggregation task.

Invoking Data Aggregators
~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`call` method serves as the gateway for invoking data aggregators. It plays a crucial role in applying the specified aggregation operations to input features, ultimately computing the desired dataset-wide metrics.

The primary purpose of a data aggregator's :code:`call` method is to integrate it into the data flow graph. This method accepts input features as arguments, which can come from either the outputs of data processors within the data flow or directly from the source features of the dataset. By calling this method, users can seamlessly apply aggregation operations, facilitating the creation of complex data processing pipelines.

**Example: Invoking a Sum Aggregator**

Let's illustrate the usage of the :code:`call` method with a practical example. Consider a scenario where we want to calculate the sum of a numerical feature using a :class:`SumAggregator` within a data flow. Here's how we can achieve this using the :code:`call` method:

.. code-block:: python

    # Using the imdb dataset as an example
    ds = datasets.load_dataset("imdb", split="train")

    # Create a data flow with the features from the dataset
    flow = DataFlow(ds.features)

    # Call the sum aggregator with input features
    sum_feature = sum_aggregator.call(x=len(flow.src_features["text"]))

Advanced Usage: Aggregating Values Over Multiple Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some scenarios, you may need to aggregate values across multiple datasets rather than within a single dataset. This can be achieved by manually building the data flow with the specified aggregators and then applying the built data flow multiple times. By doing this, the aggregation states are maintained across the different apply calls, resulting in a final aggregated value that encompasses all datasets.

**Example: Aggregating Sum Over Multiple Datasets**

Let's illustrate this process with an example where we calculate the sum of a numerical feature across multiple datasets using a :class:`SumAggregator`.

.. code-block:: python

    from datasets import load_dataset
    from hyped.data.flow import DataFlow
    from hyped.data.flow.aggregators.ops import SumAggregator

    # Load multiple datasets
    ds1 = load_dataset("imdb", split="train[:10%]")
    ds2 = load_dataset("imdb", split="train[10%:20%]")

    # Create a data flow and add the sum aggregator
    flow = DataFlow(ds1.features)
    ref = SumAggregator().call(x=len(flow.src_features["text"]))

    # manually build the data flow
    flow, vals = flow.build(collect=flow.src_features, aggregators={"sum": ref})

    # Apply the data flow to the first dataset
    ds1, vals1 = flow.apply(ds1)
    print(f"Aggregated Value after first dataset: {vals1}")

    # Apply the data flow to the second dataset
    ds1, vals2 = flow.apply(ds1)
    print(f"Aggregated Value after second dataset: {vals2}")

    # the values object returned by the build function always contains
    # the up-to-date values, while the apply function returns a snapshot
    # of the aggregated values at that point
    assert vals != vals1  # vals1 is outdated
    assert vals == vals2  # vals2 is up-to-date

In this example, we first create a data flow and configure a :class:`SumAggregator` to calculate the sum of a numerical feature (length of the :code:`text`). We manually build the data flow graph by adding the sum aggregator node. We then apply the data flow to two different datasets sequentially.

**Notes:**

- **Maintaining State**: By building the data flow graph manually and then applying it, the state of the aggregations is maintained across apply calls. This ensures that the aggregations are cumulative and not reset between calls.
- **Output Values**: The :code:`apply` calls return the aggregated values at that point. These values represent the global aggregated values, encompassing all data processed up to that point. The :code:`build` function returns a values object that is always up-to-date.

This advanced usage enables efficient aggregation of values across multiple datasets, making it a powerful feature for comprehensive data analysis tasks.


Implementing Custom Data Aggregators
------------------------------------

Custom data aggregators provide a way to extend the functionality of the data flow framework by implementing custom aggregation operations tailored to specific use cases. Here's a step-by-step guide on how to implement a custom data aggregator:

1. Define Input References
~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by defining input reference class (:code:`InputRefs`). This class specifies the structure and types of input features expected by the data aggregator. Ensure that input references match the features required for aggregation.

.. code-block:: python

    import datasets
    from hyped.data.flow.refs.ref import FeatureRef
    from hyped.data.flow.refs.inputs import InputRefs, CheckFeatureEquals

    class CustomAggregatorInputRefs(InputRefs):
        x: Annotated[FeatureRef, CheckFeatureEquals(datasets.Value("float32"))]

For more information on specifying input references, please refer to the :doc:`InputRefs <api/data.flow.refs.inputs>` documentation.

2. Define Configuration
~~~~~~~~~~~~~~~~~~~~~~~

If your custom data aggregator requires configurable parameters, define a configuration class (:class:`CustomConfig`) inheriting from :class:`BaseDataAggregatorConfig`. This class allows users to customize the behavior of the aggregator by adjusting configuration parameters.

.. code-block:: python

    from hyped.data.flow.aggregators.base import BaseDataAggregatorConfig

    class CustomConfig(BaseDataAggregatorConfig):
        initial_value: float = 0.0

3. Implement Custom Aggregator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a custom aggregator class (:code:`CustomAggregator`) inheriting from :class:`BaseDataAggregator`. Override the :code:`initialize`, :code:`extract`, and :code:`update` methods to define the aggregation logic. Access configuration values and input features within these methods to perform custom aggregations.

.. code-block:: python

    from hyped.data.flow.aggregators.base import Batch, BaseDataAggregator

    class CustomAggregator(BaseDataAggregator[CustomConfig, CustomAggregatorInputRefs, float]):

        def initialize(self, features: Features) -> tuple[float, None]:
            return self.config.initial_value, None

        async def extract(self, inputs: Batch, index: list[int], rank: int) -> float:
            return sum(inputs["x"])

        async def update(self, val: float, ctx: float, state: None) -> tuple[float, None]:
            return val + ctx, None

Here's a breakdown of each method:

- :code:`initialize`: The :code:`initialize` function is responsible for initializing the aggregator before the aggregation process begins. It takes the dataset's features as input and returns an initial value for aggregation and an initial context. The initial value represents the starting point for the aggregation process, while the initial context provides any additional state information required during aggregation. In some cases, the context might not be applicable, in which case it can be set to :code:`None`.
- :code:`extract`: The :code:`extract` function extracts information from the input data batch. It takes the batch of input data along with any additional parameters required for extraction. This function typically operates asynchronously and in parallel, allowing for efficient processing, especially in multi-process setups. It should return the extracted information relevant to the aggregation process.
- :code:`update`: The :code:`update` function is responsible for updating the aggregated value based on the extracted information, taking into account the current aggregated value, newly extracted context values, and the current aggregation state. It's important to note that the executor calls the :code:`update` function in a thread-safe manner, relieving users from implementing locking mechanisms themselves. The update function returns the updated aggregated value and state.
  
**Best Practices**:

- **Optimize Extract Function**: Since the extract function runs in parallel and can benefit from parallel processing, it's recommended to handle all possible overhead within this function. By optimizing the extract function, you can improve the overall efficiency of the aggregation process.
- **Keep Update Lightweight**: While the extract function can handle heavier computations efficiently, the update function should focus on lightweight update computations. Since the update function is called in a thread-safe manner, heavy computations within this function may impact performance. Keeping the update function lightweight ensures faster aggregation updates and maintains overall system performance.

4. Instantiate and Apply the Custom Aggregator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instantiate the custom aggregator with optional configuration parameters. Use the :code:`call` method to apply the aggregator to input features within the data flow. Provide input features as arguments to the :code:`call` method, and retrieve the aggregated output values for further analysis or processing.

.. code-block:: python

    # Instantiate the custom aggregator
    custom_aggregator = CustomAggregator(CustomConfig(initial_value=10.0))

    # Apply the custom aggregator to input features
    aggregated_value = custom_aggregator.call(x=flow.src_features["numerical_feature"])
