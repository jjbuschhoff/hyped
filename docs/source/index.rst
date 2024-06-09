.. hyped documentation master file, created by
   sphinx-quickstart on Sat Apr 13 18:42:56 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to hyped's documentation!
=================================

Hyped is a versatile framework built on top of `HuggingFace datasets <https://huggingface.co/docs/datasets/en/index>`_, designed to simplify the management and execution of data pipelines. With Hyped, you can define data pipelines as a Directed Acyclic Graph (DAG) of data processors, leveraging the rich ecosystem of the datasets library while also providing the flexibility to implement custom processors when needed.

Hyped aims to offer an intuitive high-level interface for defining and executing data pipelines, all while being flexible and scalable.

Features
--------

- **Seamless Integration with Hugging Face Datasets**: Utilize the extensive collection of datasets available through HuggingFace. Hyped handles data loading and preprocessing using HuggingFace's powerful tools.
- **Flexible Data Processing**: Flexible and Configurable Data Processing: Define complex data processing workflows by linking data processors in a DAG. Each data processor in Hyped is fully configurable, allowing users to fine-tune their behavior according to specific requirements. Hyped comes with a set of general-purpose processors out of the box, allowing for a wide range of transformations and manipulations on your data, while enabling users to customize workflows seamlessly.
- **Modular Design**: Break down complex workflows into reusable components, improving code readability and maintainability. Hyped's modular approach allows for the creation of clear and organized data processing pipelines. By designing data processors as reusable components, you can easily repurpose them across different pipelines, reducing redundancy and fostering efficient development practices.
- **Custom Processor Support**: Implement custom data processors tailored to your specific requirements. Whether you need to apply domain-specific transformations or integrate with external libraries, Hyped provides the flexibility to extend its functionality as needed.
- **Efficient Execution**: Execute your data pipelines efficiently, whether you're working with small datasets or processing large volumes of data. Hyped supports multiprocessing and data streaming out of the box, enabling efficient utilization of computational resources and avoiding memory limitations when processing large datasets.
- **Scalability**: Hyped provides scalability to handle diverse workload demands, allowing you to seamlessly scale your data processing tasks as needed. Whether you're processing small datasets on a single machine or dealing with large volumes of data across distributed computing environments, Hyped adapts to your workload requirements, ensuring efficient execution and resource utilization.

Basic Concepts
--------------

There are three basic concepts in Hyped: Data Flow, Data Processors, and Unified Data Framework. The following will give a short overview of each.

Data Flow
~~~~~~~~~
Data Flows provides a structured way to organize and execute data processing tasks. It represents the sequence of operations applied to input data to produce an output. Data flows are modeled as Directed Acyclic Graphs (DAGs), where each node represents a processing step, and the edges denote the flow of data between these steps. This structure ensures that data flows are organized, manageable, and easy to visualize.

Data Processors
~~~~~~~~~~~~~~~
Data Processors are the building blocks of data flows. Each processor encapsulates a specific data transformation or operation, making them modular and reusable. By connecting multiple processors, users can construct complex data flows to handle a wide range of data processing tasks. This modularity not only simplifies the development process but also enhances the maintainability and scalability of the data processing workflows.

Unified Data Framework
~~~~~~~~~~~~~~~~~~~~~~
The Unified Data Framework in Hyped integrates seamlessly with the Hugging Face datasets library. This integration provides a unified approach to working with various dataset formats, such as PDF, CSV, and JSON. Additionally, it enables efficient streaming of data to and from disk, allowing users to process datasets larger than the available memory. This framework ensures that data handling in Hyped is both versatile and efficient.


Content
-------

Below is a list of key sections in this documentation, guiding you through installation, basic usage, API references, and more.

.. toctree::

   getting_started
   data_flow
   data_processors
   data_aggregators
   operators
   add_ons

.. toctree::
   :maxdepth: 2
   :caption: API References

   api/hyped

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
