.. hyped documentation master file, created by
   sphinx-quickstart on Sat Apr 13 18:42:56 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to hyped's documentation!
=================================

Hyped is a versatile framework built on top of Hugging Face Datasets, designed to simplify the management and execution of data pipelines. With Hyped, you can define data pipelines as sequences of data processors, leveraging the rich ecosystem of Hugging Face datasets while also providing the flexibility to implement custom processors when needed.

Features
========

- **Seamless Integration with Hugging Face Datasets**: Utilize the extensive collection of datasets available through `HuggingFace <https://huggingface.co/docs/datasets/en/index>`_ with ease. Hyped handles data loading and preprocessing using HuggingFace's powerful tools.
- **Flexible Data Processing**: Define complex data processing workflows using a sequence of data processors. Hyped comes with a set of general-purpose processors out of the box, allowing for a wide range of transformations and manipulations on your data.
- **Configurable Data Processors**: Each data processor in Hyped is fully configurable, allowing users to fine-tune their behavior according to specific requirements. This flexibility enables users to customize data processing workflows and adapt them to different use cases seamlessly.
- **Custom Processor Support**: Implement custom data processors tailored to your specific requirements. Whether you need to apply domain-specific transformations or integrate with external libraries, Hyped provides the flexibility to extend its functionality as needed.
- **Efficient Execution**: Execute your data pipelines efficiently, whether you're working with small datasets or processing large volumes of data. Hyped supports multiprocessing and data streaming out of the box, enabling efficient utilization of computational resources and avoiding memory limitations when processing large datasets.
- **Scalability**: Hyped provides scalability to handle diverse workload demands, allowing you to seamlessly scale your data processing tasks as needed. Whether you're processing small datasets on a single machine or dealing with large volumes of data across distributed computing environments, Hyped adapts to your workload requirements, ensuring efficient execution and resource utilization.

Content
=======

Below is a list of key sections in this documentation, guiding you through installation, basic usage, API references, and more.

.. toctree::

   getting_started
   feature_access
   data_pipe
   data_processors
   statistic_processors
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
