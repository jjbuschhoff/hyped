Statistic Processors
====================

Dataset statistic processors are a specialized category of data processors within Hyped's framework that compute statistics over the entire dataset. Unlike standard data processors, which operate atomically on individual examples in the dataset, dataset statistic processors aggregate information from the entire dataset to compute various statistics. This distinction is crucial as dataset statistic processors provide insights into the overall characteristics and distributions of the dataset, enabling users to make informed decisions about data preprocessing, modeling, and evaluation strategies. Let's explore the functionality and usage of dataset statistic processors in more detail.

Example: Computing Mean and Standard Deviation
----------------------------------------------

Here's an example demonstrating the use of a dataset statistic processor to compute the mean and standard deviation of a specific feature in the dataset:

.. code-block:: python

   from hyped.data.pipe import DataPipe
   from hyped.data.processors.statistics.report import StatisticsReport
   from hyped.data.processors.statistics.value.mean_and_std import MeanAndStd, MeanAndStdConfig

   # Define a data processing pipeline with a MeanAndStd processor
   pipe = DataPipe(
       [
           MeanAndStd(
               MeanAndStdConfig(
                   statistic_key="stat",
                   feature_key="value"
               )
           )
       ]
   )

   # Apply the data processing pipeline to the dataset and collect statistics
   with StatisticsReport() as report:
       ds = pipe.apply(ds)

In this example, a :code:`MeanAndStd` dataset statistic processor is used to compute the mean and standard deviation of a feature named :code:`value` in the dataset. The computed statistics are then collected using a :code:`StatisticsReport` instance, providing insights into the distribution of the feature across the dataset.
