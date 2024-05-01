Feature Access
==============

Feature keys play a pivotal role in defining and processing data within Hyped. They serve as identifiers for individual features or data fields, allowing for precise control and manipulation of the data flow throughout the processing pipeline. Hyped utilizes specialized classes, such as :code:`FeatureKey`, :code:`FeatureCollection`, and :code:`FeatureDict`, in the configuration of processors to model the data flow. In this sub-section, we delve deeper into the concept of feature keys and explore these classes, which provide a structured approach to managing and configuring input features for data processing tasks. Understanding these classes is essential for effectively harnessing the power of Hyped's data processing capabilities.

FeatureKey
----------

The :code:`FeatureKey` class is a fundamental component of Hyped's data processing framework, designed to facilitate the indexing of (nested) features within data examples. This class represents a tuple of :code:`str`, :code:`int`, and :code:`slice` instances, serving as a path specifier to navigate to a specific feature within an example. By leveraging :code:`FeatureKey`, users can precisely identify and access individual data fields, enabling targeted processing and manipulation of data. Let's explore how :code:`FeatureKey` operates through an illustrative example:

.. code-block:: python

   # define an example of nested features
   example = {"seq": [{"A": 0}, {"A": 1}, {"A": 2}]}
   
   # Refers to the value 0 in the example
   key = FeatureKey("seq", 0, "A")
   # Refers to the sequence [1, 2] extracted from the example
   key = FeatureKey("seq", slice(1), "A")

In this example, the :code:`FeatureKey` instance :code:`key` specifies the path to access the value 0 within the :code:`example` dictionary. Understanding how to construct and utilize :code:`FeatureKey` instances is crucial for effectively navigating and processing data within Hyped's data processing pipelines. Let's delve deeper into the functionality and usage of the :code:`FeatureKey` class.

**Main Functionalities**

In this section, we outline the main functionalities of the :code:`FeatureKey` class, which provide essential methods for extracting specific features from datasets, single examples, and batches of examples.

1. :code:`index_features(features: datasets.Features) -> datasets.features.features.FeatureType`:

   This method extracts the values of the specified feature represented by the :code:`FeatureKey` instance from a dataset's features dictionary.

   .. code-block:: python

    # some sample dataset features dictionary
    features = Features({"text": Value("string")})

    # get the feature type of the text feature, in this case Value("string")
    value = FeatureKey("text").index_features(features)

2. :code:`index_example(example: dict[str, Any]) -> Any`:

   This method extracts the value of the specified feature represented by the :code:`FeatureKey` instance from a single example.

   .. code-block:: python

    # define an example of nested features
    example = {"seq": [{"A": 0}, {"A": 1}, {"A": 2}]}
    
    # get the feature value of the nested feature, in this case 0
    value = FeatureKey("seq", 0, "A").index_example(example)

3. :code:`index_batch(batch: dict[str, list[Any]]) -> list[Any]`:

   This method extracts the values of the specified feature represented by the :code:`FeatureKey` instance from a batch of examples.

   .. code-block:: python

    # define a batch of three examples
    batch = {"val": [{"A": 0}, {"A": 1}, {"A": 2}]}
    
    # get the feature values of example in the batch, in this case [0, 1, 2]
    values = FeatureKey("val", "A").index_batch(batch)


FeatureCollection
-----------------

The :code:`FeatureCollection` class builds on top of the :code:`FeatureKey` class but is designed to facilitate the dynamic construction of features from sub-features. This class allows users to define complex features by combining existing features and constant values in the form of nested :code:`dict` or :code:`list` types on the fly. Let's explore how :code:`FeatureCollection`` operates through an illustrative example:

.. code-block:: python

    feature_collection = FeatureCollection(
        {
            "const": Const("This is a constant value feature"),
            "ref": [
                FeatureKey("key", "A"),
                FeatureKey("key", "B")
            ]
        }
    )

In this example, the :code:`feature_collection` instance :code:`feature_collection` is created with two sub-features: a constant value feature defined by the :code:`Const` class and reference features obtained from the example using the :code:`FeatureKey` class.

**Main Functionalities**

The :code:`FeatureCollection` class provides the exact same main functionalities as the :code:`FeatureKey`:

- :code:`index_features(features: datasets.Features) -> Any`:
    
    Extracts the values of the specified feature represented by the :code:`FeatureCollection` instance from a dataset's features dictionary.

- :code:`index_example(example: dict[str, Any]) -> Any`:
    
    Extracts the value of the specified feature represented by the :code:`FeatureCollection` instance from a single example.

- :code:`index_batch(batch: dict[str, list[Any]]) -> list[Any]`:

    Extracts the values of the specified feature represented by the :code:`FeatureCollection` instance from a batch of examples.



FeatureDict
-----------

The :code:`FeatureDict` class is a specialized variant of the :code:`FeatureCollection` class within Hyped's data processing framework. While both classes serve the purpose of dynamically constructing features from sub-features, the :code:`FeatureDict` class guarantees that the top-level structure of the feature is a :code:`dictionary`. This ensures that the feature being constructed is inherently structured as a dictionary. Let's explore how :code:`FeatureDict` operates through an illustrative example:

.. code-block:: python

    # valid feature dict
    feature_dict = FeatureDict({"const": Const("This is a constant value feature"), "ref": FeatureKey("key")})

    # invalid feature dict
    feature_dict = FeatureDict([Const("This is a constant value feature"), FeatureKey("key")])

In the first example, feature_dict is a valid :code:`FeatureDict` instance constructed with a dictionary structure, where each key corresponds to a sub-feature. However, in the second example, :code:`feature_dict` is an invalid :code:`FeatureDict` instance because the input is not a dictionary.

**Main Functionalities**

The main functionalities of the :code:`FeatureDict` class are identical to those of the :code:`FeatureCollection` class, with the only difference being the return type of the :code:`index_batch`` function.

- :code:`index_features(features: datasets.Features) -> Any`:
    
    Extracts the values of the specified feature represented by the :code:`FeatureCollection` instance from a dataset's features dictionary.

- :code:`index_example(example: dict[str, Any]) -> Any`:
    
    Extracts the value of the specified feature represented by the :code:`FeatureCollection` instance from a single example.

- :code:`index_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]`:

    Extracts the values of the specified feature represented by the :code:`FeatureDict` instance from a batch of examples. The return type is a dictionary where each key corresponds to a feature name, and the values are lists containing the extracted feature values for each example in the batch.

