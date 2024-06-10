Operators
=========

Operators play a crucial role in data processing pipelines by providing high-level functions for performing common operations on feature references. These functions streamline the process of adding processors to the data flow graph, abstracting away the underlying complexities and allowing users to focus on defining the desired transformations.

For more information about operators and a list of available operators, please refer to the :doc:`operator api documentation <api/data.flow.ops>`.

**Example: Using the `collect` Operator**

One such widely used operator is the `collect` operator, which allows users to gather features from various nodes in the data flow and consolidate them into a single feature. It's particularly handy for defining the output structure of a data flow.

To illustrate the `collect` operator, let's consider a scenario where we have a data flow that applies two different tokenizers to the text feature of the IMDb dataset.

.. code-block:: python

    import datasets
    from hyped.data.flow import DataFlow
    from hyped.data.flow.ops import collect
    from hyped.data.flow.processors.tokenizers.transformers import TransformersTokenizer

    # load the imdb dataset
    ds = datasets("imdb", split="train")
    
    # initialize the data flow
    flow = DataFlow(ds.features)

    # create two tokenizers
    tokenizerA = TransformersTokenizer(tokenizer="bert-base-uncased")
    tokenizerB = TransformersTokenizer(tokenizer="roberta-base")
    # call the tokenizer to the text features
    featuresA = tokenizerA.call(text=flow.src_features.text)
    featuresB = tokenizerB.call(text=flow.src_features.text)

Now, suppose we want to collect the token IDs produced by each tokenizer into a single output dataset. Since the features originate from different nodes in the data flow, we need to collect them before using them as the output.

.. code-block:: python

    # Collect token IDs from both tokenizers
    combined = collect(
        {
            "input_ids_A": featuresA.input_ids,
            "input_ids_B": featuresB.input_ids
        }
    )

With the features collected, we can now proceed to use them as the output for our data flow.

.. code-block:: python

    # Apply the collected features as the output for the data flow
    out_ds, _ = flow.apply(ds, collect=combined)

In this example, the `collect` operator enables us to efficiently consolidate features from distinct processing paths within the data flow, streamlining the output specification process.


Magic Operators
---------------

Magic Operators in Hyped serve as convenient shortcuts, providing syntactic sugar for common operations on feature references. These operators enable users to perform various transformations or manipulations effortlessly. Let's explore a simple example to illustrate the functionality of magic operators.

.. code-block:: python

    import datasets
    from hyped.data.flow import DataFlow

    # load the wiki-qa dataset
    ds = datasets("wiki_qa", split="train")

    # initialize the data flow
    flow = DataFlow(ds.features)

    # combine the question and answer features into a single text feature
    prompt = (
        "Question: " + flow.src_features.question
        + "\nAnswer: " + flow.src_features.answer
    )

In this example, the prompt variable demonstrates the use of magic operators to concatenate the question and answer features, streamlining the data processing workflow.


Pitfalls of Magic Operators
---------------------------

While operators in Hyped provide powerful abstractions for working with feature references, they can also introduce some pitfalls, especially when it comes to operator overloading. A common issue arises when users attempt to compare two feature references using comparison operators like :code:`==`. 

In Hyped, the expression :code:`flow.src_features.question == flow.src_features.answer` does not return a boolean value indicating whether the two reference objects are the same. Instead, it evaluates to another feature reference, which compares the features referenced by the two feature references. This can lead to confusion and unintended results in your data processing pipeline.

.. note:: 

    Always keep in mind that operators on feature references in Hyped are designed to create new references rather than evaluate to primitive data types like booleans.

Consider the following example:

.. code-block:: python

    import datasets
    from hyped.data.flow import DataFlow

    # load the wiki-qa dataset
    ds = datasets("wiki_qa", split="train")

    # initialize the data flow
    flow = DataFlow(ds.features)

    # attempt to compare question and answer features
    comparison = flow.src_features.question == flow.src_features.answer

In this example, :code:`comparison` is a feature reference that represents a comparison operation, not a boolean value. This behavior is by design, as feature references in Hyped are intended to be used within the context of the data flow graph, where operations between features generate new feature references.

Compare Feature Reference Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To properly compare whether two feature references are the same, you should use the :code:`is` operator or explicitly check their identities:

.. code-block:: python

    # Proper way to compare feature references
    are_same = flow.src_features.question is flow.src_features.answer

    # or using identity comparison
    are_same = id(flow.src_features.question) == id(flow.src_features.answer)

In this example, :code:`comparison` is a feature reference that represents a comparison operation, not a boolean value. This behavior is by design, as feature references in Hyped are intended to be used within the context of the data flow graph, where operations between features generate new feature references.

Compare Feature Reference Pointers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compare if two references point to the same feature, you can use the :code:`ptr` attribute defined by feature references. This attribute provides a unique identifier for the feature itself, allowing you to check if two references point to the same underlying feature:

.. code-block:: python

    # Proper way to compare if two references point to the same feature
    same_feature = flow.src_features.question.ptr == flow.src_features.answer.ptr

In this example, :code:`same_feature` will be a boolean value indicating whether the two feature references point to the same feature. By being aware of these pitfalls and using the correct comparison methods, you can avoid unintended behaviors and ensure that your data processing pipeline operates as expected.