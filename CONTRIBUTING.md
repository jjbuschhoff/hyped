# Contribution Guidelines

Thank you for considering contributing to Hyped! Contributions are welcome and encouraged to help improve the project and make it even better. To maintain a collaborative and welcoming community, please follow these guidelines when contributing to the project:

## Bug Reports and Feature Requests
If you encounter a bug or have a feature request, please open an issue on the issue tracker with a clear description of the problem or enhancement you'd like to see. Be sure to check if a similar issue has already been reported before creating a new one.

## Pull Requests
We welcome pull requests from contributors of all skill levels. If you'd like to contribute code changes, please follow these steps:

1. Fork the repository and create your branch from main.
2. Make your changes and ensure that your code follows the project's coding style and conventions. For more information see 
3. Write tests for your changes if applicable.
4. Ensure that all existing tests and pre-commit hooks pass.
5. Submit a pull request with a clear description of your changes and why they're needed.

## Code Style and Conventions
Please follow the project's coding style and conventions when contributing code changes. This helps maintain consistency and readability across the codebase. Some key conventions to follow include:

1. Naming Conventions: Use descriptive names for variables, functions, and classes that accurately reflect their purpose and functionality. Additionally, adhere to the following naming conventions:
    - Abstract classes should be named starting with Base.
    - Use CamelCase for class names and snake_case for variable and function names.
    - Prefix private variables and functions with a single underscore (_).
3. Indentation and Formatting: Use consistent indentation and formatting throughout the codebase. For Python projects, we follow PEP 8 guidelines for code formatting.
4. Comments and Documentation: Include comments and docstrings to explain complex logic, algorithms, or any non-obvious code sections. Document public functions and classes using docstrings.
5. Imports: Organize imports according to PEP 8 guidelines. Separate imports into three groups: standard library imports, related third-party imports, and local application/library imports. Use absolute imports whenever possible.
6. Error Handling: Handle exceptions gracefully and provide informative error messages when appropriate. Avoid using bare except clauses and handle specific exception types whenever possible.

Additionally, ensure that pre-commit hooks do not fail before submitting your pull request. Pre-commit hooks help ensure code quality and adherence to project standards.

To run pre-commit hooks, execute the following command in your project directory:

```bash
pre-commit run --all-files
```

If any issues are identified, resolve them before submitting your pull request.

## Documentation
Improvements to documentation are always appreciated. If you find errors or outdated information in the documentation, or if you'd like to add new documentation, please submit a pull request with your changes.

We appreciate your contributions to Hyped and look forward to working with you!
