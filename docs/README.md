# kompot Documentation

This directory contains the source code for kompot's documentation.

## Building the Documentation

To build the documentation locally:

1. Install the documentation dependencies:
   ```bash
   pip install -e ".[docs]"
   ```
   
   Or alternatively:
   ```bash
   pip install -r docs/requirements.txt
   ```

2. Build the HTML documentation:
   ```bash
   cd docs
   make html
   ```

3. View the documentation in your browser:
   ```bash
   # On macOS
   open build/html/index.html
   
   # On Linux
   xdg-open build/html/index.html
   
   # On Windows
   start build/html/index.html
   ```

## Documentation Structure

- `source/` - Contains the source files for the documentation
- `source/conf.py` - Sphinx configuration
- `source/*.rst` - ReStructuredText files defining the documentation structure
- `build/` - Generated documentation (not tracked in git)

## Updating the Documentation

When updating the documentation:

1. Edit the relevant `.rst` files in the `source/` directory
2. Run `make html` to rebuild the documentation
3. Review your changes in a web browser
4. Commit your changes to git (only the source files, not the built files)