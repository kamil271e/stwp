"""Legacy API entry point - redirects to new package structure.

This file is kept for backwards compatibility.
The API has been moved to src/stwp/api/main.py
"""

from stwp.api.main import main

if __name__ == "__main__":
    main()
