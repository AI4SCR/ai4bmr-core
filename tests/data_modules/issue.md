I run the following test:
```python
def test():
    import os
    from dotenv import find_dotenv
    print()
    print('find_dotenv():', find_dotenv())
    print('find_dotenv(usecwd):', find_dotenv(usecwd=True))
    # set a break point here and run the lines below manually
    # find_dotenv() does not work anymore
    # %% now run manually
    print('find_dotenv():', find_dotenv())
    print('find_dotenv(usecwd):', find_dotenv(usecwd=True))
    assert True
```

My directory structure is the following:
```text
tests/data_modules
├── .env
├── .env.prod
├── __init__.py
├── issue.md
└── test_project.py
```
