# Recomendation system
Top-K sku recomendation. This is done by calc the cosine distance of all client, and rating products
## Requirements

- [Python >= 3.6](https://www.python.org/)
- [Pipenv](https://github.com/pypa/pipenv)

## Development

1. Clone repository: `git clone https://github.com/Joaquinecc/api-bristol.git`
2. Install dependencies: `pipenv install`
3. Activate virtualenv: `pipenv shell`
4. Create a file called `settings-params.json` in root directory
5. Insert the following lines into the file:

   ```
{
    "read_path":"input file",
    "path_write":"csv file directory to save the final result",
    "chunksize":10000,
    "topNProduct":15,
    "topNSimilarity":30
}
   ```
6. Run script: `python main.py`

