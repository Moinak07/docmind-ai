TEST_QUESTIONS = [
    {
        "id": "advanced_python_author",
        "question": "Who is the author of 120 Advanced Python Interview Questions?",
        "expected": {
            "source": "Advanced Python Interview Questions.pdf",
            "page": 1,
            "text_contains": "By Hernando Abella",
        },
    },
    {
        "id": "advanced_python_copyright",
        "question": "What is the copyright year of the Advanced Python Interview Questions PDF?",
        "expected": {
            "source": "Advanced Python Interview Questions.pdf",
            "page": 2,
            "text_contains": "COPYRIGHT 2023",
        },
    },
    {
        "id": "advanced_python_first_toc_item",
        "question": "What is the first item listed in the Advanced Python Interview Questions table of contents?",
        "expected": {
            "source": "Advanced Python Interview Questions.pdf",
            "page": 3,
            "text_contains": 'Why would you use the "pass" statement?',
        },
    },
    {
        "id": "advanced_python_pep8_toc_item",
        "question": "Which table of contents entry is about the Python Enhancement Proposal?",
        "expected": {
            "source": "Advanced Python Interview Questions.pdf",
            "page": 3,
            "text_contains": "PEP 8: The Python Enhancement Proposal",
        },
    },
    # --- CSV ground truth -----------------------------------------------------
    # CSVLoader emits one Document per data row with metadata {"source", "row"}.
    # "row" is 0-based and counts data rows only: csv.DictReader consumes the
    # header line as field names, so it is never a Document. The first data row
    # of IRIS.csv ("5.1,3.5,1.4,0.2,Iris-setosa") is therefore row 0, and
    # backend.chunking packs consecutive rows into one chunk that inherits the
    # first row's metadata -> the chunk holding row 0 is tagged row 0.
    #
    # page is deliberately absent below: load_document() stamps page "N/A" on
    # every non-PDF format, so a CSV chunk is addressable only by row.
    #
    # text_contains uses CSVLoader's real rendering ("key: value" per line, not
    # the raw comma-separated line). evaluate.normalize_text() collapses runs of
    # whitespace on both sides before comparing, so the newlines here match the
    # chunk regardless of how the rows were joined.
    {
        "id": "iris_columns",
        "question": "What are the column names in the IRIS dataset?",
        "expected": {
            "source": "IRIS.csv",
            "row": 0,
            # All five column names appear as the keys of every rendered row.
            "text_contains": (
                "sepal_length: 5.1\n"
                "sepal_width: 3.5\n"
                "petal_length: 1.4\n"
                "petal_width: 0.2\n"
                "species: Iris-setosa"
            ),
        },
    },
    {
        "id": "iris_first_row_values",
        "question": "What are the sepal_length and sepal_width values in the very first row?",
        "expected": {
            "source": "IRIS.csv",
            "row": 0,
            "text_contains": "sepal_length: 5.1\nsepal_width: 3.5",
        },
    },
]
