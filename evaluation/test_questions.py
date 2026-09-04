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
]
