# PDF Parsing Pipeline to Embeddings CLI

This repository contains a command-line interface (CLI) tool for converting JSON documents outputted by the PDF parsing pipeline to embeddings. The tool is designed to process the parsed documents and generate embeddings using the SentenceTransformer model.

## Requirements

- Python >3.8
- poetry package manager

## Usage

To run the CLI tool locally, use the following command:

```bash
python main.py [OPTIONS] INPUT_DIR OUTPUT_DIR
```

Alternatively build and use the image via Docker after creating an environment file and filling in the correct variables. 
```bash
cp .env.example .env 
```

```bash
make build 
make run_embeddings_generation
```

### Options

- `--s3`: Specifies whether the input and output directories are in S3. If this option is provided, the tool will read from and write to S3 instead of the local file system.
- `--redo`: Redo encoding for files that have already been parsed. By default, files with IDs that already exist in the output directory are skipped.
- `--device`: Specifies the device to use for embeddings generation. Available options are "cuda" (for GPU) and "cpu".
- `--limit`: Optionally limits the number of text samples to process. Useful for debugging.

### Arguments

- `INPUT_DIR`: The directory containing the JSON files outputted by the PDF parsing pipeline.
- `OUTPUT_DIR`: The directory to save the embeddings to.

### Example

To convert JSON documents located in the `input` directory to embeddings and save them in the `output` directory, you can use the following command:

```bash
python main.py --s3 input output
```

Make sure to replace `input` and `output` with the appropriate directories or S3 paths.

