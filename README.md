# OpenAI Embeddings & Completion Using Rust

This program's purpose is to generate embeddings data frames from all scraped forthright documents and later use them for analysis using OpenAI completions.

## Building

The program can be easily built using Rust's `cargo`.

1. Install [Rust](https://rustup.rs/). 
2. Clone the repo 
```sh
git clone https://github.com/loyal812/rust-ai-embedding.git
```
3. Run `cargo build --release`
4. Compiled binary will be in the `target/release` directory.

All commands (Linux):
```sh
# install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# clone the repo
git clone https://github.com/loyal812/rust-ai-embedding.git
# build the binary
cargo build --release
```

## Usage

The program has two modes: `Embed` and `Ask`.

```
Usage: demo_embeddings --key <KEY> <COMMAND>

Commands:
  embed  Creates new embeddings DataFrame from raw txt files
  ask    Ask the GPT to provide an answer based on generated embeddings
  help   Print this message or the help of the given subcommand(s)

Options:
  -k, --key <KEY>  OpenAI API Key [env: KEY=]
  -h, --help       Print help
```

Use `Embed` to generate DataFrame csv file required to find the most relevant file for the question.

```
Usage: demo_embeddings --key <KEY> embed [OPTIONS] --files <FILES>

Options:
  -f, --files <FILES>  Path to the Forthright documents to embed
      --df <DF>        Path to the embeddings DataFrame csv file [default: df.csv]
  -h, --help           Print help
```

Use `Ask` to ask your question based on embedded documents.

```
Usage: demo_embeddings --key <KEY> ask --df <DF> --question <QUESTION>

Options:
      --df <DF>              Path to the embeddings DataFrame csv file
  -q, --question <QUESTION>  Question to ask the model for
  -h, --help                 Print help
```