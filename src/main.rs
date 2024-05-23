use anyhow::{anyhow, Result};
use openai::{
  chat::{ChatCompletion, ChatCompletionMessage, ChatCompletionMessageRole},
  embeddings::{Embedding, Embeddings},
  set_key,
};
use polars::prelude::*;

static EMBED_MODEL: &str = "text-embedding-ada-002";
static GPT_MODEL: &str = "gpt-3.5-turbo";
static TOKEN_BUDGET: usize = 4000;

async fn strings_ranked_by_relatedness<'a, 'b>(
  query: &'b str,
  df: &'a DataFrame,
) -> Result<Vec<&'a str>> {
  println!("SRR: Creating query embedding");
  let embed = Embedding::create(EMBED_MODEL, query, "").await?;

  println!("SRR: Calculating relatedness");
  let mut strings_and_relatedness = df
    .column("embedding")?
    .list()?
    .into_iter()
    .filter_map(|l| l)
    .map(|s| {
      s.f64()
        .unwrap()
        .into_iter()
        .filter_map(|v| v)
        .collect::<Vec<_>>()
    })
    .map(|vec| embed.distance(&Embedding { vec }))
    .zip(df.column("text")?.utf8()?.into_iter().filter_map(|s| s))
    .collect::<Vec<_>>();

  println!("SRR: Sorting texts");
  strings_and_relatedness.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

  println!("SRR: Done");
  Ok(
    strings_and_relatedness
      .into_iter()
      .map(|(_, s)| s)
      .collect(),
  )
}

async fn query_message(query: &str, df: &DataFrame) -> Result<String> {
  println!("Query: Searching for a related text");
  let texts = strings_ranked_by_relatedness(query, &df).await?;
  let mut message = concat!(
    "Use the below Forthright documents ",
    "to answer the subsequent question. ",
    "If the answer cannot be found in the documents, ",
    "write \"I could not find an answer.\""
  )
  .to_owned();

  let question = format!("Question: {query}");

  println!("Query: Building query message");
  let bpe = tiktoken_rs::cl100k_base()?;
  for text in texts {
    let next_document = format!(r#"\n\nForthright document:\n"""\n{text}\n""""#);
    if bpe
      .encode_with_special_tokens(&format!("{message}{next_document}\n\n{question}"))
      .len()
      > TOKEN_BUDGET
    {
      break;
    }
    message.push_str(&next_document);
  }

  println!("Query: Done");
  Ok(format!("{message}\n\n{question}"))
}

pub async fn ask(query: &str, df: &DataFrame) -> Result<String> {
  println!("Ask: Start query building");
  let message = query_message(query, df).await?;

  let messages = vec![
    ChatCompletionMessage {
      role: ChatCompletionMessageRole::System,
      content: Some("You answer questions about Forthright documents.".into()),
      name: None,
      function_call: None,
    },
    ChatCompletionMessage {
      role: ChatCompletionMessageRole::User,
      content: Some(message),
      name: None,
      function_call: None,
    },
  ];

  println!("Ask: Start completion");
  let response = ChatCompletion::builder(GPT_MODEL, messages)
    .temperature(0.)
    .create()
    .await?;

  response
    .choices
    .into_iter()
    .next()
    .ok_or(anyhow!("No completion returned"))?
    .message
    .content
    .ok_or(anyhow!("No completion message!"))
}

const BATCH_SIZE: usize = 1000;
pub async fn create_embeddings() -> Result<DataFrame> {
  let files: Vec<String> = std::fs::read_dir("files")?
    .into_iter()
    .filter_map(|f| f.ok())
    .map(|file| std::fs::read_to_string(file.path()).unwrap())
    .map(|file| {
      file
        .split("\n\n")
        .map(|s| s.trim().to_owned())
        .collect::<Vec<_>>()
    })
    .flatten()
    .filter(|s| s.len() > 128)
    .collect();

  let mut embedding = Series::new_empty("embedding", &DataType::List(Box::new(DataType::Float64)));

  for batch in files
    .chunks(BATCH_SIZE)
    .into_iter()
    .map(|b| b.into_iter().map(|i| i.as_str()).collect::<Vec<_>>())
  {
    dbg!(&batch.len());
    let embed = Embeddings::create(EMBED_MODEL, batch, "").await?;
    embed.data.into_iter().for_each(|e| {
      let new = Series::new("", vec![Series::new("", &e.vec)]);
      embedding.append(&new).unwrap();
    });
  }

  let df = df!("text" => files, "embedding" => embedding)?;

  Ok(df)
}

pub fn str_to_list(s: &Series) -> Series {
  s.utf8()
    .unwrap()
    .into_iter()
    .map(|s| s.map(|s| serde_json::from_str::<Vec<f64>>(s).ok()))
    .flatten()
    .map(|l| l.map(|l| Series::new("", l)))
    .collect::<ListChunked>()
    .into_series()
}

pub fn list_to_str(s: &Series) -> Series {
  s.list()
    .unwrap()
    .into_iter()
    .map(|s| {
      s.map(|s| s.f64().unwrap().into_iter().collect::<Option<Vec<f64>>>())
        .map(|v| serde_json::to_string(&v).unwrap())
    })
    .collect::<Utf8Chunked>()
    .into_series()
}

#[tokio::main]
async fn main() -> Result<()> {
  dotenv::dotenv()?;
  set_key(std::env::var("OPENAI_KEY")?);
  
  println!("Main: Creating embeddings from files");
  let mut df = create_embeddings().await?;
  let writer = std::fs::File::create("df.csv")?;
  println!("Main: Saving embeddings");
  df.apply("embedding", list_to_str)?;
  CsvWriter::new(writer).finish(&mut df)?;
  
  println!("Main: Loading embeddings");
  let mut df = CsvReader::from_path("df.csv")?.finish()?;

  println!("Main: Parsing embeddings");
  df.apply("embedding", str_to_list)?;


  let question = "List in bullet form why the gaps in treatment are considered irrelevant.";
  let response = ask(question, &df).await?;

  println!("Question: {question}\nResponse: {response}");

  Ok(())
}
