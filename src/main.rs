use futures_util::StreamExt;
use langchain_rust::{
    chain::{Chain, ConversationalRetrieverChainBuilder},
    document_loaders::{Loader, pdf_extract_loader::PdfExtractLoader},
    embedding::{Embedder, OllamaEmbedder},
    fmt_message, fmt_template,
    llm::client::{Ollama, OllamaClient},
    memory::SimpleMemory,
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::{Document, Message},
    template_jinja2,
    vectorstore::{Retriever, VecStoreOptions, VectorStore, surrealdb::StoreBuilder},
};
use serde_json::json;
use std::{io::Write, sync::Arc};
use surrealdb::{Surreal, engine::any::Any};
use tokio::{
    fs, task::JoinSet
};

#[tokio::main]
async fn main() {
    // Connect to ollama
    let client = Arc::new(OllamaClient::try_new("http://10.22.98.20:11434").unwrap());
    let embedder = OllamaEmbedder::new(client.clone(), "nomic-embed-text", None);
    let llm = Ollama::new(client.clone(), "llama3.2", None);

    // Technically this is slow and stuff, idc
    //
    // This dynamically gets the vector size by doing a test embed.
    //
    // Technically it would be better to hard-code this value then we wouldn't
    // be doing extra embeds.
    let dimension = embedder.embed_query("test query").await.unwrap().len();

    // Get db
    let db = get_db().await;

    // Initialize the SurrealDB Vector Store
    let store = StoreBuilder::new()
        .embedder(embedder)
        .db(db)
        .vector_dimensions(dimension as i32)
        .build()
        .await
        .unwrap();

    // Intialize the tables in the database. This is required to be done only once.
    store.initialize().await.unwrap();

    store
        .add_documents(&get_documents().await, &VecStoreOptions::default())
        .await
        .unwrap();
    println!("Done storing");

    let prompt = message_formatter![
        fmt_message!(Message::new_system_message("You are a helpful assistant")),
        fmt_template!(HumanMessagePromptTemplate::new(template_jinja2!(
            "
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{{context}}

Question:{{question}}
Helpful Answer:

        ",
            "context",
            "question"
        )))
    ];

    // crate AI chain
    let chain = ConversationalRetrieverChainBuilder::new()
        .retriever(Retriever::new(store, 5))
        .memory(SimpleMemory::new().into())
        .prompt(prompt)
        .llm(llm)
        .build()
        .expect("Error building llm chain");

    loop {
        // Ask for user input
        print!("\nAsk> ");
        std::io::stdout().flush().unwrap();
        let mut query = String::new();
        std::io::stdin().read_line(&mut query).unwrap();
        println!("\n");

        let input = prompt_args! {
            "question" => query,
        };

        // TODO how do you get it to respond with the file name as well?
        let res = chain.invoke(input).await.unwrap();
        println!("{}", res);
    }
}

async fn get_documents() -> Vec<Document> {
    let mut futures = JoinSet::new();

    let mut contents = fs::read_dir("./pdfs/").await.unwrap();

    // loop thru all entries in the folder
    while let Ok(file) = contents.next_entry().await {
        if let Some(file) = file {
            if let Some(name) = file.file_name().to_str() {
                let name = name.to_string();
                // take only pdfs
                if name.ends_with("pdf") {
                    // multi-thread because it's cpu intensive
                    let fut = tokio::spawn(async move {
                        if let Ok(loader) = PdfExtractLoader::from_path(file.path()) {
                            if let Ok(loaded) = loader.load().await {
                                let doc = loaded.map(|d| d.unwrap()).collect::<Vec<_>>().await;
                                assert_eq!(doc.len(), 1);

                                // get the parsed document to modify metadata
                                if let Some(mut document) = doc.into_iter().next() {
                                    println!("{name} - ✅");

                                    // TODO how do we get this back at the end?
                                    document.metadata.insert("document_name".to_string(), name.into());
                                    return Some(document);
                                }
                            }
                        }
                        println!("{name} - ❌");
                        None
                    });
                    futures.spawn(fut);
                }
            }
        } else {
            break
        }
    }

    let results = futures.join_all().await;
    results
        .into_iter()
        .flatten()
        .flatten()
        .collect()
}

async fn get_db() -> Surreal<Any> {
    let surrealdb_config = surrealdb::opt::Config::new()
        .set_strict(true)
        .capabilities(surrealdb::opt::capabilities::Capabilities::all());
    // .user(surrealdb::opt::auth::Root {
    // username: "root".into(),
    // password: "root".into(),
    // });

    let db = surrealdb::engine::any::connect(("ws://localhost:8000", surrealdb_config))
        .await
        .unwrap();
    db.use_ns("test").await.unwrap();
    db.use_db("test").await.unwrap();

    db
}

