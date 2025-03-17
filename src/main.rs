use futures_util::StreamExt;
use inline_colorization::*;
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
use std::{io::Write, sync::Arc};
use surrealdb::{Surreal, engine::any::Any};
use tokio::{
    fs::{self, DirEntry},
    task::JoinSet,
    time::Instant,
};

#[tokio::main]
async fn main() {
    let start_time = Instant::now();

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
    println!("{color_yellow}Discovered vector dimensions to be {dimension}{color_reset}");

    // Get db
    let db = get_db().await;

    // Initialize the SurrealDB Vector Store
    let store = StoreBuilder::new()
        .embedder(embedder)
        .db(db.clone())
        .vector_dimensions(dimension as i32)
        .build()
        .await
        .unwrap();

    // Intialize the tables in the database. This is required to be done only once.
    store.initialize().await.unwrap();

    // read the dir
    let mut dir = fs::read_dir("./pdfs")
        .await
        .expect("Directory doesn't exist");

    // filer the dir
    let mut futures = JoinSet::new();

    while let Ok(Some(entry)) = dir.next_entry().await {
        let file = entry.file_name();
        if let Ok(name) = file.into_string() {
            #[derive(serde::Deserialize, Debug)]
            struct DbDocument {
                #[allow(dead_code)]
                collection: String,
                #[allow(dead_code)]
                id: surrealdb::sql::Thing,
            }

            // we only want pdfs
            if !name.ends_with("pdf") {
                continue;
            }

            let l = db.clone();
            let fut = tokio::spawn(async move {
                // check if the file exists in the db yet
                if let Ok(result) = l.query("SELECT id,metadata.collection AS collection FROM document WHERE metadata.document_name = $name")
                .bind(("name", name))
                .await
                .expect("DB query failed")
                .take::<Vec<DbDocument>>(0)
                {
                    if result.len() == 0 {
                        // file isn't preset; we'll need it later to store
                        return Some(entry);
                    }
                }
                None
            });
            futures.spawn(fut);
        }
    }

    let results = futures.join_all().await;
    let pdfs_to_store = results.into_iter().flatten().flatten().collect();

    store
        .add_documents(&embed_pdfs(pdfs_to_store).await, &VecStoreOptions::default())
        .await
        .unwrap();

    let prompt = message_formatter![
        fmt_message!(Message::new_system_message(
            "You are a helpful assistant, helping new users of Infinite Campus use it."
        )),
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

    let retriever = Retriever::new(store, 20).with_options(VecStoreOptions {
        score_threshold: Some(0.5),
        ..Default::default()
    });

    // crate AI chain
    let chain = ConversationalRetrieverChainBuilder::new()
        .retriever(retriever)
        .memory(SimpleMemory::new().into())
        .prompt(prompt)
        .llm(llm)
        .build()
        .expect("Error building llm chain");

    println!(
        "{color_magenta}Startup took {}s{color_reset}",
        start_time.elapsed().as_secs()
    );
    loop {
        // Ask for user input
        print!("\n{color_green}Ask> {color_yellow}");
        std::io::stdout().flush().unwrap();
        let mut query = String::new();
        std::io::stdin().read_line(&mut query).unwrap();
        println!("{color_reset}\n");

        let input = prompt_args! {
            "question" => query,
        };

        // TODO how do you get it to respond with the file name as well?
        let res = chain.invoke(input).await.unwrap();
        println!("{}", res);
    }
}

async fn embed_pdfs(pdfs: Vec<DirEntry>) -> Vec<Document> {
    let mut futures = JoinSet::new();

    let now = Instant::now();
    let len = pdfs.len();
    // loop thru all entries in the folder
    for file in pdfs {
        if let Some(name) = file.file_name().to_str() {
            let name = name.to_string();
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
                            document
                                .metadata
                                .insert("document_name".to_string(), name.into());
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

    let results = futures.join_all().await;

    let milis = now.elapsed().as_millis();
    println!("Loaded {len} documents in {milis}ms");

    results.into_iter().flatten().flatten().collect()
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
