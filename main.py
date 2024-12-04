import json

import gradio as gr
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers.data import JsonLexer

from llm import invoke_model
from rag import embed_chunks, retrieve_context, process_and_split_csv
import numpy as np
import pandas as pd


def display_chunks(chunks_info):
    json_data = json.dumps(chunks_info, indent=4)  # Pretty-print JSON
    formatter = HtmlFormatter(style="friendly", full=False)
    highlighted = highlight(json_data, JsonLexer(), formatter)

    return f"""
    <style>
        {formatter.get_style_defs()}
        .json-container {{
            white-space: pre-wrap; /* Preserve formatting */
            word-wrap: break-word; /* Break long words/lines */
            overflow-y: auto; /* Only vertical scrolling */       
            overflow-x: auto; /* Vertical scrolling */     
            max-height: 500px; /* Set a maximum height with vertical scroll */
            padding: 10px;
            border: 1px solid #ddd;
            background-color: #f8f8f8;
        }}
    </style>
    <div class="json-container">{highlighted}</div>
    """


def gradio_handler(question, temperature, enable_rag, enable_bm25):
    context = ""
    chunks_info = ["No context available"]
    if enable_rag:
        chunked_df = pd.read_csv("chunked_articles.csv")
        articles_df = pd.read_csv("dataframe.csv")

        chunks = chunked_df["Chunk"].tolist()

        embeddings = np.load("embeddings.npy")
        context, chunks_info = retrieve_context(question, embeddings, chunks, is_retrieve_bm25=enable_bm25)
        for item in chunks_info:
            article_id = int(chunked_df.loc[item["chunk_id"], "ID"])
            row = articles_df[articles_df['ID'] == article_id]
            item["article id:"] = article_id
            item["Article title"] = row["Title"].values[0]
            item["Article URL"] = row["URL"].values[0]

        print("chunks info:", chunks_info)

    return invoke_model(question, context, temperature), display_chunks(chunks_info)

def main():
    recreate_embeddings = False
    chunk_size = 200

    input_csv = 'dataframe.csv'
    df, chunks = process_and_split_csv(input_csv, chunk_size=chunk_size)

    if recreate_embeddings:
        embeddings = embed_chunks(chunks)
        np.save("embeddings.npy", embeddings)

    demo = gr.Interface(
        flagging_mode="never",
        fn=gradio_handler,
        inputs=[
            "text",
            gr.Slider(0, 1, value=0.7, step=0.1, label="Temperature", info="Choose between 0 and 1"),
            gr.Checkbox(label="Enable rag", info="Check whether to use rag (retrieval augmented generation)",
                        value=True),
            gr.Checkbox(label="Enable BM25", info="Use BM25 retriever (semantic is default)",
                        value=False),
        ],
        outputs=[
            gr.TextArea(label="output"),
            gr.HTML(label="chunks"),
        ],
    )
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
