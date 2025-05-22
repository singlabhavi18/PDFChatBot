from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = None
chain = None

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global db, chain
    try:
        pdf = request.files['pdf']
        filename = secure_filename(pdf.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        pdf.save(path)

        # Load and split PDF
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(data)

        # Build vector store
        db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory="./chroma_db",
            collection_name="local-rag"
        )
        db.persist()

        # Setup LLM
        llm = ChatOllama(model="llama3")

        # Setup retriever with multi-query
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI assistant. Rephrase the following question to help retrieve more relevant documents:
Original question: {question}"""
        )

        retriever = MultiQueryRetriever.from_llm(
            db.as_retriever(),
            llm,
            prompt=prompt
        )

        # RAG prompt
        rag_prompt = ChatPromptTemplate.from_template("""Answer the question based ONLY on the context:
{context}
Question: {question}""")

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        # Summary generation
        full_text = "\n".join([chunk.page_content for chunk in chunks])
        summary_prompt = f"Give a short summary for the following:\n\n{full_text}"
        summary_response = llm.invoke(summary_prompt)
        summary = str(summary_response.content) if hasattr(summary_response, 'content') else str(summary_response)

        # Improved suggested question generation
        question_prompt = f"""Based on the following content, generate 5 questions that can be strictly answered using the content itself. Avoid generic or opinion-based questions.

Content:
{summary}

Only output the questions in a numbered list.
"""
        question_response = llm.invoke(question_prompt)
        raw_questions = str(question_response.content)

        # Parse into list (remove empty lines)
        questions = [line.strip() for line in raw_questions.split('\n') if line.strip() and any(c.isalpha() for c in line)]

        return jsonify({
            "message": "PDF uploaded and processed",
            "summary": summary,
            "suggested_questions": questions
        })

    except Exception as e:
        print("Error during upload:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    global chain
    question = request.json.get('question')
    if not chain:
        return jsonify({"answer": "No PDF processed yet."})
    answer = chain.invoke(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
