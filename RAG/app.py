from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_pipeline import answer_query

app = FastAPI(title="RAG HTML App")

# Mount static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load templates
templates = Jinja2Templates(directory="templates")


# -----------------------------
# Home Page
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": None,
            "question": None,
            "sources": None
        }
    )


# -----------------------------
# Handle Form Submission
# -----------------------------
@app.post("/ask", response_class=HTMLResponse)
def ask_question(
    request: Request,
    query: str = Form(...)
):
    result = answer_query(query)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"]
        }
    )
