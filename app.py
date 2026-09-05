"""
TeachMate AI — Streamlit deployment
LangChain + ChatGroq + RAG + Chroma + Pydantic
"""

import os
import re
import json
import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional

import streamlit as st
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

st.set_page_config(
    page_title="TeachMate AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(r"""
<style>
.stApp { background:#1c1510; color:#eadccf; }
[data-testid="stSidebar"] { background:#241b15; border-right:1px solid #4b3324; }
[data-testid="stSidebar"] * { color:#e6d5c3 !important; }
.hero {
    background:linear-gradient(135deg,#120e0b,#241811 50%,#4a2d1d);
    border:1px solid #5a3a26; border-radius:22px; padding:36px 30px;
    margin-bottom:22px; box-shadow:0 12px 30px rgba(0,0,0,.35); text-align:center;
}
.hero h1 { color:#fff7ed; margin:0 0 8px 0; font-size:42px; font-weight:800; }
.hero h3 { color:#d8ad83; margin:0 0 8px 0; font-weight:500; }
.hero p { color:#c4ad98; margin:0; }
div[data-testid="stTabs"] button { color:#cdb9a7; font-weight:600; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#e0ae7c; }
.output-card {
    background:#241b15; border:1px solid #533927; border-radius:16px;
    padding:24px; color:#e6d5c3; line-height:1.75;
}
.output-card h1,.output-card h2,.output-card h3 { color:#d8a979; }
.urdu-output {
    direction:rtl; text-align:right;
    font-family:"Noto Nastaliq Urdu","Noto Sans Arabic","Segoe UI",sans-serif;
}
.stButton > button {
    background:linear-gradient(135deg,#6b432c,#91603f);
    color:white; border:1px solid #a47753; border-radius:10px;
    font-weight:700; min-height:45px;
}
.stButton > button:hover { background:linear-gradient(135deg,#805333,#a36c46); color:white; }
[data-testid="stFileUploader"] {
    background:#2c2119; border:1px dashed #76543c; border-radius:12px; padding:10px;
}
.footer { text-align:center; color:#9f8874; padding:28px 0 8px 0; }
.small-note { color:#a99381; font-size:13px; }
</style>
""", unsafe_allow_html=True)

GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = Path("./teachmate_chroma")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
if not groq_api_key:
    st.error("GROQ_API_KEY is not configured. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

@st.cache_resource
def get_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        max_retries=2,
        api_key=groq_api_key,
    )

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name="teachmate_curriculum",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

llm = get_llm()
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})



class LessonPlan(BaseModel):
    class_name: str
    subject: str
    chapter_unit: str
    topic: str
    duration: str
    student_learning_outcomes: List[str]
    resources_materials: List[str]
    teaching_learning_methods: List[str]
    motivation_warmup: str
    methodology_procedure: List[str]
    teacher_will_explain: str
    summary: str
    assessment_classwork: str
    assignment_home_task: str

class WorksheetQuestion(BaseModel):
    question: str
    question_type: str
    marks: Optional[int] = None

class Worksheet(BaseModel):
    title: str
    grade: str
    subject: str
    topic: str
    instructions: str
    questions: List[WorksheetQuestion]
    answer_key: List[str]

class AssessmentQuestion(BaseModel):
    question: str
    question_type: str
    marks: int
    cognitive_level: Optional[str] = None

class Assessment(BaseModel):
    title: str
    grade: str
    subject: str
    topic: str
    total_marks: int
    questions: List[AssessmentQuestion]
    answer_key: List[str]
    marking_scheme: List[str]

class DifferentiatedLearning(BaseModel):
    struggling_learners: List[str]
    average_learners: List[str]
    advanced_learners: List[str]

class RemedialPlan(BaseModel):
    identified_learning_gap: str
    simplified_explanation: str
    remedial_strategy: List[str]
    activities: List[str]
    practice_questions: List[str]
    assessment: List[str]
    follow_up: str

class LearningObjectives(BaseModel):
    remember: List[str]
    understand: List[str]
    apply: List[str]
    analyze: List[str]
    evaluate: List[str]
    create: List[str]



def load_document(file_path):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(file_path).load()
    if suffix == ".docx":
        return UnstructuredWordDocumentLoader(file_path).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(file_path, encoding="utf-8").load()
    raise ValueError("Supported files: PDF, DOCX, TXT, MD")

def process_curriculum(file_path):
    if not file_path:
        raise ValueError("Upload a curriculum document first.")
    docs = load_document(file_path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=120
    ).split_documents(docs)
    vectorstore.add_documents(chunks)
    return (
        f"Curriculum processed successfully. "
        f"Source sections: {len(docs)} | Searchable chunks: {len(chunks)}"
    )


@st.cache_resource(show_spinner=False)
def build_uploaded_retriever(file_hash, file_bytes, file_name):
    """Build an isolated in-memory Chroma retriever for one uploaded file."""
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise ValueError("Supported files: PDF, DOCX, TXT, MD")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        docs = load_document(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    if not docs:
        raise ValueError("No readable content was found in the uploaded file.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=120
    ).split_documents(docs)

    if not chunks:
        raise ValueError("The uploaded file could not be divided into searchable chunks.")

    collection_name = f"teachmate_upload_{file_hash[:24]}"
    uploaded_store = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        collection_name=collection_name
    )

    return (
        uploaded_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
        len(docs),
        len(chunks)
    )


def get_uploaded_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def retrieve_uploaded_context(uploaded_retriever, query):
    """Retrieve context only from the file uploaded in the AI Assistant."""
    try:
        docs = uploaded_retriever.invoke(query)
    except Exception:
        return "NO_RELEVANT_DOCUMENT_CONTEXT_FOUND"

    if not docs:
        return "NO_RELEVANT_DOCUMENT_CONTEXT_FOUND"

    return "\n\n".join(
        f"[Document Source {i}]\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )

def retrieve_context(query):
    try:
        docs = retriever.invoke(query)
    except Exception:
        return "NO_RELEVANT_CURRICULUM_CONTEXT_FOUND"
    if not docs:
        return "NO_RELEVANT_CURRICULUM_CONTEXT_FOUND"
    return "\n\n".join(
        f"[Source {i}: {d.metadata.get('source', 'uploaded curriculum')}]\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )



RULES = """
You are TeachMate AI, an educational teaching assistant.

Use supplied curriculum context for curriculum-specific information.

Do not invent curriculum-specific facts, learning outcomes,
textbook requirements, or syllabus details.

If requested curriculum information is absent, clearly state that
it is not available in the provided curriculum.

General pedagogical suggestions are allowed but must not be
presented as official curriculum facts.

Adapt material to the selected grade and subject.

LANGUAGE RULES:

- If Subject is "Urdu", generate the ENTIRE educational response in
  natural Urdu using Urdu script.

- If Subject is "Islamic/Quranic Education", "Islamiyat", "Islamiat",
  or "Islamic Studies", generate the ENTIRE educational response in
  natural Urdu using Urdu script unless the teacher explicitly asks
  for English.

- For all other subjects, use English unless the teacher explicitly
  requests another language.

- Do not write Urdu using Roman Urdu.

- Preserve standard Arabic Quranic text, duas, Surah names, Hadith text,
  or other Arabic religious expressions in Arabic where appropriate.

- Explain Arabic religious text in Urdu when the subject is
  Islamic/Quranic Education.

- Never fabricate Quranic verses, Surah/Ayah references, Hadith references,
  translations, or religious quotations.
"""


def prompt(instructions):
    return ChatPromptTemplate.from_messages([
        ('system', RULES + '\n\n' + instructions),
        ('human', 'Grade: {grade}\nSubject: {subject}\nTopic: {topic}\nCurriculum context:\n{context}\n\nUser instructions: {user_instructions}')
    ])

LESSON_PLAN_PROMPT = """
You are TeachMate AI, an expert school lesson-plan designer.

Create a professional, classroom-ready lesson plan using the teacher's
provided information and, when available, the retrieved curriculum context.

The lesson plan must follow the school's required lesson-plan structure.

LESSON INFORMATION:
Class: {grade}
Subject: {subject}
Chapter/Unit: {chapter_unit}
Topic: {topic}
Sub-topic: {subtopic}
Duration: {duration}
Difficulty: {difficulty}
Number of Students: {students}
Additional Instructions: {additional_instructions}

CURRICULUM CONTEXT:
{context}


REQUIRED LESSON PLAN STRUCTURE:

1. Class

2. Subject

3. Chapter/Unit

4. Topic

5. Duration


6. Student Learning Outcomes

Generate an appropriate number of clear, measurable learning outcomes
for the selected grade, subject and topic.

Do NOT force the number of outcomes to a fixed number.

Use measurable action verbs where appropriate, such as:
- identify
- explain
- describe
- differentiate
- solve
- demonstrate
- apply
- compare
- analyze
- evaluate
- create

The outcomes must be realistic and achievable within the lesson.


7. Resources and Materials

List the resources and classroom materials actually needed for the lesson.

Do not add unnecessary resources.


8. Teaching and Learning Methods:
Instructional Technique(s)

Select appropriate teaching and learning methods based on the
subject, topic and grade.

Possible methods include:
- direct instruction
- questioning
- discussion
- demonstration
- guided practice
- collaborative learning
- activity-based learning
- problem-solving
- inquiry-based learning
- brainstorming
- peer learning

Use only the methods that are appropriate for the lesson.


9. Motivation / Warm-up

Create a short, engaging activity that introduces or activates
prior knowledge related to the topic.

The activity should be appropriate for the selected grade.


10. Methodology / Procedure

Provide a logical, classroom-ready sequence of teaching steps.

Clearly explain how the lesson should progress from introduction
to teaching, student participation and practice.

The procedure should be practical enough for a teacher to follow
directly in the classroom.


11. Teacher Will Explain

Clearly describe what the teacher will explain during the lesson.

Include relevant:
- concepts
- definitions
- examples
- demonstrations
- rules
- formulas
- vocabulary
- explanations

Adapt the explanation to the selected grade and difficulty level.


12. Summary

Provide a short recap of the most important learning points.

Include a suitable way for the teacher to check whether students
understood the main idea.


13. Assessment / Classwork

Create suitable classwork or assessment activities that check
whether students achieved the learning outcomes.

The assessment should be appropriate for the grade, subject,
topic and difficulty level.


14. Assignment / Home Task

Provide a meaningful homework or home task that reinforces the
lesson.

The task should be appropriate for the selected grade and topic.


IMPORTANT RULES:

- Follow the required lesson-plan structure.
- Do not force a fixed number of learning outcomes.
- Learning outcomes should be measurable and age-appropriate.
- Keep all activities appropriate for the selected grade.
- Keep the difficulty appropriate for the selected difficulty level.
- Respect the teacher's requested lesson duration.
- Make the lesson practical for a real classroom.
- Use clear and professional language suitable for teachers.
- Adapt examples, activities and explanations to the subject and topic.
- Do not add unrelated sections.
- Do not invent curriculum-specific facts.
- When curriculum context is provided, use it for curriculum-specific
  information.
- If requested curriculum information is not present in the supplied
  curriculum context, do not present invented information as a
  curriculum fact.
- General pedagogical suggestions are allowed when clearly presented
  as general teaching suggestions.
- Do not diagnose learning disabilities, medical conditions or
  psychological conditions.
- If the subject is Urdu, write the entire lesson plan in Urdu script.
- If the subject is Islamic/Quranic Education, Islamiyat, Islamiat,
  or Islamic Studies, write the entire lesson plan in Urdu script
  unless the teacher explicitly requests English.
- For other subjects, use English unless another language is requested.
- Do not use Roman Urdu.
- For Islamic/Quranic Education, preserve Quranic Arabic and other
  authentic Arabic religious text in Arabic where appropriate,
  while explanations should be in Urdu.
- Never fabricate Quranic verses, Ayah references, Hadith references,
  translations, or religious quotations.
"""
worksheet_prompt = prompt('Create an age-appropriate worksheet with the requested question types and an answer key.')
assessment_prompt = prompt('Create an assessment with the requested marks, question types, answer key and marking scheme. Use Bloom levels where useful.')
diff_prompt = prompt('Create differentiated learning material for struggling, average and advanced learners.')
remedial_prompt = prompt('Create an educational remedial plan without diagnosing disabilities or medical conditions.')
objectives_prompt = prompt('Create measurable, age-appropriate learning objectives using Bloom taxonomy where useful.')


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

def structured(schema):
    return llm.with_structured_output(
        schema,
        method="json_schema"
    )


# ============================================================
# LESSON PLAN GENERATOR
# ============================================================

def generate_lesson_plan(
    grade,
    subject,
    chapter_unit,
    topic,
    subtopic,
    duration,
    difficulty,
    students,
    additional_instructions
):

    # --------------------------------------------------------
    # Retrieve curriculum context
    # --------------------------------------------------------

    context = retrieve_context(
        f"{grade} {subject} {chapter_unit} {topic} {subtopic}"
    )


    # --------------------------------------------------------
    # Build lesson-plan prompt
    # --------------------------------------------------------

    lesson_prompt = LESSON_PLAN_PROMPT.format(

        grade=grade,

        subject=subject,

        chapter_unit=chapter_unit,

        topic=topic,

        subtopic=subtopic,

        duration=duration,

        difficulty=difficulty,

        students=students,

        additional_instructions=additional_instructions,

        context=context
    )


    # --------------------------------------------------------
    # Structured Pydantic output
    # --------------------------------------------------------

    structured_llm = structured(LessonPlan)


    # --------------------------------------------------------
    # Generate lesson plan
    # --------------------------------------------------------

    result = structured_llm.invoke(
        lesson_prompt
    )


    # --------------------------------------------------------
    # Format for Gradio
    # --------------------------------------------------------

    return format_lesson_plan(result)


# ============================================================
# WORKSHEET
# ============================================================

def generate_worksheet(
    grade,
    subject,
    topic,
    difficulty,
    count,
    types
):

    context = retrieve_context(
        f"{grade} {subject} {topic}"
    )

    return (
        worksheet_prompt
        | structured(Worksheet)
    ).invoke({

        "grade": grade,

        "subject": subject,

        "topic": topic,

        "context": context,

        "user_instructions":
            f"Difficulty: {difficulty}; "
            f"Questions: {count}; "
            f"Types: {types}"
    })


# ============================================================
# ASSESSMENT
# ============================================================

def generate_assessment(
    grade,
    subject,
    topic,
    marks,
    difficulty,
    types
):

    context = retrieve_context(
        f"{grade} {subject} {topic}"
    )

    return (
        assessment_prompt
        | structured(Assessment)
    ).invoke({

        "grade": grade,

        "subject": subject,

        "topic": topic,

        "context": context,

        "user_instructions":
            f"Marks: {marks}; "
            f"Difficulty: {difficulty}; "
            f"Types: {types}"
    })


# ============================================================
# DIFFERENTIATED LEARNING
# ============================================================

def generate_differentiated(
    grade,
    subject,
    topic
):

    return (
        diff_prompt
        | structured(DifferentiatedLearning)
    ).invoke({

        "grade": grade,

        "subject": subject,

        "topic": topic,

        "context":
            retrieve_context(
                f"{grade} {subject} {topic}"
            ),

        "user_instructions":
            "Generate appropriate differentiated "
            "learning material for struggling, "
            "average and advanced learners."
    })


# ============================================================
# REMEDIAL LEARNING
# ============================================================

def generate_remedial(
    grade,
    subject,
    topic,
    difficulty
):

    return (
        remedial_prompt
        | structured(RemedialPlan)
    ).invoke({

        "grade": grade,

        "subject": subject,

        "topic": topic,

        "context":
            retrieve_context(
                f"{grade} {subject} {topic}"
            ),

        "user_instructions":
            f"Learning difficulty: {difficulty}"
    })


# ============================================================
# LEARNING OBJECTIVES
# ============================================================

def generate_objectives(
    grade,
    subject,
    topic
):

    return (
        objectives_prompt
        | structured(LearningObjectives)
    ).invoke({

        "grade": grade,

        "subject": subject,

        "topic": topic,

        "context":
            retrieve_context(
                f"{grade} {subject} {topic}"
            ),

        "user_instructions":
            "Generate useful, measurable and "
            "age-appropriate objectives."
    })


# ============================================================
# GENERIC RENDER FUNCTION
# ============================================================

def format_worksheet(worksheet):
    """Format a Worksheet as a clean, teacher-ready worksheet."""

    lines = []

    # Title
    lines.append(f"# 📝 {worksheet.title}")
    lines.append("")

    # Basic information
    lines.append(
        f"**Grade:** {worksheet.grade}  \n"
        f"**Subject:** {worksheet.subject}  \n"
        f"**Topic:** {worksheet.topic}"
    )
    lines.append("")

    # Student name / date
    lines.append(
        "**Student Name:** ________________________________  \n"
        "**Date:** ____________________"
    )
    lines.append("")

    # Instructions
    lines.append("## 📌 Instructions")
    lines.append("")
    lines.append(worksheet.instructions)
    lines.append("")

    # Questions
    lines.append("## ✏️ Questions")
    lines.append("")

    for i, q in enumerate(worksheet.questions, 1):

        question = str(q.question).strip()
        question_type = str(q.question_type).strip().lower()

        # Remove accidental numbering from AI output
        question = re.sub(
            r"^\s*(?:question\s*)?\d+[\.\):\-]\s*",
            "",
            question,
            flags=re.IGNORECASE
        )

        # Remove escaped newlines
        question = question.replace("\\n", "\n")

        # Question heading
        lines.append(f"### Q{i}. {question}")

        # Marks
        if q.marks is not None:
            lines.append(f"**[{q.marks} Mark{'s' if q.marks != 1 else ''}]**")

        lines.append("")

        # MCQ formatting
        if "mcq" in question_type or "multiple choice" in question_type:

            # Try to detect options already generated by the model
            if "\\n" in question:
                options = question.split("\\n")
                for option in options:
                    option = option.strip()
                    if option:
                        lines.append(f"- {option}")
                lines.append("")

            else:
                lines.append(
                    "☐ A) ____________________    "
                    "☐ B) ____________________    "
                    "☐ C) ____________________    "
                    "☐ D) ____________________"
                )
                lines.append("")

        # True / False
        elif "true" in question_type and "false" in question_type:
            lines.append("☐ True        ☐ False")
            lines.append("")

        # Fill in the blank
        elif "fill" in question_type or "blank" in question_type:
            lines.append("")
            lines.append("Answer: __________________________________________")
            lines.append("")

        # Short / long / problem-solving questions
        else:
            lines.append("")
            lines.append("Answer:")
            lines.append("")
            lines.append("____________________________________________________________")
            lines.append("")
            lines.append("____________________________________________________________")
            lines.append("")

    # Answer key
    lines.append("---")
    lines.append("")
    lines.append("## 🔑 Answer Key")
    lines.append("")

    for i, answer in enumerate(worksheet.answer_key, 1):

        answer = str(answer).strip()

        # Remove accidental numbering from answer
        answer = re.sub(
            r"^\s*(?:Q(?:uestion)?\s*)?\d+[\.\):\-]\s*",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = answer.replace("\\n", "\n")

        lines.append(f"**Q{i}.** {answer}")
        lines.append("")

    return "\n".join(lines)

def render(obj):

    # If already formatted as Markdown/text
    if isinstance(obj, str):
        return obj

    # Pydantic object
    if isinstance(obj, BaseModel):
        data = obj.model_dump()

    # Dictionary
    elif isinstance(obj, dict):
        data = obj

    else:
        return str(obj)

    out = []

    for key, value in data.items():

        out.append(
            f"## {key.replace('_', ' ').title()}"
        )

        if isinstance(value, list):

            for item in value:
                out.append(f"- {item}")

        else:

            out.append(str(value))

        out.append("")

    return "\n".join(out)


# ============================================================
# SAFE EXECUTION
# ============================================================


def safe(fn, *args):

    try:
        result = fn(*args)
        return render(result)

    except Exception as e:

        return (
            f"**Error:** `{type(e).__name__}: {e}`"
        )




# ============================================================
# LESSON PLAN FORMATTER
# ============================================================

def format_lesson_plan(plan: LessonPlan):

    subject_lower = plan.subject.lower().strip()

    urdu_mode = (
        subject_lower == "urdu"
        or "islamic" in subject_lower
        or "quranic" in subject_lower
        or "islamiat" in subject_lower
        or "islamiyat" in subject_lower
    )

    outcomes = "\n\n".join(
        f"{i + 1}. {re.sub(r'^\s*\d+[\.\)]\s*', '', str(outcome))}"
        for i, outcome in enumerate(
            plan.student_learning_outcomes
        )
    )

    resources = "\n".join(
        f"- {item}"
        for item in plan.resources_materials
    )

    methods = "\n".join(
        f"- {item}"
        for item in plan.teaching_learning_methods
    )

    procedure = "\n".join(
        f"{i + 1}. {re.sub(r'^\s*\d+[\.\)]\s*', '', str(step))}"
        for i, step in enumerate(
            plan.methodology_procedure
        )
    )


    # ========================================================
    # URDU / ISLAMIAT VERSION
    # ========================================================

    if urdu_mode:

        return f"""
# جماعت: {plan.class_name}

**مضمون:** {plan.subject}

# باب / یونٹ: {plan.chapter_unit}

# موضوع: {plan.topic}

**دورانیہ:** {plan.duration}


## 1. طلبہ کے تعلیمی نتائج

اس سبق کے اختتام تک طلبہ اس قابل ہوں گے کہ:

{outcomes}


## 2. وسائل اور تدریسی مواد

{resources}


## 3. تدریسی و تعلیمی طریقے

### تدریسی حکمتِ عملی

{methods}


## 4. ترغیب / ابتدائی سرگرمی

{plan.motivation_warmup}


## 5. طریقۂ تدریس / سبق کی کارروائی

{procedure}


### استاد وضاحت کرے گا:

{plan.teacher_will_explain}


## 6. خلاصہ

{plan.summary}


## 7. جائزہ / جماعتی کام

{plan.assessment_classwork}


## 8. گھر کا کام

{plan.assignment_home_task}
"""


    # ========================================================
    # ENGLISH VERSION
    # ========================================================

    return f"""
# Class: {plan.class_name}

**Subject:** {plan.subject}

# Chapter/Unit: {plan.chapter_unit}

# Topic: {plan.topic}

**Duration:** {plan.duration}


## 1. Student Learning Outcomes

By the end of this lesson, students will be able to:

{outcomes}


## 2. Resources and Materials

{resources}


## 3. Teaching and Learning Methods

### Instructional Technique(s)

{methods}


## 4. Motivation / Warm-up

{plan.motivation_warmup}


## 5. Methodology / Procedure

{procedure}


### Teacher Will Explain:

{plan.teacher_will_explain}


## 6. Summary

{plan.summary}


## 7. Assessment / Classwork

{plan.assessment_classwork}


## 8. Assignment / Home Task

{plan.assignment_home_task}
"""

assistant_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are TeachMate AI, a professional classroom teaching assistant.

Help teachers with explanations, classroom activities, examples, questioning strategies,
teaching methods, classroom resources and pedagogical suggestions.
Adapt every response to the selected grade and subject.

DOCUMENT RULES:
- If uploaded document context is supplied, use that document as the primary source
  for document-specific questions.
- Do not invent facts that are claimed to come from the uploaded document.
- If requested information is not present in the uploaded document, clearly say that
  the document does not provide that information.
- General pedagogical suggestions are allowed, but do not present them as facts from
  the uploaded document.
- If no document context is supplied, answer normally using your general educational knowledge.

LANGUAGE RULES:
- If Subject is Urdu, respond completely in natural Urdu script. Never use Roman Urdu.
- If Subject is Islamic/Quranic Education, Islamiyat, Islamiat, or Islamic Studies,
  respond completely in Urdu script unless the teacher explicitly requests English.
- Preserve authentic Arabic Quranic or religious text in Arabic where appropriate.
- Never invent Quranic verses, Ayah references, Hadith references, translations,
  or religious quotations.
- For all other subjects, respond in English unless the teacher requests another language.

Be practical, clear, accurate and teacher-friendly."""
    ),
    (
        "human",
        """Grade: {grade}
Subject: {subject}

Teacher's Question:
{question}

Uploaded Document Context:
{context}"""
    )
])


def teaching_assistant(grade, subject, question, uploaded_retriever=None):
    if not question.strip():
        return "Please enter a question."

    if uploaded_retriever is not None:
        context = retrieve_uploaded_context(
            uploaded_retriever,
            f"{grade} {subject} {question}"
        )
    else:
        context = "NO DOCUMENT UPLOADED. Answer using general educational knowledge."

    response = (assistant_prompt | llm).invoke({
        "grade": grade,
        "subject": subject,
        "question": question,
        "context": context
    })

    return response.content


def format_assessment(assessment):
    """Format an Assessment as a clean, professional question paper."""

    lines = []

    # ------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------

    lines.append(f"# 📊 {assessment.title}")
    lines.append("")

    lines.append(
        f"**Grade:** {assessment.grade}  \n"
        f"**Subject:** {assessment.subject}  \n"
        f"**Topic:** {assessment.topic}  \n"
        f"**Total Marks:** {assessment.total_marks}"
    )

    lines.append("")

    lines.append(
        "**Student Name:** __________________________________________  \n"
        "**Roll No.:** ____________________    "
        "**Date:** ____________________"
    )

    lines.append("")
    lines.append("---")
    lines.append("")

    # ------------------------------------------------------------
    # QUESTION PAPER
    # ------------------------------------------------------------

    lines.append("## 📝 Question Paper")
    lines.append("")

    for i, q in enumerate(assessment.questions, 1):

        question = str(q.question).strip()
        question_type = str(q.question_type).strip().lower()

        # Remove numbering generated by the AI
        question = re.sub(
            r"^\s*(?:question\s*)?\d+[\.\):\-]\s*",
            "",
            question,
            flags=re.IGNORECASE
        )

        # Convert escaped newlines into real newlines
        question = question.replace("\\n", "\n")

        # --------------------------------------------------------
        # QUESTION
        # --------------------------------------------------------

        option_lines = question.split("\n")

        if "mcq" in question_type or "multiple choice" in question_type:

            # First line is the actual question
            question_text = option_lines[0].strip()

            lines.append(f"### Q{i}. {question_text}")

            if q.marks is not None:
                lines.append(
                    f"**[{q.marks} Mark{'s' if q.marks != 1 else ''}]**"
                )

            lines.append("")

            # Display options neatly
            for option in option_lines[1:]:
                option = option.strip()

                if option:
                    lines.append(f"- {option}")

            lines.append("")

            lines.append("**Answer:** ____________________")
            lines.append("")

        elif "true" in question_type and "false" in question_type:

            lines.append(f"### Q{i}. {question}")

            if q.marks is not None:
                lines.append(
                    f"**[{q.marks} Mark{'s' if q.marks != 1 else ''}]**"
                )

            lines.append("")
            lines.append("☐ True        ☐ False")
            lines.append("")

        elif "fill" in question_type or "blank" in question_type:

            lines.append(f"### Q{i}. {question}")

            if q.marks is not None:
                lines.append(
                    f"**[{q.marks} Mark{'s' if q.marks != 1 else ''}]**"
                )

            lines.append("")
            lines.append("**Answer:** __________________________________________")
            lines.append("")

        elif "short" in question_type:

            lines.append(f"### Q{i}. {question}")

            if q.marks is not None:
                lines.append(
                    f"**[{q.marks} Mark{'s' if q.marks != 1 else ''}]**"
                )

            lines.append("")
            lines.append("**Answer:**")
            lines.append("")
            lines.append(
                "____________________________________________________________"
            )
            lines.append("")
            lines.append(
                "____________________________________________________________"
            )
            lines.append("")

        elif "long" in question_type:

            lines.append(f"### Q{i}. {question}")

            if q.marks is not None:
                lines.append(
                    f"**[{q.marks} Mark{'s' if q.marks != 1 else ''}]**"
                )

            lines.append("")
            lines.append("**Answer:**")
            lines.append("")

            for _ in range(5):
                lines.append(
                    "____________________________________________________________"
                )
                lines.append("")

        else:

            lines.append(f"### Q{i}. {question}")

            if q.marks is not None:
                lines.append(
                    f"**[{q.marks} Mark{'s' if q.marks != 1 else ''}]**"
                )

            lines.append("")
            lines.append("**Working / Answer:**")
            lines.append("")

            for _ in range(3):
                lines.append(
                    "____________________________________________________________"
                )
                lines.append("")

    # ------------------------------------------------------------
    # ANSWER KEY
    # ------------------------------------------------------------

    lines.append("---")
    lines.append("")
    lines.append("## 🔑 Answer Key")
    lines.append("")

    for i, answer in enumerate(assessment.answer_key, 1):

        answer = str(answer).strip()

        answer = re.sub(
            r"^\s*(?:question\s*)?\d+[\.\):\-]\s*",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = answer.replace("\\n", "\n")

        lines.append(f"**Q{i}.** {answer}")
        lines.append("")

    # ------------------------------------------------------------
    # MARKING SCHEME
    # ------------------------------------------------------------

    lines.append("---")
    lines.append("")
    lines.append("## 📋 Marking Scheme")
    lines.append("")

    for i, item in enumerate(assessment.marking_scheme, 1):

        item = str(item).strip()

        item = re.sub(
            r"^\s*\d+[\.\):\-]\s*",
            "",
            item
        )

        lines.append(f"**{i}.** {item}")
        lines.append("")

    return "\n".join(lines)

SUBJECTS = [
    "English", "Urdu", "Mathematics", "Science", "History",
    "Geography", "Computer Science", "Islamiyat"
]
GRADES = [f"Grade {i}" for i in range(1, 13)]
DIFFICULTIES = ["Easy", "Medium", "Difficult", "Mixed"]

def is_urdu_subject(subject):
    s = subject.lower().strip()
    return (
        s == "urdu" or "islamic" in s or "quranic" in s
        or "islamiat" in s or "islamiyat" in s
    )

def display_output(text, subject=""):
    if not text:
        return
    cls = "output-card urdu-output" if is_urdu_subject(subject) else "output-card"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def run_safely(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        return None



st.markdown("""
<div class="hero">
    <h1>🎓 TeachMate AI</h1>
    <h3>Your Intelligent Teaching Companion</h3>
    <p>Curriculum-aware GenAI support for teachers from Grade 1 to Grade 12 🇵🇰</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎓 TeachMate AI")
    st.markdown('<p class="small-note">AI-powered educational assistant</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📌 Project")
    st.markdown("**Subjects:** 8  \n**Grades:** 1–12  \n**RAG:** Chroma  \n**LLM:** ChatGroq  \n**Framework:** LangChain")
    st.markdown("---")
    st.markdown("### 💡 AI Assistant")
    st.markdown(
        "Upload a document inside the **AI Assistant** when you want TeachMate "
        "to answer from your own material. Without a document, it answers normally "
        "using its general educational knowledge."
    )

tabs = st.tabs([
    "📚 Lesson Planner", "📝 Worksheet", "📊 Assessment",
    "🎯 Differentiated", "🔄 Remedial", "🎯 Objectives", "🤖 AI Assistant"
])

with tabs[0]:
    st.subheader("📚 Generate a Complete Lesson Plan")
    st.caption("Create a professional, classroom-ready lesson plan using your selected grade, subject and curriculum.")

    c1, c2, c3 = st.columns(3)
    with c1:
        lp_grade = st.selectbox("Grade", GRADES, index=4, key="lp_grade")
    with c2:
        lp_subject = st.selectbox("Subject", SUBJECTS, index=2, key="lp_subject")
    with c3:
        lp_duration = st.text_input("Duration", "40 minutes", key="lp_duration")

    c1, c2 = st.columns(2)
    with c1:
        lp_chapter = st.text_input("Chapter / Unit", placeholder="e.g. Unit 3: Algebraic Expressions", key="lp_chapter")
    with c2:
        lp_topic = st.text_input("Topic", placeholder="e.g. Indices", key="lp_topic")

    lp_subtopic = st.text_input("Sub-topic (optional)", placeholder="e.g. Laws of Indices", key="lp_subtopic")

    c1, c2 = st.columns(2)
    with c1:
        lp_difficulty = st.selectbox("Difficulty", DIFFICULTIES, index=1, key="lp_difficulty")
    with c2:
        lp_students = st.number_input("Number of Students", min_value=1, value=30, step=1, key="lp_students")

    lp_extra = st.text_area("Additional Instructions", placeholder="Any special requirements...", key="lp_extra")

    if st.button("✨ Generate Lesson Plan", type="primary", use_container_width=True, key="generate_lp"):
        if not lp_topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("TeachMate is creating your lesson plan..."):
                result = run_safely(
                    generate_lesson_plan, lp_grade, lp_subject, lp_chapter,
                    lp_topic, lp_subtopic, lp_duration, lp_difficulty,
                    lp_students, lp_extra
                )
            if result:
                display_output(result, lp_subject)

with tabs[1]:
    st.subheader("📝 Worksheet Generator")
    c1, c2 = st.columns(2)
    with c1:
        w_grade = st.selectbox("Grade", GRADES, index=4, key="w_grade")
        w_subject = st.selectbox("Subject", SUBJECTS, index=2, key="w_subject")
        w_topic = st.text_input("Topic", key="w_topic")
    with c2:
        w_difficulty = st.selectbox("Difficulty", DIFFICULTIES, index=1, key="w_difficulty")
        w_count = st.number_input("Number of Questions", min_value=1, value=10, step=1, key="w_count")
        w_types = st.text_input("Question Types", "MCQ, short questions, fill in blanks", key="w_types")

    if st.button("✨ Generate Worksheet", type="primary", use_container_width=True, key="generate_ws"):
        if not w_topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Creating worksheet..."):
                result = run_safely(
                    generate_worksheet,
                    w_grade,
                    w_subject,
                    w_topic,
                    w_difficulty,
                    int(w_count),
                    w_types
                )
    
            if result:
                display_output(format_worksheet(result), w_subject)

with tabs[2]:
    st.subheader("📊 Assessment Generator")
    c1, c2 = st.columns(2)
    with c1:
        a_grade = st.selectbox("Grade", GRADES, index=6, key="a_grade")
        a_subject = st.selectbox("Subject", SUBJECTS, index=3, key="a_subject")
        a_topic = st.text_input("Topic", key="a_topic")
    with c2:
        a_marks = st.number_input("Total Marks", min_value=1, value=25, step=1, key="a_marks")
        a_difficulty = st.selectbox("Difficulty", DIFFICULTIES, index=1, key="a_difficulty")
        a_types = st.text_input("Question Types", "MCQ, short questions, long questions", key="a_types")

    if st.button(
        "✨ Generate Assessment",
        type="primary",
        use_container_width=True,
        key="generate_assessment"
    ):
        if not a_topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Creating assessment..."):
                try:
                    result = generate_assessment(
                        a_grade,
                        a_subject,
                        a_topic,
                        int(a_marks),
                        a_difficulty,
                        a_types
                    )
    
                    display_output(
                        format_assessment(result),
                        a_subject
                    )
    
                except Exception as e:
                    st.error(
                        f"Assessment generation failed: "
                        f"{type(e).__name__}: {e}"
                    )

with tabs[3]:
    st.subheader("🎯 Differentiated Learning")
    c1, c2 = st.columns(2)
    with c1:
        dl_grade = st.selectbox("Grade", GRADES, index=5, key="dl_grade")
    with c2:
        dl_subject = st.selectbox("Subject", SUBJECTS, index=3, key="dl_subject")
    dl_topic = st.text_input("Topic", key="dl_topic")

    if st.button("✨ Generate Differentiated Learning", type="primary", use_container_width=True, key="generate_dl"):
        if not dl_topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Creating differentiated material..."):
                result = run_safely(generate_differentiated, dl_grade, dl_subject, dl_topic)
            if result:
                display_output(render(result), dl_subject)

with tabs[4]:
    st.subheader("🔄 Remedial Learning")
    c1, c2 = st.columns(2)
    with c1:
        rl_grade = st.selectbox("Grade", GRADES, index=4, key="rl_grade")
        rl_subject = st.selectbox("Subject", SUBJECTS, index=2, key="rl_subject")
    with c2:
        rl_topic = st.text_input("Topic", key="rl_topic")
        rl_difficulty = st.text_input("Learning Difficulty", placeholder="e.g. struggles with fractions", key="rl_difficulty")

    if st.button("✨ Generate Remedial Plan", type="primary", use_container_width=True, key="generate_rl"):
        if not rl_topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Creating remedial plan..."):
                result = run_safely(generate_remedial, rl_grade, rl_subject, rl_topic, rl_difficulty)
            if result:
                display_output(render(result), rl_subject)

with tabs[5]:
    st.subheader("🎯 Learning Objectives")
    c1, c2 = st.columns(2)
    with c1:
        lo_grade = st.selectbox("Grade", GRADES, index=4, key="lo_grade")
    with c2:
        lo_subject = st.selectbox("Subject", SUBJECTS, index=2, key="lo_subject")
    lo_topic = st.text_input("Topic", key="lo_topic")

    if st.button("✨ Generate Objectives", type="primary", use_container_width=True, key="generate_lo"):
        if not lo_topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Creating learning objectives..."):
                result = run_safely(generate_objectives, lo_grade, lo_subject, lo_topic)
            if result:
                display_output(render(result), lo_subject)

with tabs[6]:
    st.subheader("🤖 AI Teaching Assistant")
    st.caption(
        "Ask TeachMate anything about teaching. Upload a document when you want "
        "answers grounded in your own curriculum, textbook, notes or teacher guide."
    )

    c1, c2 = st.columns(2)
    with c1:
        ta_grade = st.selectbox("Grade", GRADES, index=4, key="ta_grade")
    with c2:
        ta_subject = st.selectbox("Subject", SUBJECTS, index=2, key="ta_subject")

    st.markdown("### 📎 Optional Document")
    st.caption(
        "Upload a PDF, DOCX, TXT or Markdown file. If you do not upload a file, "
        "TeachMate will answer using its general educational knowledge."
    )

    ta_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "docx", "txt", "md"],
        key="ta_file",
        help="Examples: curriculum, syllabus, textbook chapter, lecture notes, teacher guide."
    )

    ta_uploaded_retriever = None
    ta_file_name = None

    if ta_file is not None:
        ta_file_bytes = ta_file.getvalue()
        ta_file_name = ta_file.name
        ta_file_hash = get_uploaded_file_hash(ta_file_bytes)

        try:
            with st.spinner("Reading and indexing your document..."):
                (
                    ta_uploaded_retriever,
                    ta_source_sections,
                    ta_searchable_chunks
                ) = build_uploaded_retriever(
                    ta_file_hash,
                    ta_file_bytes,
                    ta_file_name
                )

            st.success(
                f"📚 Document ready: **{ta_file_name}** • "
                f"{ta_searchable_chunks} searchable chunks"
            )
            st.caption(
                "TeachMate will use this uploaded document as the primary source "
                "for document-specific questions."
            )
        except Exception as e:
            st.error(
                f"Could not process the uploaded document: "
                f"{type(e).__name__}: {e}"
            )
            ta_uploaded_retriever = None

    st.markdown("### 💬 Ask TeachMate")
    ta_question = st.text_area(
        "Your Question",
        height=150,
        placeholder=(
            "Example: According to this document, what should Grade 5 students "
            "learn about fractions?"
        ),
        key="ta_question"
    )

    if st.button(
        "💬 Ask TeachMate",
        type="primary",
        use_container_width=True,
        key="ask_ta"
    ):
        if not ta_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("TeachMate is thinking..."):
                result = run_safely(
                    teaching_assistant,
                    ta_grade,
                    ta_subject,
                    ta_question,
                    ta_uploaded_retriever
                )

            if result:
                if ta_uploaded_retriever is not None and ta_file_name:
                    st.info(f"📚 Answer based on: **{ta_file_name}**")
                else:
                    st.info("🤖 Answered using TeachMate AI's general educational knowledge")
                display_output(result, ta_subject)


st.markdown("""
<div class="footer">
    🎓 <b style="color:#d8a979;">TeachMate AI</b><br>
    <span style="font-size:13px;">
        LangChain • ChatGroq • RAG • Chroma • Pydantic • Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
