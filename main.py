from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from google import genai
from google.genai import types
import os
import sys
import traceback
from io import StringIO

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class CodeRequest(BaseModel):
    code: str


# Response model
class CodeResponse(BaseModel):
    error: List[int]
    result: str


# AI structured output model
class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Execute the code
        exec(code)

        # Capture printed output
        output = sys.stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        # Get exact traceback
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    """
    Use Gemini AI to identify exact error line numbers.
    """

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY")
    )

    prompt = f"""
Analyze this Python code and its traceback.

Identify the exact line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{traceback_text}

Return ONLY the line numbers where the error exists.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.INTEGER
                        )
                    )
                },
                required=["error_lines"]
            )
        )
    )

    result = ErrorAnalysis.model_validate_json(response.text)

    return result.error_lines


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    """
    Execute Python code and analyze errors with AI.
    """

    execution_result = execute_python_code(request.code)

    # Successful execution
    if execution_result["success"]:
        return CodeResponse(
            error=[],
            result=execution_result["output"]
        )

    # Error occurred -> use AI analysis
    error_lines = analyze_error_with_ai(
        request.code,
        execution_result["output"]
    )

    return CodeResponse(
        error=error_lines,
        result=execution_result["output"]
    )


