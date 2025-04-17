import os
import markdown
import PyPDF2
from openai import OpenAI
import json
from datetime import datetime

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', ''))
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure paths
SUMMARIES_DIR = os.path.join(os.path.dirname(__file__), 'summaries')
QUESTIONS_DIR = os.path.join(os.path.dirname(__file__), 'questions')

# Ensure directories exist
os.makedirs(SUMMARIES_DIR, exist_ok=True)
os.makedirs(QUESTIONS_DIR, exist_ok=True)

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            str: Extracted text from the PDF
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ''

    @staticmethod
    def summarize_text(text, max_length=1000):
        """
        Summarize the extracted text using OpenAI's API.
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum length of summary
        
        Returns:
            str: Summarized text
        """
        try:
            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
                {"role": "user", "content": f"Provide a concise summary of the following text, focusing on key points: {text[:4000]}"}
            ],
            max_tokens=max_length)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Unable to generate summary."

    @staticmethod
    def generate_study_questions(summary):
        """
        Generate study questions based on the summary using OpenAI's API.
        
        Args:
            summary (str): Text summary to generate questions from
        
        Returns:
            list: List of generated study questions
        """
        try:
            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert educator creating study questions."},
                {"role": "user", "content": f"Generate 5-7 diverse, thought-provoking study questions based on this summary: {summary}"}
            ])
            questions = response.choices[0].message.content.strip().split('\n')
            return [q.strip() for q in questions if q.strip()]
        except Exception as e:
            print(f"Error generating study questions: {e}")
            return ["Unable to generate study questions."]

    @staticmethod
    def format_questions_as_markdown(questions):
        """
        Convert study questions to markdown format.
        
        Args:
            questions (list): List of study questions
        
        Returns:
            str: Markdown-formatted questions
        """
        markdown_questions = "\n".join([f"- {q}" for q in questions])
        return markdown.markdown(markdown_questions)

    @staticmethod
    def save_summary(pdf_path, summary):
        """
        Save PDF summary to a JSON file.
        
        Args:
            pdf_path (str): Path to the original PDF
            summary (str): Generated summary
        
        Returns:
            str: Path to the saved summary file
        """
        filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path = os.path.join(SUMMARIES_DIR, filename)
        
        summary_data = {
            "pdf_path": pdf_path,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        return summary_path

    @staticmethod
    def save_questions(pdf_path, questions):
        """
        Save study questions to a JSON file.
        
        Args:
            pdf_path (str): Path to the original PDF
            questions (list): Generated study questions
        
        Returns:
            str: Path to the saved questions file
        """
        filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        questions_path = os.path.join(QUESTIONS_DIR, filename)
        
        questions_data = {
            "pdf_path": pdf_path,
            "questions": questions,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(questions_path, 'w') as f:
            json.dump(questions_data, f, indent=4)
        
        return questions_path

    @staticmethod
    def ask_ai_about_pdf(pdf_path, question):
        """
        Ask an AI question about a specific PDF.
        
        Args:
            pdf_path (str): Path to the PDF
            question (str): User's specific question about the PDF
        
        Returns:
            str: AI's response to the question
        """
        text = PDFProcessor.extract_text_from_pdf(pdf_path)
        
        try:
            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert PDF assistant. Answer questions based on the PDF content precisely."},
                {"role": "user", "content": f"PDF Content: {text[:4000]}\n\nQuestion: {question}"}
            ])
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error answering PDF question: {e}")
            return "I'm sorry, I couldn't process the question."
