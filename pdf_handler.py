import os
import markdown
import PyPDF2
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', ''))
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API

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
