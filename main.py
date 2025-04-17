import sys
import os
import sqlite3
import threading
import queue
import time
import logging
from datetime import datetime
import json
import re

import numpy as np
import sounddevice as sd
import speech_recognition as sr
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QIcon, QColor, QPalette, QFont, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplitter, QComboBox, QAction, QMenu, QToolBar, QStatusBar, QFileDialog, QProgressBar, QLabel, QTextEdit, QListWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QWidget, QTabWidget, QPushButton
from pdf_handler import PDFProcessor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filename='note_taker.log',
    filemode='a'
)

# Constants
DB_PATH = 'notes.db'
AUDIO_PATH = 'temp_audio.wav'
CONFIG_PATH = 'config.json'

# Configuration management
def load_config():
    default_config = {
        "theme": "light",
        "font_size": 14,
        "microphone_index": None,
        "categories": ["Personal", "Work", "Ideas", "To-Do"],
        "language": "en-US"
    }
    
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return {**default_config, **json.load(f)}
        except Exception as e:
            logging.error(f"Error loading config: {e}")
    
    return default_config

def save_config(config):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        logging.error(f"Error saving config: {e}")
        return False

# Database setup
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Notes table
    c.execute('''CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        category TEXT,
        tags TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )''')
    
    # Categories table
    c.execute('''CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )''')
    
    conn.commit()
    
    # Add default categories if needed
    config = load_config()
    for category in config["categories"]:
        try:
            c.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (category,))
        except:
            pass
    
    conn.commit()
    conn.close()

# Text processing utilities
def extract_title_from_content(content, max_length=50):
    """Extract a title from the content automatically"""
    if not content:
        return "Untitled Note"
    
    # Try to find the first sentence or phrase
    lines = content.split('\n')
    first_line = lines[0] if lines else content
    
    # Remove special characters and extra spaces
    first_line = re.sub(r'[^\w\s]', '', first_line).strip()
    
    # Truncate if too long
    if len(first_line) > max_length:
        first_line = first_line[:max_length] + "..."
    
    return first_line or "Untitled Note"

# Note-taking logic
class NoteTaker(QtCore.QObject):
    note_ready = QtCore.pyqtSignal(str)
    stream_update = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    status_update = QtCore.pyqtSignal(str)
    microphones_detected = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.is_listening = False
        self.current_transcript = ''
        self.recognizer = sr.Recognizer()
        self.config = load_config()
        self.microphone_index = self.config.get("microphone_index")
        self.language = self.config.get("language", "en-US")
        
        # Initialize with better defaults for speech recognition
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 300  # Increased sensitivity
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        self.recognizer.pause_threshold = 0.8  # Shorter pause threshold
        
        # Scan for available microphones on initialization
        self.scan_microphones()
    
    def scan_microphones(self):
        """Scan for available microphones and emit list to UI"""
        try:
            microphones = []
            logging.info("Available microphones:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                logging.info(f"Microphone {index}: {name}")
                microphones.append((index, name))
            
            self.microphones_detected.emit(microphones)
        except Exception as e:
            logging.error(f"Error listing microphones: {e}")
            self.error.emit(f"Microphone Error: {e}")
    
    def set_microphone_index(self, index):
        """Set the microphone index to use"""
        self.microphone_index = index
        self.config["microphone_index"] = index
        save_config(self.config)
        logging.info(f"Microphone index set to {index}")
    
    def set_language(self, language):
        """Set the language for speech recognition"""
        self.language = language
        self.config["language"] = language
        save_config(self.config)
        logging.info(f"Recognition language set to {language}")

    def start_listening(self):
        """Start the voice recognition process"""
        self.current_transcript = ''
        self.is_listening = True
        self.status_update.emit("Listening...")
        threading.Thread(target=self._record_and_transcribe, daemon=True).start()

    def stop_listening(self):
        """Stop the voice recognition process"""
        self.is_listening = False
        self.status_update.emit("Stopped listening")

    def _record_and_transcribe(self):
        try:
            # Try to use the configured microphone
            try:
                if self.microphone_index is not None:
                    mic = sr.Microphone(device_index=int(self.microphone_index))
                    logging.info(f"Using microphone with index {self.microphone_index}")
                else:
                    mic = sr.Microphone()
                    logging.info("Using default microphone")
            except Exception as e:
                logging.warning(f"Failed to use configured microphone: {e}")
                # Fallback to default
                mic = sr.Microphone()
                logging.info("Falling back to default microphone")
            
            # Adjust for ambient noise
            with mic as source:
                self.status_update.emit("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                logging.info("Adjusted for ambient noise with enhanced settings")
                self.status_update.emit("Listening...")
            
            def callback(recognizer, audio):
                try:
                    logging.debug("Audio chunk received, attempting transcription")
                    self.status_update.emit("Processing speech...")
                    
                    # Try with enhanced settings
                    text = recognizer.recognize_google(
                        audio, 
                        language=self.language, 
                        show_all=False
                    )
                    
                    if text:
                        logging.info(f"Transcribed text: {text}")
                        # Add a space only if there's existing text
                        separator = ' ' if self.current_transcript else ''
                        self.current_transcript += f'{separator}{text}'
                        self.stream_update.emit(self.current_transcript)
                        self.status_update.emit("Listening...")
                        
                except sr.UnknownValueError:
                    logging.warning("Could not understand audio - trying again with different settings")
                    try:
                        # Try again with different settings
                        text = recognizer.recognize_google(
                            audio, 
                            language=self.language, 
                            show_all=True
                        )
                        if text and 'alternative' in text:
                            best_guess = text['alternative'][0]['transcript']
                            logging.info(f"Second attempt transcribed text: {best_guess}")
                            # Add a space only if there's existing text
                            separator = ' ' if self.current_transcript else ''
                            self.current_transcript += f'{separator}{best_guess}'
                            self.stream_update.emit(self.current_transcript)
                            self.status_update.emit("Listening...")
                    except:
                        self.status_update.emit("Didn't catch that...")
                
                except sr.RequestError as e:
                    logging.error(f'Speech recognition request failed: {e}')
                    self.error.emit(f'Network error: {e}')
                    self.status_update.emit("Network error")
            
            # Use a separate recognizer for the background listener
            listener = sr.Recognizer()
            listener.energy_threshold = self.recognizer.energy_threshold
            listener.dynamic_energy_threshold = self.recognizer.dynamic_energy_threshold
            listener.dynamic_energy_adjustment_damping = self.recognizer.dynamic_energy_adjustment_damping
            listener.dynamic_energy_ratio = self.recognizer.dynamic_energy_ratio
            listener.pause_threshold = self.recognizer.pause_threshold
            
            stop_listening = listener.listen_in_background(mic, callback)
            
            # Keep the thread alive while listening
            while self.is_listening:
                time.sleep(0.1)
            
            # Stop the background listener when needed
            stop_listening(wait_for_stop=False)
        
        except Exception as e:
            logging.critical(f"Critical error in recording: {e}")
            self.error.emit(f"Recording failed: {e}")
            self.status_update.emit("Recording failed")

# Main Window
class NoteTakerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Note Taker Pro")
        self.setGeometry(200, 100, 1000, 700)
        
        # Load configuration
        self.config = load_config()
        
        # State variables
        self.current_note_id = None
        self.current_category = None
        self.is_modified = False
        self.search_term = ""
        
        # Initialize the note taker
        self.taker = NoteTaker()
        self.taker.note_ready.connect(self.append_transcribed_note)
        self.taker.stream_update.connect(self.update_live_transcript)
        self.taker.error.connect(self.show_error)
        self.taker.status_update.connect(self.update_status_bar)
        self.taker.microphones_detected.connect(self.populate_microphone_menu)
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_statusbar()
        
        # Apply theme (after UI is set up)
        self.apply_theme(self.config.get("theme", "light"))
        
        # Load initial notes
        self.load_notes()
        
        # Auto-save timer
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self.auto_save)
        self.auto_save_timer.start(60000)  # Auto-save every minute

    def setup_pdf_tab(self):
        pdf_tab = QWidget()
        pdf_layout = QVBoxLayout()

        # PDF File Selection
        pdf_file_layout = QHBoxLayout()
        self.pdf_file_path = QLineEdit()
        self.pdf_file_path.setPlaceholderText('Select PDF file')
        pdf_select_btn = QPushButton('Select PDF')
        pdf_select_btn.clicked.connect(self.select_pdf_file)
        pdf_file_layout.addWidget(self.pdf_file_path)
        pdf_file_layout.addWidget(pdf_select_btn)
        pdf_layout.addLayout(pdf_file_layout)

        # Process PDF Button
        process_pdf_btn = QPushButton('Process PDF')
        process_pdf_btn.clicked.connect(self.process_pdf)
        pdf_layout.addWidget(process_pdf_btn)

        # Progress Bar
        self.pdf_progress = QProgressBar()
        pdf_layout.addWidget(self.pdf_progress)

        # Summary Display
        self.pdf_summary_display = QTextEdit()
        self.pdf_summary_display.setReadOnly(True)
        pdf_layout.addWidget(QLabel('Summary:'))
        pdf_layout.addWidget(self.pdf_summary_display)

        # Study Questions Display
        self.study_questions_display = QListWidget()
        pdf_layout.addWidget(QLabel('Study Questions:'))
        pdf_layout.addWidget(self.study_questions_display)

        pdf_tab.setLayout(pdf_layout)
        return pdf_tab

    def select_pdf_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select PDF File', '', 'PDF Files (*.pdf)')
        if file_path:
            self.pdf_file_path.setText(file_path)

    def process_pdf(self):
        pdf_path = self.pdf_file_path.text()
        if not pdf_path:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Please select a PDF file first.')
            return

        # Reset UI
        self.pdf_summary_display.clear()
        self.study_questions_display.clear()
        self.pdf_progress.setValue(0)

        try:
            # Extract Text
            self.pdf_progress.setValue(25)
            extracted_text = PDFProcessor.extract_text_from_pdf(pdf_path)
            if not extracted_text:
                raise ValueError('No text could be extracted from the PDF.')

            # Generate Summary
            self.pdf_progress.setValue(50)
            summary = PDFProcessor.summarize_text(extracted_text)
            self.pdf_summary_display.setText(summary)

            # Generate Study Questions
            self.pdf_progress.setValue(75)
            questions = PDFProcessor.generate_study_questions(summary)
            self.study_questions_display.addItems(questions)

            self.pdf_progress.setValue(100)
            QtWidgets.QMessageBox.information(self, 'Success', 'PDF processed successfully!')

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'An error occurred: {str(e)}')
            self.pdf_progress.setValue(0)

    def setup_notes_tab(self):
        # Create a placeholder notes tab widget
        notes_tab = QWidget()
        notes_layout = QVBoxLayout()
        notes_label = QLabel('Notes Tab Placeholder')
        notes_layout.addWidget(notes_label)
        notes_tab.setLayout(notes_layout)
        return notes_tab

    def setup_audio_tab(self):
        # Create a placeholder audio tab widget
        audio_tab = QWidget()
        audio_layout = QVBoxLayout()
        audio_label = QLabel('Audio Tab Placeholder')
        audio_layout.addWidget(audio_label)
        audio_tab.setLayout(audio_layout)
        return audio_tab

    def setup_summary_tab(self):
        # Create a placeholder summary tab widget
        summary_tab = QWidget()
        summary_layout = QVBoxLayout()
        summary_label = QLabel('Summary Tab Placeholder')
        summary_layout.addWidget(summary_label)
        summary_tab.setLayout(summary_layout)
        return summary_tab

    def setup_ui(self):
        # Central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        # Main layout with splitter
        main_layout = QtWidgets.QVBoxLayout(central)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Add tabs
        self.tab_widget.addTab(self.setup_notes_tab(), 'Notes')
        self.tab_widget.addTab(self.setup_audio_tab(), 'Audio')
        self.tab_widget.addTab(self.setup_summary_tab(), 'Summary')
        self.tab_widget.addTab(self.setup_pdf_tab(), 'PDF Insights')
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Sidebar for notes list and categories
        sidebar_widget = QtWidgets.QWidget()
        sidebar = QtWidgets.QVBoxLayout(sidebar_widget)
        
        # Category filter
        category_layout = QtWidgets.QHBoxLayout()
        category_layout.addWidget(QtWidgets.QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.setStyleSheet("font-size: 14px; padding: 4px;")
        self.category_combo.currentTextChanged.connect(self.filter_by_category)
        category_layout.addWidget(self.category_combo)
        sidebar.addLayout(category_layout)
        
        # Search box
        search_layout = QtWidgets.QHBoxLayout()
        search_layout.addWidget(QtWidgets.QLabel("Search:"))
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search notes...")
        self.search_edit.setStyleSheet("font-size: 14px; padding: 6px;")
        self.search_edit.textChanged.connect(self.search_notes)
        search_layout.addWidget(self.search_edit)
        sidebar.addLayout(search_layout)
        
        # Notes list
        sidebar.addWidget(QtWidgets.QLabel("<b>Notes</b>"))
        self.notes_list = QtWidgets.QListWidget()
        self.notes_list.setStyleSheet("font-size: 16px; padding: 8px;")
        self.notes_list.itemClicked.connect(self.load_selected_note)
        sidebar.addWidget(self.notes_list)
        
        # New note button
        self.new_note_btn = QtWidgets.QPushButton("+ New Note")
        self.new_note_btn.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px;")
        self.new_note_btn.clicked.connect(self.new_note)
        sidebar.addWidget(self.new_note_btn)
        
        # Add sidebar to splitter
        splitter.addWidget(sidebar_widget)
        
        # Main note editing area
        edit_widget = QtWidgets.QWidget()
        edit_layout = QtWidgets.QVBoxLayout(edit_widget)
        
        # Title and category for current note
        title_layout = QtWidgets.QHBoxLayout()
        
        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.setPlaceholderText("Note Title")
        self.title_edit.setStyleSheet("font-size: 20px; font-weight: bold; padding: 8px;")
        self.title_edit.textChanged.connect(self.mark_as_modified)
        title_layout.addWidget(self.title_edit, 3)
        
        self.note_category_combo = QComboBox()
        self.note_category_combo.setStyleSheet("font-size: 14px; padding: 4px;")
        self.note_category_combo.currentTextChanged.connect(self.mark_as_modified)
        title_layout.addWidget(QtWidgets.QLabel("Category:"))
        title_layout.addWidget(self.note_category_combo, 1)
        
        edit_layout.addLayout(title_layout)
        
        # Live transcript area
        self.live_transcript = QtWidgets.QTextEdit()
        self.live_transcript.setPlaceholderText("Live Transcript...")
        self.live_transcript.setStyleSheet("font-size: 16px; padding: 8px;")
        self.live_transcript.setReadOnly(True)
        edit_layout.addWidget(self.live_transcript, 2)
        
        # Content edit area
        self.content_edit = QtWidgets.QTextEdit()
        self.content_edit.setPlaceholderText("Your note will appear here...")
        self.content_edit.setStyleSheet("font-size: 16px; padding: 8px;")
        self.content_edit.textChanged.connect(self.mark_as_modified)
        edit_layout.addWidget(self.content_edit, 3)
        
        # Tags field
        tags_layout = QtWidgets.QHBoxLayout()
        tags_layout.addWidget(QtWidgets.QLabel("Tags:"))
        self.tags_edit = QtWidgets.QLineEdit()
        self.tags_edit.setPlaceholderText("Enter tags separated by commas")
        self.tags_edit.setStyleSheet("font-size: 14px; padding: 4px;")
        self.tags_edit.textChanged.connect(self.mark_as_modified)
        tags_layout.addWidget(self.tags_edit)
        edit_layout.addLayout(tags_layout)
        
        # Controls
        controls = QtWidgets.QHBoxLayout()
        
        self.start_btn = QtWidgets.QPushButton("Start Voice Note")
        self.start_btn.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self.start_voice_note)
        controls.addWidget(self.start_btn)
        
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px;")
        self.stop_btn.clicked.connect(self.stop_voice_note)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.stop_btn)
        
        self.save_btn = QtWidgets.QPushButton("Save Note")
        self.save_btn.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px;")
        self.save_btn.clicked.connect(self.save_note)
        controls.addWidget(self.save_btn)
        
        self.delete_btn = QtWidgets.QPushButton("Delete Note")
        self.delete_btn.setStyleSheet("font-size: 14px; padding: 8px;")
        self.delete_btn.clicked.connect(self.delete_note)
        controls.addWidget(self.delete_btn)
        
        edit_layout.addLayout(controls)
        
        # Add edit area to splitter
        splitter.addWidget(edit_widget)
        
        # Set default splitter sizes (30% sidebar, 70% edit area)
        splitter.setSizes([300, 700])
        
        # Add the splitter to the main layout
        main_layout.addWidget(splitter)

    def setup_menu(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        new_action = QAction('&New Note', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_note)
        file_menu.addAction(new_action)
        
        save_action = QAction('&Save Note', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_note)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('&Export Notes', self)
        export_action.triggered.connect(self.export_notes)
        file_menu.addAction(export_action)
        
        import_action = QAction('&Import Notes', self)
        import_action.triggered.connect(self.import_notes)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('&Edit')
        
        undo_action = QAction('&Undo', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.content_edit.undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction('&Redo', self)
        redo_action.setShortcut('Ctrl+Y')
        redo_action.triggered.connect(self.content_edit.redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        cut_action = QAction('&Cut', self)
        cut_action.setShortcut('Ctrl+X')
        cut_action.triggered.connect(self.content_edit.cut)
        edit_menu.addAction(cut_action)
        
        copy_action = QAction('&Copy', self)
        copy_action.setShortcut('Ctrl+C')
        copy_action.triggered.connect(self.content_edit.copy)
        edit_menu.addAction(copy_action)
        
        paste_action = QAction('&Paste', self)
        paste_action.setShortcut('Ctrl+V')
        paste_action.triggered.connect(self.content_edit.paste)
        edit_menu.addAction(paste_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('&Tools')
        
        # Microphone submenu
        mic_menu = QMenu('&Microphone', self)
        self.mic_menu = mic_menu  # Store as instance variable to update later
        tools_menu.addMenu(mic_menu)
        
        refresh_mic_action = QAction('Refresh Microphone List', self)
        refresh_mic_action.triggered.connect(self.taker.scan_microphones)
        tools_menu.addAction(refresh_mic_action)
        
        tools_menu.addSeparator()
        
        # Language submenu
        language_menu = QMenu('Recognition &Language', self)
        
        languages = [
            ("English (US)", "en-US"),
            ("English (UK)", "en-GB"),
            ("Spanish", "es-ES"),
            ("French", "fr-FR"),
            ("German", "de-DE"),
            ("Chinese", "zh-CN"),
            ("Japanese", "ja-JP"),
            ("Russian", "ru-RU"),
            ("Portuguese", "pt-BR"),
            ("Italian", "it-IT")
        ]
        
        for name, code in languages:
            lang_action = QAction(name, self)
            lang_action.setCheckable(True)
            if code == self.taker.language:
                lang_action.setChecked(True)
            lang_action.triggered.connect(lambda _, c=code: self.taker.set_language(c))
            language_menu.addAction(lang_action)
        
        tools_menu.addMenu(language_menu)
        
        # Settings menu
        settings_menu = menubar.addMenu('&Settings')
        
        # Theme submenu
        theme_menu = QMenu('&Theme', self)
        
        light_theme_action = QAction('Light', self)
        light_theme_action.setCheckable(True)
        if self.config.get("theme") == "light":
            light_theme_action.setChecked(True)
        light_theme_action.triggered.connect(lambda: self.set_theme("light"))
        theme_menu.addAction(light_theme_action)
        
        dark_theme_action = QAction('Dark', self)
        dark_theme_action.setCheckable(True)
        if self.config.get("theme") == "dark":
            dark_theme_action.setChecked(True)
        dark_theme_action.triggered.connect(lambda: self.set_theme("dark"))
        theme_menu.addAction(dark_theme_action)
        
        settings_menu.addMenu(theme_menu)
        
        # Font size submenu
        font_menu = QMenu('&Font Size', self)
        
        for size in [12, 14, 16, 18, 20]:
            font_action = QAction(f'{size}pt', self)
            font_action.setCheckable(True)
            if self.config.get("font_size") == size:
                font_action.setChecked(True)
            font_action.triggered.connect(lambda _, s=size: self.set_font_size(s))
            font_menu.addAction(font_action)
        
        settings_menu.addMenu(font_menu)
        
        settings_menu.addSeparator()
        
        manage_categories_action = QAction('Manage &Categories', self)
        manage_categories_action.triggered.connect(self.manage_categories)
        settings_menu.addAction(manage_categories_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        keyboard_shortcuts_action = QAction('&Keyboard Shortcuts', self)
        keyboard_shortcuts_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction(keyboard_shortcuts_action)

    def setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Add actions to toolbar
        new_action = QAction('New', self)
        new_action.triggered.connect(self.new_note)
        toolbar.addAction(new_action)
        
        save_action = QAction('Save', self)
        save_action.triggered.connect(self.save_note)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        start_action = QAction('Record', self)
        start_action.triggered.connect(self.start_voice_note)
        toolbar.addAction(start_action)
        
        stop_action = QAction('Stop', self)
        stop_action.triggered.connect(self.stop_voice_note)
        toolbar.addAction(stop_action)
        
        toolbar.addSeparator()
        
        export_action = QAction('Export', self)
        export_action.triggered.connect(self.export_notes)
        toolbar.addAction(export_action)

    def setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def update_status_bar(self, message):
        self.statusbar.showMessage(message)

    def populate_microphone_menu(self, microphones):
        """Populate the microphone menu with detected devices"""
        self.mic_menu.clear()
        
        for index, name in microphones:
            mic_action = QAction(f"{name}", self)
            mic_action.setCheckable(True)
            if self.taker.microphone_index == index:
                mic_action.setChecked(True)
            mic_action.triggered.connect(lambda _, idx=index: self.taker.set_microphone_index(idx))
            self.mic_menu.addAction(mic_action)

    def apply_theme(self, theme):
        """Apply light or dark theme to the application"""
        if theme == "dark":
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #2d2d2d; color: #e0e0e0; }
                QLineEdit, QTextEdit, QComboBox, QListWidget { 
                    background-color: #3d3d3d; 
                    color: #e0e0e0; 
                    border: 1px solid #5d5d5d; 
                    border-radius: 4px; 
                }
                QPushButton { 
                    background-color: #4d4d4d; 
                    color: #e0e0e0; 
                    border: 1px solid #5d5d5d;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton:hover { background-color: #5d5d5d; }
                QPushButton:pressed { background-color: #3d3d3d; }
                QMenuBar, QMenu { background-color: #2d2d2d; color: #e0e0e0; }
                QMenuBar::item:selected, QMenu::item:selected { background-color: #4d4d4d; }
            """)
        else:  # Light theme
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #f6f6f6; color: #333; }
                QLineEdit, QTextEdit, QComboBox, QListWidget { 
                    background-color: white; 
                    color: #333; 
                    border: 1px solid #ddd; 
                    border-radius: 4px; 
                }
                QPushButton { 
                    background-color: #f0f0f0; 
                    color: #333; 
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton:hover { background-color: #e5e5e5; }
                QPushButton:pressed { background-color: #d0d0d0; }
            """)
        
        # Override specific button colors
        self.start_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.stop_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.save_btn.setStyleSheet("background-color: #FFA500; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")

    def set_theme(self, theme):
        """Set the application theme"""
        self.config["theme"] = theme
        save_config(self.config)
        self.apply_theme(theme)

    def set_font_size(self, size):
        """Set the application font size"""
        self.config["font_size"] = size
        save_config(self.config)
        
        font = QFont()
        font.setPointSize(size)
        QApplication.setFont(font)
        
        # Update specific components that need font adjustment
        self.content_edit.setStyleSheet(f"font-size: {size}px; padding: 8px;")
        self.live_transcript.setStyleSheet(f"font-size: {size}px; padding: 8px;")
        self.notes_list.setStyleSheet(f"font-size: {size}px; padding: 8px;")

    def load_categories(self):
        """Load categories from database to combo boxes"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM categories ORDER BY name")
        categories = [row[0] for row in c.fetchall()]
        conn.close()


# Clear and populate the category comboboxes
        self.category_combo.clear()
        self.note_category_combo.clear()
        
        # Add "All Categories" to filter combo only
        self.category_combo.addItem("All Categories")
        
        # Add categories to both combo boxes
        for category in categories:
            self.category_combo.addItem(category)
            self.note_category_combo.addItem(category)

    def load_notes(self, filter_category=None, search_term=None):
        """Load notes from database with optional filtering"""
        self.notes_list.clear()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        query = "SELECT id, title, category, created_at FROM notes"
        params = []
        
        # Build the WHERE clause based on filters
        where_clauses = []
        
        if filter_category and filter_category != "All Categories":
            where_clauses.append("category = ?")
            params.append(filter_category)
        
        if search_term:
            where_clauses.append("(title LIKE ? OR content LIKE ?)")
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        query += " ORDER BY created_at DESC"
        
        c.execute(query, params)
        notes = c.fetchall()
        conn.close()
        
        for note_id, title, category, created_at in notes:
            # Create list item with title and optional category
            item_text = title
            if category:
                item_text += f" [{category}]"
            
            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, note_id)
            
            # Format the date for tooltip
            try:
                created_datetime = datetime.fromisoformat(created_at)
                formatted_date = created_datetime.strftime("%B %d, %Y at %I:%M %p")
                item.setToolTip(f"Created: {formatted_date}")
            except:
                pass
            
            self.notes_list.addItem(item)
        
        # Update the status bar
        self.update_status_bar(f"Loaded {len(notes)} notes")

    def filter_by_category(self, category):
        """Filter notes by selected category"""
        self.current_category = category if category != "All Categories" else None
        self.load_notes(filter_category=self.current_category, search_term=self.search_term)

    def search_notes(self, term):
        """Search notes by title and content"""
        self.search_term = term.strip()
        self.load_notes(filter_category=self.current_category, search_term=self.search_term)

    def load_selected_note(self, item):
        """Load the selected note from the list"""
        # Check for unsaved changes
        if self.is_modified:
            reply = QtWidgets.QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "You have unsaved changes. Save before loading another note?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self.save_note()
            elif reply == QtWidgets.QMessageBox.Cancel:
                return
        
        note_id = item.data(QtCore.Qt.UserRole)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT title, content, category, tags FROM notes WHERE id=?", (note_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            self.current_note_id = note_id
            title, content, category, tags = row
            
            self.title_edit.setText(title)
            self.content_edit.setText(content)
            
            # Set the category
            index = self.note_category_combo.findText(category) if category else 0
            if index >= 0:
                self.note_category_combo.setCurrentIndex(index)
            
            # Set tags
            self.tags_edit.setText(tags or "")
            
            # Reset modification flag
            self.is_modified = False
            self.update_status_bar(f"Loaded note: {title}")

    def new_note(self):
        """Create a new note"""
        # Check for unsaved changes
        if self.is_modified:
            reply = QtWidgets.QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "You have unsaved changes. Save before creating a new note?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self.save_note()
            elif reply == QtWidgets.QMessageBox.Cancel:
                return
        
        self.current_note_id = None
        self.title_edit.clear()
        self.content_edit.clear()
        self.tags_edit.clear()
        self.notes_list.clearSelection()
        self.is_modified = False
        self.update_status_bar("New note created")

    def save_note(self):
        """Save the current note to the database"""
        title = self.title_edit.text().strip()
        content = self.content_edit.toPlainText().strip()
        category = self.note_category_combo.currentText()
        tags = self.tags_edit.text().strip()
        
        # If title is empty but there's content, generate a title
        if not title and content:
            title = extract_title_from_content(content)
            self.title_edit.setText(title)
        
        if not content:
            QtWidgets.QMessageBox.warning(self, "Missing Content", "Please enter some content for your note.")
            return
        
        now = datetime.now().isoformat()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        try:
            if self.current_note_id:
                # Update existing note
                c.execute(
                    "UPDATE notes SET title=?, content=?, category=?, tags=?, updated_at=? WHERE id=?", 
                    (title, content, category, tags, now, self.current_note_id)
                )
            else:
                # Insert new note
                c.execute(
                    "INSERT INTO notes (title, content, category, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)", 
                    (title, content, category, tags, now, now)
                )
                self.current_note_id = c.lastrowid
            
            conn.commit()
            self.is_modified = False
            self.update_status_bar(f"Note '{title}' saved successfully")
            
            # Refresh the notes list but maintain the current filters
            self.load_notes(filter_category=self.current_category, search_term=self.search_term)
            
        except Exception as e:
            logging.error(f"Error saving note: {e}")
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save note: {e}")
        
        finally:
            conn.close()

    def auto_save(self):
        """Auto-save the current note if modified"""
        if self.is_modified and self.content_edit.toPlainText().strip():
            try:
                self.save_note()
                self.update_status_bar("Note auto-saved")
            except Exception as e:
                logging.error(f"Auto-save error: {e}")

    def delete_note(self):
        """Delete the current note"""
        if not self.current_note_id:
            QtWidgets.QMessageBox.warning(self, "No Note Selected", "Please select a note to delete.")
            return
        
        confirm = QtWidgets.QMessageBox.question(
            self, 
            "Confirm Delete", 
            "Are you sure you want to delete this note? This action cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        try:
            c.execute("DELETE FROM notes WHERE id=?", (self.current_note_id,))
            conn.commit()
            
            self.current_note_id = None
            self.title_edit.clear()
            self.content_edit.clear()
            self.tags_edit.clear()
            self.is_modified = False
            
            self.update_status_bar("Note deleted")
            
            # Refresh the notes list
            self.load_notes(filter_category=self.current_category, search_term=self.search_term)
            
        except Exception as e:
            logging.error(f"Error deleting note: {e}")
            QtWidgets.QMessageBox.critical(self, "Delete Error", f"Failed to delete note: {e}")
        
        finally:
            conn.close()

    def start_voice_note(self):
        """Start recording and transcribing voice"""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.live_transcript.clear()
        self.taker.start_listening()

    def stop_voice_note(self):
        """Stop recording and add the transcript to the note"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.taker.stop_listening()
        
        # Get the transcribed text
        live_text = self.live_transcript.toPlainText().strip()
        
        if live_text:
            # Add to content
            current_content = self.content_edit.toPlainText().strip()
            
            # Add a separator if there's existing content
            separator = "\n\n" if current_content else ""
            
            # Add the transcribed text with a timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.content_edit.setText(f"{current_content}{separator}[{timestamp}] {live_text}")
            
            # If there's no title, suggest one from the transcription
            if not self.title_edit.text().strip():
                suggested_title = extract_title_from_content(live_text)
                self.title_edit.setText(suggested_title)
            
            # Mark as modified to trigger a save
            self.mark_as_modified()
        
        self.update_status_bar("Voice recording stopped")

    def update_live_transcript(self, text):
        """Update the live transcript area with the current text"""
        self.live_transcript.setPlainText(text)

    def append_transcribed_note(self, text):
        """Append the transcribed note to the content"""
        if text:
            self.content_edit.append(f"\n{text}\n")
            self.mark_as_modified()

    def show_error(self, msg):
        """Display error messages"""
        QtWidgets.QMessageBox.critical(self, "Error", msg)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_status_bar("Error occurred")

    def mark_as_modified(self):
        """Mark the current note as modified"""
        self.is_modified = True

    def export_notes(self):
        """Export notes to a file"""
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 
            "Export Notes", 
            "", 
            "JSON Files (*.json);;Text Files (*.txt)", 
            options=options
        )
        
        if not file_path:
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT id, title, content, category, tags, created_at, updated_at FROM notes")
            notes = [
                {
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "category": row[3],
                    "tags": row[4],
                    "created_at": row[5],
                    "updated_at": row[6]
                }
                for row in c.fetchall()
            ]
            conn.close()
            
            # Export as JSON
            if file_path.endswith(".json"):
                with open(file_path, 'w') as f:
                    json.dump({"notes": notes}, f, indent=2)
            
            # Export as text
            elif file_path.endswith(".txt"):
                with open(file_path, 'w') as f:
                    for note in notes:
                        f.write(f"Title: {note['title']}\n")
                        if note['category']:
                            f.write(f"Category: {note['category']}\n")
                        if note['tags']:
                            f.write(f"Tags: {note['tags']}\n")
                        f.write(f"Created: {note['created_at']}\n")
                        f.write(f"Updated: {note['updated_at']}\n")
                        f.write(f"\n{note['content']}\n")
                        f.write("\n" + "-" * 50 + "\n\n")
            
            QtWidgets.QMessageBox.information(self, "Export Successful", f"Notes exported to {file_path}")
            
        except Exception as e:
            logging.error(f"Export error: {e}")
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export notes: {e}")

    def import_notes(self):
        """Import notes from a file"""
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "Import Notes", 
            "", 
            "JSON Files (*.json)", 
            options=options
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "notes" not in data:
                raise ValueError("Invalid file format")
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            imported_count = 0
            
            for note in data["notes"]:
                # Required fields
                title = note.get("title", "Imported Note")
                content = note.get("content", "")
                
                # Optional fields with defaults
                category = note.get("category", "")
                tags = note.get("tags", "")
                created_at = note.get("created_at", datetime.now().isoformat())
                updated_at = note.get("updated_at", created_at)
                
                # Insert the note
                c.execute(
                    "INSERT INTO notes (title, content, category, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (title, content, category, tags, created_at, updated_at)
                )
                
                imported_count += 1
            
            conn.commit()
            conn.close()
            
            self.load_notes()
            QtWidgets.QMessageBox.information(
                self, 
                "Import Successful", 
                f"Successfully imported {imported_count} notes"
            )
            
        except Exception as e:
            logging.error(f"Import error: {e}")
            QtWidgets.QMessageBox.critical(self, "Import Error", f"Failed to import notes: {e}")

    def manage_categories(self):
        """Open dialog to manage categories"""
        # Load current categories
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name FROM categories ORDER BY name")
        categories = c.fetchall()
        conn.close()
        
        # Create dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Manage Categories")
        dialog.setMinimumWidth(400)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Category list
        layout.addWidget(QtWidgets.QLabel("<b>Categories</b>"))
        category_list = QtWidgets.QListWidget()
        
        for cat_id, cat_name in categories:
            item = QtWidgets.QListWidgetItem(cat_name)
            item.setData(QtCore.Qt.UserRole, cat_id)
            category_list.addItem(item)
        
        layout.addWidget(category_list)
        
        # Add category
        add_layout = QtWidgets.QHBoxLayout()
        add_layout.addWidget(QtWidgets.QLabel("New Category:"))
        new_category_edit = QtWidgets.QLineEdit()
        add_layout.addWidget(new_category_edit)
        add_btn = QtWidgets.QPushButton("Add")
        add_layout.addWidget(add_btn)
        layout.addLayout(add_layout)
        
        # Delete category
        delete_btn = QtWidgets.QPushButton("Delete Selected")
        layout.addWidget(delete_btn)
        
        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        layout.addWidget(close_btn)
        
        # Connect signals
        def add_category():
            name = new_category_edit.text().strip()
            if not name:
                return
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            try:
                c.execute("INSERT INTO categories (name) VALUES (?)", (name,))
                conn.commit()
                
                # Add to list
                item = QtWidgets.QListWidgetItem(name)
                item.setData(QtCore.Qt.UserRole, c.lastrowid)
                category_list.addItem(item)
                
                # Clear input
                new_category_edit.clear()
                
                # Refresh UI categories
                self.load_categories()
                
            except sqlite3.IntegrityError:
                QtWidgets.QMessageBox.warning(dialog, "Duplicate", "This category already exists.")
            
            finally:
                conn.close()
        
        def delete_category():
            selected = category_list.currentItem()
            if not selected:
                return
            
            cat_id = selected.data(QtCore.Qt.UserRole)
            cat_name = selected.text()
            
            confirm = QtWidgets.QMessageBox.question(
                dialog, 
                "Confirm Delete", 
                f"Are you sure you want to delete the category '{cat_name}'? Notes in this category will be set to uncategorized.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            
            if confirm != QtWidgets.QMessageBox.Yes:
                return
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            try:
                # Update notes with this category
                c.execute("UPDATE notes SET category = NULL WHERE category = ?", (cat_name,))
                
                # Delete the category
                c.execute("DELETE FROM categories WHERE id = ?", (cat_id,))
                
                conn.commit()
                
                # Remove from list
                category_list.takeItem(category_list.row(selected))
                
                # Refresh UI categories
                self.load_categories()
                
            except Exception as e:
                logging.error(f"Error deleting category: {e}")
                QtWidgets.QMessageBox.critical(dialog, "Delete Error", f"Failed to delete category: {e}")
            
            finally:
                conn.close()
        
        # Connect signals
        add_btn.clicked.connect(add_category)
        new_category_edit.returnPressed.connect(add_category)
        delete_btn.clicked.connect(delete_category)
        close_btn.clicked.connect(dialog.accept)
        
        # Show dialog
        dialog.exec_()

    def show_about(self):
        """Show about dialog"""
        QtWidgets.QMessageBox.about(
            self, 
            "About Voice Note Taker Pro",
            "<h2>Voice Note Taker Pro</h2>"
            "<p>Version 2.0</p>"
            "<p>A powerful application for taking and organizing voice notes.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Voice-to-text transcription</li>"
            "<li>Note organization with categories and tags</li>"
            "<li>Search and filter capabilities</li>"
            "<li>Dark and light themes</li>"
            "<li>Import/Export functionality</li>"
            "</ul>"
        )

    def show_shortcuts(self):
        """Show keyboard shortcuts dialog"""
        QtWidgets.QMessageBox.information(
            self, 
            "Keyboard Shortcuts",
            "<h3>Keyboard Shortcuts</h3>"
            "<table>"
            "<tr><td><b>Ctrl+N</b></td><td>New Note</td></tr>"
            "<tr><td><b>Ctrl+S</b></td><td>Save Note</td></tr>"
            "<tr><td><b>Ctrl+Q</b></td><td>Exit Application</td></tr>"
            "<tr><td><b>Ctrl+Z</b></td><td>Undo</td></tr>"
            "<tr><td><b>Ctrl+Y</b></td><td>Redo</td></tr>"
            "<tr><td><b>Ctrl+X</b></td><td>Cut</td></tr>"
            "<tr><td><b>Ctrl+C</b></td><td>Copy</td></tr>"
            "<tr><td><b>Ctrl+V</b></td><td>Paste</td></tr>"
            "</table>"
        )

    def closeEvent(self, event):
        """Handle application close event"""
        if self.is_modified:
            reply = QtWidgets.QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "You have unsaved changes. Save before exiting?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self.save_note()
            elif reply == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
        
        event.accept()

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Create application
    app = QtWidgets.QApplication(sys.argv)
    
    # Apply font from config
    config = load_config()
    font_size = config.get("font_size", 14)
    font = QFont()
    font.setPointSize(font_size)
    app.setFont(font)
    
    # Create and show window
    window = NoteTakerApp()
    window.show()
    
    # Run application
    sys.exit(app.exec_())