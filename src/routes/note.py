import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Blueprint, jsonify, request
from openai import OpenAI, OpenAIError
from src.models.note import Note, db

note_bp = Blueprint('note', __name__)
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / '.env')
TRANSLATE_PROMPT_PATH = ROOT_DIR / 'prompts' / 'translate_prompt.md'
OPENROUTER_MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free'

@note_bp.route('/notes', methods=['GET'])
def get_notes():
    """Get all notes, ordered by most recently updated"""
    notes = Note.query.order_by(Note.updated_at.desc()).all()
    return jsonify([note.to_dict() for note in notes])

@note_bp.route('/notes', methods=['POST'])
def create_note():
    """Create a new note"""
    try:
        data = request.json
        if not data or 'title' not in data or 'content' not in data:
            return jsonify({'error': 'Title and content are required'}), 400
        
        note = Note(title=data['title'], content=data['content'])
        db.session.add(note)
        db.session.commit()
        return jsonify(note.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@note_bp.route('/notes/<int:note_id>', methods=['GET'])
def get_note(note_id):
    """Get a specific note by ID"""
    note = Note.query.get_or_404(note_id)
    return jsonify(note.to_dict())

@note_bp.route('/notes/<int:note_id>', methods=['PUT'])
def update_note(note_id):
    """Update a specific note"""
    try:
        note = Note.query.get_or_404(note_id)
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        note.title = data.get('title', note.title)
        note.content = data.get('content', note.content)
        db.session.commit()
        return jsonify(note.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@note_bp.route('/notes/<int:note_id>', methods=['DELETE'])
def delete_note(note_id):
    """Delete a specific note"""
    try:
        note = Note.query.get_or_404(note_id)
        db.session.delete(note)
        db.session.commit()
        return '', 204
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@note_bp.route('/notes/search', methods=['GET'])
def search_notes():
    """Search notes by title or content"""
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    
    notes = Note.query.filter(
        (Note.title.contains(query)) | (Note.content.contains(query))
    ).order_by(Note.updated_at.desc()).all()
    
    return jsonify([note.to_dict() for note in notes])

@note_bp.route('/notes/translate', methods=['POST'])
def translate_note_content():
    data = request.get_json(silent=True) or {}
    content = data.get('content')
    target_language = data.get('target_language')

    if not isinstance(content, str) or not content.strip():
        return jsonify({'error': 'Note content is required for translation.'}), 400
    if not isinstance(target_language, str) or not target_language.strip():
        return jsonify({'error': 'Target language is required.'}), 400

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        return jsonify({'error': 'Translation is unavailable: OPENROUTER_API_KEY is not configured.'}), 503

    try:
        system_prompt = TRANSLATE_PROMPT_PATH.read_text(encoding='utf-8').format(
            target_language=target_language.strip()
        )
    except OSError:
        return jsonify({'error': 'Translation is unavailable: the translation prompt could not be loaded.'}), 500

    try:
        client = OpenAI(
            api_key=api_key,
            base_url='https://openrouter.ai/api/v1',
        )
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': content},
            ],
        )
    except OpenAIError as exc:
        return jsonify({'error': f'Translation request failed ({type(exc).__name__}).'}), 502

    choices = response.choices or []
    translation = choices[0].message.content if choices and choices[0].message else None
    if not translation:
        return jsonify({'error': 'Translation service returned an empty result.'}), 502
    return jsonify({'translation': translation})

