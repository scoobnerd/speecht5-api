from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import warnings
warnings.filterwarnings("ignore")
import torch
import soundfile as sf
import numpy as np
import io
import base64
import librosa
import uuid
import os

app = Flask(__name__)
CORS(app)

# Global variables for models
processor = None
model = None
vocoder = None
voice_embeddings = {}

def load_models():
    global processor, model, vocoder
    print("Loading SpeechT5 models...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    print("Models loaded successfully!")

def create_speaker_embedding(audio_data, sample_rate):
    """Create speaker embedding from audio data"""
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    
    # Create a simple speaker embedding (this is a simplified approach)
    # In production, you'd use a proper speaker encoder
    embedding = np.random.randn(512).astype(np.float32)  # Placeholder
    return torch.tensor(embedding).unsqueeze(0)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "SpeechT5 API is running"})

@app.route('/test_connection', methods=['POST'])
def test_connection():
    return jsonify({"success": True, "message": "Connection successful"})

@app.route('/create_clone', methods=['POST'])
def create_voice_clone():
    try:
        data = request.get_json()
        
        if not data or 'audioBase64' not in data:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Decode base64 audio
        audio_base64 = data['audioBase64']
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load audio
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Create speaker embedding
        speaker_embedding = create_speaker_embedding(audio_data, sample_rate)
        
        # Generate unique ID for this voice clone
        clone_id = str(uuid.uuid4())
        
        # Store the embedding
        voice_embeddings[clone_id] = speaker_embedding
        
        return jsonify({
            "id": clone_id,
            "name": data.get('name', 'Voice Clone'),
            "status": "ready",
            "description": data.get('description', '')
        })
        
    except Exception as e:
        print(f"Error creating voice clone: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data or 'voiceCloneId' not in data:
            return jsonify({"error": "Missing text or voiceCloneId"}), 400
        
        text = data['text']
        voice_clone_id = data['voiceCloneId']
        
        # Get speaker embedding
        if voice_clone_id not in voice_embeddings:
            return jsonify({"error": "Voice clone not found"}), 404
        
        speaker_embedding = voice_embeddings[voice_clone_id]
        
        # Process text
        inputs = processor(text=text, return_tensors="pt")
        
        # Generate speech
        with torch.no_grad():
            speech = model.generate_speech(
                inputs["input_ids"], 
                speaker_embedding, 
                vocoder=vocoder
            )
        
        # Convert to audio bytes
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, speech.numpy(), 16000, format='WAV')
        audio_buffer.seek(0)
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return jsonify({
            "audioBase64": audio_base64,
            "message": "Text converted to speech successfully"
        })
        
    except Exception as e:
        print(f"Error in text to speech: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_status', methods=['POST'])
def get_status():
    try:
        data = request.get_json()
        clone_id = data.get('cloneId')
        
        if clone_id in voice_embeddings:
            return jsonify({"status": "ready"})
        else:
            return jsonify({"status": "not_found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_clone', methods=['POST'])
def delete_clone():
    try:
        data = request.get_json()
        clone_id = data.get('cloneId')
        
        if clone_id in voice_embeddings:
            del voice_embeddings[clone_id]
            return jsonify({"message": "Voice clone deleted successfully"})
        else:
            return jsonify({"error": "Voice clone not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_models()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
