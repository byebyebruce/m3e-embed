
import numpy as np
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('m3e-base')


app = Flask(__name__)
models = {
    #'text-embedding-ada-002': EmbeddingModel()
}

@app.route('/v1/embeddings', methods=['POST'])
def embed_text():
    data = request.get_json()
    texts = data.get('input', [])
    #model_name = data.get('model', '')

    if not isinstance(texts, list):
        return jsonify({'error': 'inputs must be a list'}), 400

    #if model_name not in models:
    #    return jsonify({'error': 'model not supported'}), 400

    #model = models[model_name]
    result = {
        'data': [],
        'model': 'm3e-base',
        'object': 'list',
        'usage': {
            'prompt_tokens': 0,
            'total_tokens': 0
        }
    }

    embeddings = model.encode(texts)
    for i, embedding in enumerate(embeddings):
        normalized_embedding = embedding / np.linalg.norm(embedding)
        result['data'].append({
            'embedding': normalized_embedding.tolist(),  # Convert numpy array to list
            'index': i,
            'object': 'embedding'
        })

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

