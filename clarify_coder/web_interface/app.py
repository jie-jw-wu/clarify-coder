from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the parent directory to the path so we can import clarify_coder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clarify_coder import ClarifyCoder

app = Flask(__name__, static_folder='templates/template_1', template_folder='templates')

# Initialize ClarifyCoder
clarify_coder = ClarifyCoder()

@app.route('/')
def index():
    return render_template('template_1/index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        user_message = request.json.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get response from the LLM
        llm_response = clarify_coder.llm.get_response(user_message)
        
        return jsonify({
            'user_message': user_message,
            'llm_response': llm_response,
            'llm_name': clarify_coder.interface
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)