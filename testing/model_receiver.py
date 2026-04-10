"""
model_receiver.py — Receives pt + zip files from slave /modelfile call.
Stores files in ./files/ folder.

Run: python model_receiver.py
Port: 8000  (matches master_url in slave/config.json)
"""

import os
from flask import Flask, request, jsonify

app = Flask(__name__)

FILES_DIR = os.path.join(os.path.dirname(__file__), 'files')
os.makedirs(FILES_DIR, exist_ok=True)


@app.route('/api/auth/signin', methods=['POST'])
def signin():
    return jsonify({"userdata": {"user_id": "test_user"}}), 200


@app.route('/api/auth/slave', methods=['POST'])
def slave_auth():
    return jsonify({"token": "test_token"}), 200


@app.route('/modelfile', methods=['POST'])
def receive_model():
    material_code = request.form.get('material_code', '')
    package_type  = request.form.get('package_type', '')
    item_type     = request.form.get('type', '')

    print(f"[receiver] material_code={material_code} type={item_type}")

    saved = []

    if 'pt_file' in request.files:
        pt = request.files['pt_file']
        pt_path = os.path.join(FILES_DIR, pt.filename)
        pt.save(pt_path)
        print(f"[receiver] saved pt:  {pt_path}")
        saved.append(pt.filename)

    if 'zip_file' in request.files:
        zf = request.files['zip_file']
        zip_path = os.path.join(FILES_DIR, zf.filename)
        zf.save(zip_path)
        print(f"[receiver] saved zip: {zip_path}")
        saved.append(zf.filename)

    if not saved:
        return jsonify({'message': 'no files received'}), 400

    return jsonify({'message': 'files received', 'saved': saved}), 200


if __name__ == '__main__':
    print(f"[receiver] storing files in: {FILES_DIR}")
    app.run(host='0.0.0.0', port=8000, debug=False)
