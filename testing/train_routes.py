"""
Train Routes  —  Train Plan Slave Communication
===============================================
Sends training commands from Master to a CountBot slave.

Endpoints:
    POST /trainplan/start  - Initiate a train session (with print config)
    POST /trainplan/train  - Begin active training for a specific line item
"""

from flask import Blueprint, request, jsonify, current_app
from app.utils.auth import token_required
from pymongo import MongoClient
from bson import ObjectId
import requests

bp = Blueprint('train_routes', __name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_bay(db, bay_id):
    """Fetch a bay document by ObjectId string."""
    return db['bays'].find_one({"_id": ObjectId(bay_id)})


def _slave_headers():
    return {
        'Content-Type': 'application/json',
        'Authorization': current_app.config['AUTHORIZATION_TOKEN']
    }


def _slave_url(ip, path):
    return f"http://{ip}:8501{path}"


# ─── Train Plan Endpoints ─────────────────────────────────────────────────────

@bp.route('/trainplan/start', methods=['POST'])
@token_required
def trainplan_start(current_user):
    """Initiate a train plan session — forward start config to the slave."""
    mclient = MongoClient(current_app.config['MONGO_URI'])
    db = mclient[current_user['databasename']]

    data = request.get_json() or {}

    picklist_id = data.get('picklist_id')
    lineitem_id = data.get('lineitem_id')
    bay_id      = data.get('bay_id')

    if not picklist_id:
        return jsonify({"message": "picklist_id is required"}), 400
    if not lineitem_id:
        return jsonify({"message": "lineitem_id is required"}), 400
    if not bay_id:
        return jsonify({"message": "bay_id is required"}), 400

    bay = _get_bay(db, bay_id)
    if not bay or 'ip_address' not in bay:
        return jsonify({"message": "Bay not found or missing IP address"}), 400
    
    picklist_details = db['picklists'].find_one({"_id": ObjectId(picklist_id)}).get('lineitems', [])
    lineitem_details = next(
        (li for li in picklist_details if str(li.get('_id')) == lineitem_id),
        None
    )

    training_config = db['item_global_settings'].find_one() or {}
    if '_id' in training_config:
        training_config['_id'] = str(training_config['_id'])
    if lineitem_details and '_id' in lineitem_details:
        lineitem_details = dict(lineitem_details)
        lineitem_details['_id'] = str(lineitem_details['_id'])

    payload = {
        'picklist_id': picklist_id,
        'lineitem_id': lineitem_id,
        'lineitem_details': lineitem_details,
        'training_config': training_config,
        'status': 'start'
    }

    try:
        response = requests.post(
            _slave_url(bay['ip_address'], '/train/start'),
            json=payload,
            headers=_slave_headers()
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return jsonify({"message": "Slave unreachable"}), 503

    return jsonify({"message": "start successfully"}), 200


@bp.route('/trainplan/stop', methods=['POST'])
@token_required
def trainplan_stop(current_user):
    """Send stop command to the slave."""
    mclient = MongoClient(current_app.config['MONGO_URI'])
    db = mclient[current_user['databasename']]

    data = request.get_json() or {}

    picklist_id = data.get('picklist_id')
    lineitem_id = data.get('lineitem_id')
    bay_id      = data.get('bay_id')

    if not picklist_id:
        return jsonify({"message": "picklist_id is required"}), 400
    if not lineitem_id:
        return jsonify({"message": "lineitem_id is required"}), 400
    if not bay_id:
        return jsonify({"message": "bay_id is required"}), 400

    bay = _get_bay(db, bay_id)
    if not bay or 'ip_address' not in bay:
        return jsonify({"message": "Bay not found or missing IP address"}), 400

    payload = {
        'picklist_id': picklist_id,
        'lineitem_id': lineitem_id,
        'bay_id':      bay_id,
        'status': 'stop'
    }

    try:
        response = requests.post(
            _slave_url(bay['ip_address'], '/train/stop'),
            json=payload,
            headers=_slave_headers()
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return jsonify({"message": "Slave unreachable"}), 503

    return jsonify({"message": "trainstart successfully"}), 200


@bp.route('/modelfile', methods=['POST'])
@token_required
def receive_model_file(current_user):
    """Called by slave after training — saves .pt file and links to category/item."""
    material_code = request.form.get('material_code', '').strip()
    package_type  = request.form.get('package_type', '').strip()
    type_val      = request.form.get('type', '').strip()
    model_s3_url  = request.form.get('model_s3_url', '').strip()
 
    if 'pt_file' not in request.files or not request.files['pt_file'].filename:
        return jsonify({'message': 'pt_file is required'}), 400
 
    file = request.files['pt_file']
    _, ext = os.path.splitext(file.filename)
    if ext.lower() != '.pt':
        return jsonify({'message': 'Only .pt files are accepted'}), 400
 
    if not material_code or not package_type or not type_val:
        return jsonify({'message': 'material_code, package_type, and type are required'}), 400