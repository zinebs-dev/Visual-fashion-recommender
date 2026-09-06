import os
import numpy as np
import cv2
import pickle
import random
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from model import get_features
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs('template', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)


print("Chargement du modèle ResNet50...")
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
print(" Modèle chargé")


FEATURES_FILE = 'features.pkl'

if os.path.exists(FEATURES_FILE):
    print(f"Chargement de {FEATURES_FILE}...")
    with open(FEATURES_FILE, 'rb') as f:
        product_features, product_details = pickle.load(f)
    print(f" {len(product_details)} produits chargés")
else:
    print("features.pkl non trouvé - exécutez init_db.py")
    product_features = {}
    product_details = {}

@app.route('/')
def home():
    if not product_details:
        return """
        <html><body style="padding:50px;text-align:center">
            <h1 style="color:red">⚠️ Base de données vide</h1>
            <p>Exécutez d'abord: <code>python init_db.py</code></p>
            <p>Assurez-vous d'avoir des images dans static/jupes/, static/pantalon/, etc.</p>
        </body></html>
        """
    
    all_items = list(product_details.items())
    selected = random.sample(all_items, min(8, len(all_items)))
    
    products_for_template = []
    for key, details in selected:
        product = details.copy()
        product['image'] = key  # Format: "jupes/image1.jpg"
        products_for_template.append(product)
    
    return render_template('index.html', products=products_for_template)

@app.route('/upload', methods=['POST'])
def upload():
    """Recherche par image - RETOURNE JSON"""
    if not product_details:
        return jsonify({"error": "Base de données vide"})
    
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image"})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Fichier vide"})
    
  
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    query_features = get_features(filepath)
    if query_features is None:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": "Impossible de traiter l'image"})
    
    # Calculer similarités
    results = []
    for img_key, features in product_features.items():
        sim = cosine_similarity([query_features], [features])[0][0]
        product = product_details[img_key].copy()
        product['image'] = img_key
        product['similarity'] = f"{sim*100:.1f}%"
        results.append((sim, product))
    
   
    results.sort(key=lambda x: x[0], reverse=True)
    
    
    similar_products = [product for _, product in results[:4]]
    
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    return jsonify(similar_products)

# ==================== API POUR HTML ====================

@app.route('/api/products/random')
def api_random_products():
    if not product_details:
        return jsonify([])
    
    count = int(request.args.get('count', 8))
    all_keys = list(product_details.keys())
    
    if len(all_keys) <= count:
        random_keys = all_keys
    else:
        random_keys = random.sample(all_keys, count)
    
    results = []
    for key in random_keys:
        product = product_details[key].copy()
        product['image'] = key
        results.append(product)
    
    return jsonify(results)

@app.route('/api/products/category/<category>')
def api_category_products(category):
    results = []
    for key, details in product_details.items():
        if details['category'] == category:
            product = details.copy()
            product['image'] = key
            results.append(product)
    
    return jsonify(results)

@app.route('/api/products')
def api_all_products():
    results = []
    for key, details in product_details.items():
        product = details.copy()
        product['image'] = key
        results.append(product)
    
    return jsonify(results)

if __name__ == '__main__':
    print(f"\nServeur prêt: http://127.0.0.1:5000")
    print(f"Produits: {len(product_details)}")
    app.run(debug=True, port=5000)