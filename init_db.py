import os
import glob
import pickle
import numpy as np
from model import get_features
product_features = {}
product_details = {}

classes = ['jupes', 'pantalon', 'manteau', 'chaussure', 'chapeau']

for class_name in classes:
    folder = f'static/{class_name}'
    if not os.path.exists(folder):
        print(f" Création: {folder}")
        os.makedirs(folder)
        continue
    
    images = glob.glob(os.path.join(folder, '*.jpg')) + \
             glob.glob(os.path.join(folder, '*.jpeg')) + \
             glob.glob(os.path.join(folder, '*.png'))
    
    print(f"\n{class_name}: {len(images)} images")
    
    for i, img_path in enumerate(images[:5]):  
        try:
            print(f"  Traitement {i+1}: {os.path.basename(img_path)}")
            
            features = get_features(img_path)
            
            if features is not None:  
                img_name = os.path.basename(img_path)
                key = f"{class_name}/{img_name}"
                product_features[key] = features
                
                product_details[key] = {
                    'name': f"{class_name.capitalize()} {i+1}",
                    'price': f"{np.random.randint(20, 200)}€",
                    'category': class_name
                }
            else:
                print(f"  Échec extraction pour: {img_path}")
            
        except Exception as e:
            print(f"  Erreur: {e}")

with open('features.pkl', 'wb') as f:
    pickle.dump((product_features, product_details), f)

print(f"\n Base créée: {len(product_details)} produits")
print("Exécutez maintenant: python app.py")