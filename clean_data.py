# clean_data.py
import os
import re

data_folder = "data"

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_folder, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Supprimer les lignes vides multiples
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Supprimer les espaces en début/fin de chaque ligne
        lines = [line.strip() for line in content.split('\n')]
        
        # Supprimer les lignes vides ou trop courtes (menus, boutons...)
        lines = [line for line in lines if len(line) > 30]
        
        # Rejoindre
        content = '\n\n'.join(lines)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ {filename} nettoyé")

print("\n🎉 Tous les fichiers sont nettoyés !")