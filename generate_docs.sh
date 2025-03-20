#!/bin/bash

echo "Génération de la documentation avec Pydoc..."
mkdir -p docs
for script in scripts/*.py; do
    script_name=$(basename "$script" .py)
    pydoc -w "$script"
    mv "$script_name".html docs/
    echo "Documentation générée pour $script_name"
done

echo "Documentation complète disponible dans le dossier 'docs'."

