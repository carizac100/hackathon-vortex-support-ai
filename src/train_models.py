"""
Script para entrenar los modelos de ML usando los datos sintéticos.

Ejecutar con: python3 src/train_models.py
"""

import sys
from pathlib import Path

# Agregar src al path para imports
sys.path.insert(0, str(Path(__file__).parent))

from models import create_model_trainer


def main():
    """Función principal para entrenar todos los modelos."""
    print("=" * 60)
    print("🚀 ENTRENAMIENTO DE MODELOS - VORTEX SUPPORT AI")
    print("=" * 60)
    print()
    
    # Crear entrenador
    trainer = create_model_trainer()
    
    # Entrenar todos los modelos
    try:
        metrics = trainer.train_all()
        
        print()
        print("=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print()
        print("Los modelos han sido guardados en la carpeta 'models/'")
        print("Ahora puedes ejecutar el dashboard con:")
        print("  python3 -m streamlit run src/dashboard_app.py")
        print()
        
    except FileNotFoundError as e:
        print()
        print("❌ ERROR: No se encontró el archivo de datos.")
        print(f"   {e}")
        print()
        print("👉 Primero genera los datos con:")
        print("   python3 src/generate_dummy_data.py")
        print()
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ ERROR durante el entrenamiento: {e}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
