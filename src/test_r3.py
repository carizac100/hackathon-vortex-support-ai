from security import detect_pii, pii_mask, detect_phishing, sentiment_score
from preprocessing import clean_text


tests = [
    {
        "title": "Ticket normal",
        "text": "Hola, necesito ayuda con el módulo de facturación, está dando error."
    },
    {
        "title": "Ticket con PII",
        "text": "Mi correo es juan.perez@example.com y mi número es 3174568899."
    },
    {
        "title": "Ticket phishing fuerte",
        "text": "Su cuenta ha sido suspendida por actividad inusual. Verifique su identidad urgentemente."
    },
    {
        "title": "Ticket con URL sospechosa",
        "text": "Revise este enlace http://xyzsecurity.ru para validar su información."
    },
    {
        "title": "Ticket con sentimiento negativo",
        "text": "Estoy muy inconforme, este servicio es terrible y nunca funciona."
    },
    {
        "title": "Ticket mezclado (PII + phishing + emocional)",
        "text": "Mi cuenta fue hackeada, este correo sospechoso llegó: tinyurl.com/asd. "
                "Además, esto es frustrante y estoy cansado. Teléfono: 3001234567."
    }
]

for test in tests:
    print("\n" + "="*60)
    print(f"TEST: {test['title']}")
    print("="*60)

    t = test["text"]

    print("\nTexto original:")
    print(t)

    print("\nPII detectado:")
    print(detect_pii(t))

    print("\nPII enmascarado:")
    print(pii_mask(t))

    print("\nPhishing detection:")
    print(detect_phishing(t))

    print("\nSentimiento:")
    print(sentiment_score(t))

    print("\nTexto limpio:")
    print(clean_text(t))
