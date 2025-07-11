from transformers import pipeline
import argparse
import json
from typing import List, Dict

def load_model(model_path: str) -> pipeline:
    """
    Загружает обученную модель и токенизатор
    """
    return pipeline(
        "ner",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="simple",
        device=0  # device=-1 для CPU, device=0 для GPU
    )

def predict_entities(text: str, nlp: pipeline) -> List[Dict]:
    """
    Предсказывает сущности в тексте
    """
    results = nlp(text)
    return [
        {
            "text": entity["word"],
            "label": entity["entity_group"],
            "score": round(entity["score"], 4),
            "start": entity["start"],
            "end": entity["end"]
        }
        for entity in results
    ]

def format_output(entities: List[Dict], output_format: str) -> str:
    """
    Форматирует вывод в указанный формат
    """
    if output_format == "json":
        return json.dumps(entities, ensure_ascii=False, indent=2)
    else:
        return "\n".join(
            f"{entity['text']} ({entity['label']}) [conf: {entity['score']:.2f}]"
            for entity in entities
        )

def main():
    parser = argparse.ArgumentParser(description="Предсказание сущностей с помощью NER-модели")
    parser.add_argument("--text", type=str, help="Текст для анализа")
    parser.add_argument("--file", type=str, help="Файл с текстом (построчно)")
    parser.add_argument("--model", type=str, default="./ner_model_simple", 
                       help="Путь к обученной модели (по умолчанию: ./ner_model_simple)")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                       help="Формат вывода (text/json)")
    
    args = parser.parse_args()

    try:
        nlp = load_model(args.model)
        print(f"Модель загружена из {args.model}", file=sys.stderr)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}", file=sys.stderr)
        return

    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("Укажите --text или --file с входными данными", file=sys.stderr)
        return

    for text in texts:
        if not text:
            continue
            
        print(f"\nАнализ текста: {text[:50]}...", file=sys.stderr)
        entities = predict_entities(text, nlp)
        print(format_output(entities, args.format))

if __name__ == "__main__":
    import sys
    main()