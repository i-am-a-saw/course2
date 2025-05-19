from transformers import pipeline
from .use_model import scan

class SentimentAnalyzer:
    def __init__(self):
        try:
            # Используем модель, подходящую для русского языка
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="blanchefort/rubert-base-cased-sentiment",
                tokenizer="blanchefort/rubert-base-cased-sentiment",
                framework="pt"
            )
        except Exception as e:
            print(f"Ошибка загрузки модели анализа тональности: {str(e)}")
            self.analyzer = None

    def analyze_sentiment(self, text):
        if not text or not isinstance(text, str):
            return "Отрицательный"  # Значение по умолчанию для пустого текста
        if not self.analyzer:
            return "Отрицательный"  # Значение по умолчанию, если модель не загрузилась

        try:
            # Анализ тональности текста
            result = scan(text)
            #result = self.analyzer(text)[0]
            # Модель blanchefort/rubert возвращает метки: POSITIVE, NEGATIVE, NEUTRAL
            if result == "Рекомендую":
                return "Положительный"
            elif result["label"] == "Не рекомендую":
                return "Отрицательный"
            else:  # NEUTRAL
                return "Смешанные"
        except Exception as e:
            print(f"Ошибка анализа тональности: {str(e)}")
            return "Отрицательный"  # Значение по умолчанию при ошибке