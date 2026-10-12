from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


# Стоп-слова заданы вручную, чтобы предобработка не зависела от скачивания
# дополнительных корпусов NLTK.
STOP_WORDS = {
    "а", "без", "более", "бы", "был", "была", "были", "было", "быть",
    "в", "вам", "вас", "весь", "во", "вот", "все", "всего", "всех",
    "вы", "где", "да", "даже", "для", "до", "его", "ее", "если", "есть",
    "еще", "же", "за", "и", "из", "или", "им", "их", "к", "как", "ко",
    "когда", "кто", "ли", "либо", "мне", "может", "мы", "на", "над",
    "надо", "наш", "не", "него", "нее", "нет", "ни", "них", "но", "ну",
    "о", "об", "однако", "он", "она", "они", "от", "по", "под", "при",
    "про", "раз", "с", "со", "так", "такой", "там", "то", "того", "тоже",
    "только", "у", "уже", "что", "чтобы", "эта", "эти", "это", "я",
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "with", "this", "that", "it", "not"
}

WORD_RE = re.compile(r"[a-zа-яё]+", flags=re.IGNORECASE)
RUS_STEMMER = SnowballStemmer("russian")
ENG_STEMMER = SnowballStemmer("english")

POSITIVE_HINTS = {
    "хорош", "отлич", "прекрас", "класс", "удоб", "интерес", "качеств",
    "прият", "супер", "замеч", "понрав", "порад", "впечат", "рекоменд",
    "полез", "довол"
}
NEGATIVE_HINTS = {
    "плох", "ужас", "скуч", "неудоб", "разочар", "отврат", "медлен",
    "дорог", "бесполез", "гряз", "шум", "слом", "тормоз", "завис",
    "раздраж", "слаб"
}


def tokenize(text: str) -> list[str]:
    """Токенизация: выделяем только последовательности букв."""
    return WORD_RE.findall(str(text).lower())


def stem_token(token: str) -> str:
    """Стемминг русских и английских слов без внешних моделей."""
    if re.fullmatch(r"[а-яё]+", token):
        return RUS_STEMMER.stem(token)
    return ENG_STEMMER.stem(token)


def clean_text(text: str, use_stemming: bool = True) -> str:
    """
    Самостоятельная очистка текста:
    1) нижний регистр;
    2) удаление пунктуации и цифр через токенизацию;
    3) удаление стоп-слов;
    4) опциональный стемминг.
    """
    tokens = tokenize(text)
    tokens = [token for token in tokens if token not in STOP_WORDS]

    if use_stemming:
        tokens = [stem_token(token) for token in tokens]

    return " ".join(tokens)


def load_data(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [column for column in (text_column, label_column) if column not in df.columns]
    if missing:
        raise ValueError(
            f"В {path} нет колонок: {', '.join(missing)}. "
            f"Доступные колонки: {', '.join(df.columns)}"
        )

    result = df[[text_column, label_column]].dropna().copy()
    result[text_column] = result[text_column].astype(str)
    result[label_column] = result[label_column].astype(str)

    if result[label_column].nunique() < 2:
        raise ValueError("Для классификации нужно минимум два класса.")

    return result


def contains_any(text: str, hints: set[str]) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def explain_error(original: str, cleaned: str) -> str:
    """Формирует гипотезу о причине конкретной ошибки модели."""
    original_lower = original.lower()
    cleaned_tokens = cleaned.split()
    original_tokens = tokenize(original)

    if re.search(r"\b(не|нет|ни)\b", original_lower):
        return (
            "В тексте есть отрицание. Слово «не» удаляется как стоп-слово, "
            "поэтому смысл фразы может инвертироваться."
        )

    has_positive = contains_any(original, POSITIVE_HINTS)
    has_negative = contains_any(original, NEGATIVE_HINTS)
    if has_positive and has_negative:
        return (
            "В одном тексте смешаны положительные и отрицательные признаки. "
            "TF-IDF учитывает слова, но плохо понимает, какая часть фразы важнее."
        )

    if len(cleaned_tokens) <= 2:
        return (
            "После очистки осталось слишком мало слов. "
            "У модели недостаточно текстового сигнала для уверенной классификации."
        )

    if original_tokens and len(cleaned_tokens) / len(original_tokens) < 0.45:
        return (
            "Большая часть исходных токенов была удалена как стоп-слова. "
            "При агрессивной очистке могла потеряться часть контекста."
        )

    return (
        "Вероятная причина — ограниченность bag-of-words/TF-IDF: "
        "модель опирается на частоты слов и не понимает полный контекст, "
        "порядок фраз и скрытый смысл."
    )


def save_error_analysis(
    output_path: Path,
    x_test_original: pd.Series,
    x_test_clean: pd.Series,
    y_test: pd.Series,
    y_pred,
) -> pd.DataFrame:
    errors = pd.DataFrame(
        {
            "original_text": x_test_original.reset_index(drop=True),
            "cleaned_text": x_test_clean.reset_index(drop=True),
            "true_label": y_test.reset_index(drop=True),
            "predicted_label": pd.Series(y_pred),
        }
    )

    errors = errors[errors["true_label"] != errors["predicted_label"]].copy()
    errors["hypothesis"] = [
        explain_error(original, cleaned)
        for original, cleaned in zip(errors["original_text"], errors["cleaned_text"])
    ]

    # По условию нужно проанализировать 5 ошибок.
    top_errors = errors.head(5).copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_errors.to_csv(output_path, index=False)

    return top_errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML-P-6: ручная предобработка текста + TF-IDF + классификация"
    )
    parser.add_argument("--data", default="data.csv", help="Путь к CSV-файлу")
    parser.add_argument("--text-column", default="text", help="Колонка с текстом")
    parser.add_argument("--label-column", default="label", help="Колонка с классом")
    parser.add_argument("--no-stemming", action="store_true", help="Отключить стемминг")
    parser.add_argument("--min-df", type=int, default=2, help="Параметр min_df для TF-IDF")
    parser.add_argument("--max-df", type=float, default=0.90, help="Параметр max_df для TF-IDF")
    args = parser.parse_args()

    data_path = Path(args.data)
    df = load_data(data_path, args.text_column, args.label_column)

    use_stemming = not args.no_stemming
    df["clean_text"] = df[args.text_column].apply(
        lambda text: clean_text(text, use_stemming=use_stemming)
    )

    # Сначала делим данные, затем TF-IDF обучаем только на train,
    # чтобы информация из test не попадала в словарь/IDF.
    x_train_original, x_test_original, y_train, y_test = train_test_split(
        df[args.text_column],
        df[args.label_column],
        test_size=0.30,
        random_state=42,
        stratify=df[args.label_column],
    )

    x_train_clean = df.loc[x_train_original.index, "clean_text"]
    x_test_clean = df.loc[x_test_original.index, "clean_text"]

    vectorizer = TfidfVectorizer(
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    x_train_tfidf = vectorizer.fit_transform(x_train_clean)
    x_test_tfidf = vectorizer.transform(x_test_clean)

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(x_train_tfidf, y_train)
    y_pred = classifier.predict(x_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(y_test, y_pred, labels=classifier.classes_)

    print("=== ML-P-6: классификация текста ===")
    print(f"Всего объектов: {len(df)}")
    print(f"Train: {len(x_train_clean)} | Test: {len(x_test_clean)}")
    print(f"Стемминг: {'включен' if use_stemming else 'выключен'}")
    print(f"TF-IDF словарь: {len(vectorizer.vocabulary_)} признаков")
    print(f"min_df={args.min_df}, max_df={args.max_df}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(pd.DataFrame(matrix, index=classifier.classes_, columns=classifier.classes_))

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    metrics_text = (
        "ML-P-6 results\n"
        f"objects={len(df)}\n"
        f"train={len(x_train_clean)}\n"
        f"test={len(x_test_clean)}\n"
        f"stemming={use_stemming}\n"
        f"tfidf_features={len(vectorizer.vocabulary_)}\n"
        f"min_df={args.min_df}\n"
        f"max_df={args.max_df}\n"
        f"accuracy={accuracy:.4f}\n"
        f"macro_f1={f1_macro:.4f}\n\n"
        f"{report}\n"
    )
    (results_dir / "metrics.txt").write_text(metrics_text, encoding="utf-8")

    top_errors = save_error_analysis(
        results_dir / "error_analysis.csv",
        x_test_original,
        x_test_clean,
        y_test,
        y_pred,
    )

    print("\n=== Анализ ошибок ===")
    if top_errors.empty:
        print(
            "На test-выборке ошибок нет. Увеличьте test_size, используйте более сложные "
            "данные или проверьте другой random_state, чтобы выполнить анализ ошибок."
        )
    else:
        for number, row in enumerate(top_errors.itertuples(index=False), start=1):
            print(f"\nОшибка {number}")
            print(f"Текст: {row.original_text}")
            print(f"После очистки: {row.cleaned_text}")
            print(f"Истинный класс: {row.true_label}")
            print(f"Предсказание: {row.predicted_label}")
            print(f"Гипотеза: {row.hypothesis}")

    if len(top_errors) < 5:
        print(
            f"\nПредупреждение: найдено только {len(top_errors)} ошибок. "
            "Для отчета по условию требуется 5."
        )


if __name__ == "__main__":
    main()
