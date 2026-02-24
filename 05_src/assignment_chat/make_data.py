import csv
import random
from pathlib import Path


TOPICS = [
    ("Weather & Geography", [
        "geocoding converts city names into latitude and longitude",
        "weather forecasts include daily high and low temperatures",
        "precipitation probability shows the chance of rain",
        "wind speed is measured at ten meters above ground level",
        "Open-Meteo provides free weather data without an API key",
        "time zones affect how forecast dates are interpreted",
        "humidity influences how hot or cold it feels",
        "satellite data improves weather prediction accuracy",
        "climate zones affect long-term temperature patterns",
        "barometric pressure changes indicate storms",
    ]),
]


def main(n_docs: int = 200, seed: int = 7):
    rng = random.Random(seed)

    out_path = Path("05_src/assignment_chat/test.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, n_docs + 1):
        topic, sentences = rng.choice(TOPICS)
        title = f"{topic}: Note {i}"
        # Build a short paragraph
        k = rng.randint(2, 4)
        text = " ".join(rng.sample(sentences, k))
        rows.append((f"doc{i:04d}", title, text))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "text"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()