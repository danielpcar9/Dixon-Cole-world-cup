import pandas as pd
import json


def load_and_filter_matches():
    # 1. Cargar datos crudos
    df_raw = pd.read_csv("data/raw/international_results.csv")
    total_inicial = len(df_raw)

    # 2. Convertir fecha y filtrar desde 2010
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_filtrado = df_raw[df_raw["date"] >= "2010-01-01"].copy()

    # 3. Limpiar NaN en los marcadores sacando la diferencia real
    total_antes_nan = len(df_filtrado)
    df_clean = df_filtrado.dropna(subset=["home_score", "away_score"]).copy()
    eliminados_nan = total_antes_nan - len(df_clean)

    # 4. Asegurar tipos de datos corregidos
    df_clean["home_score"] = df_clean["home_score"].astype(int)
    df_clean["away_score"] = df_clean["away_score"].astype(int)
    df_clean["neutral"] = df_clean["neutral"].astype(bool)

    # Creamos la columna string de fecha para el JSON
    df_clean["date_str"] = df_clean["date"].dt.strftime("%Y-%m-%d")

    print(f"📊 Total partidos desde 2010: {len(df_clean)}")
    print(
        f"📅 Rango: {df_clean['date'].min().date()} hasta {df_clean['date'].max().date()}"
    )
    print(f"⚠️ Partidos eliminados por NaN en este rango: {eliminados_nan}")

    # 5. Mapeo ultra rápido a formato diccionario usando Pandas
    matches = [
        {
            "date": row["date_str"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": row["home_score"],
            "away_score": row["away_score"],
            "tournament": row.get("tournament", "Friendly"),
            "neutral": row["neutral"],
        }
        for row in df_clean.to_dict(orient="records")
    ]

    # 6. Guardar persistencia de los partidos muestra
    output_path = "mundial_betting/sample_matches.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    print(f"✅ Guardados {len(matches)} partidos en {output_path}")
    return matches


if __name__ == "__main__":
    load_and_filter_matches()
