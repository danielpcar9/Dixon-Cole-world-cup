import pandas as pd
from datetime import datetime
import json
import numpy as np

def load_and_filter_matches():
    df = pd.read_csv('data/raw/international_results.csv')
    
    # Convertir fecha
    df['date'] = pd.to_datetime(df['date'])
    
    # Filtrar desde 2010
    df = df[df['date'] >= '2010-01-01'].copy()
    
    # Limpiar NaN en scores
    df = df.dropna(subset=['home_score', 'away_score'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    
    print(f"Total partidos cargados: {len(df)}")
    print(f"Desde: {df['date'].min().date()} hasta {df['date'].max().date()}")
    print(f"Partidos eliminados por NaN: {len(df) - len(df)} wait, corrected above")
    
    # Convertir a formato del modelo
    matches = []
    for _, row in df.iterrows():
        matches.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "home_team": row['home_team'],
            "away_team": row['away_team'],
            "home_score": int(row['home_score']),
            "away_score": int(row['away_score']),
            "tournament": row.get('tournament', 'Friendly'),
            "neutral": bool(row.get('neutral', False))
        })
    
    # Guardar
    output_path = 'mundial_betting/sample_matches.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Guardados {len(matches)} partidos en {output_path}")
    return matches

if __name__ == "__main__":
    load_and_filter_matches()
