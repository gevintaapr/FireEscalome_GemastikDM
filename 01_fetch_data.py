"""
01_fetch_data.py
Ambil data titik panas (NASA FIRMS) dan data cuaca (Open-Meteo) untuk
wilayah prioritas karhutla Indonesia (Riau, Sumsel, Jambi, Kalbar, Kalteng).

SEBELUM JALANKAN:
1. Daftar FIRMS MAP_KEY gratis (instan) di:
   https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Ganti FIRMS_MAP_KEY di bawah dengan key kamu.
3. Open-Meteo tidak butuh API key sama sekali.

Jalankan: python 01_fetch_data.py
Output tersimpan di folder data_raw/
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# ============================================================
# KONFIGURASI — sesuaikan di sini
# ============================================================
FIRMS_MAP_KEY = "27a1beae6a640eaeedeeb64c429514a1"
FIRMS_SOURCE = "VIIRS_NOAA20_SP"  # menggunakan NOAA-20 SP (Standard Processing) untuk data historis

# Bounding box format: "west,south,east,north"
BBOX_SUMATRA = "95,-6,109,3"
BBOX_KALIMANTAN = "108,-4,118,4"

START_DATE = "2020-01-01"  # mundur hingga 2020 (6 tahun data) untuk menemukan pattern machine learning yang lebih baik
END_DATE = (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d")  # mundur 2 hari karena Open-Meteo Archive butuh waktu update data

OUTPUT_DIR = "data_raw"

FETCH_FIRMS = False  # Ubah ke True jika ingin mendownload ulang data FIRMS
FETCH_WEATHER = True # Ubah ke False jika tidak ingin mendownload data cuaca

# Titik representatif tiap provinsi rawan, untuk tarik cuaca
WEATHER_POINTS = {
    "Riau": (0.5, 101.4),
    "SumSel": (-3.0, 104.7),
    "Jambi": (-1.6, 103.6),
    "KalBar": (0.0, 109.3),
    "KalTeng": (-1.7, 113.3),
}


def fetch_firms_range(bbox: str, start_date: str, end_date: str, source: str = FIRMS_SOURCE) -> pd.DataFrame:
    """
    FIRMS area API dibatasi maksimal 5 hari per request -> looping per 5 hari.
    Dokumentasi: https://firms.modaps.eosdis.nasa.gov/api/
    """
    all_chunks = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{FIRMS_MAP_KEY}/{source}/{bbox}/5/{date_str}"
        )
        try:
            df_chunk = pd.read_csv(url)
            if len(df_chunk) > 0:
                all_chunks.append(df_chunk)
                print(f"  {date_str}: {len(df_chunk)} titik panas")
            else:
                print(f"  {date_str}: 0 titik panas")
        except Exception as e:
            print(f"  {date_str}: GAGAL ambil data -> {e}")

        current += timedelta(days=5)
        time.sleep(1)  # jaga-jaga rate limit

    if not all_chunks:
        return pd.DataFrame()
    return pd.concat(all_chunks, ignore_index=True)


def fetch_openmeteo_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Open-Meteo historical weather API - gratis, tanpa API key.
    Dokumentasi: https://open-meteo.com/en/docs/historical-weather-api
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "precipitation_sum",
            "windspeed_10m_max",
            "winddirection_10m_dominant",
            "relative_humidity_2m_max",
        ]),
        "timezone": "Asia/Jakarta",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    daily = response.json()["daily"]
    df = pd.DataFrame(daily)
    df["latitude"] = lat
    df["longitude"] = lon
    return df


def main():
    if FIRMS_MAP_KEY == "GANTI_DENGAN_MAP_KEY_KAMU":
        raise SystemExit(
            "Isi dulu FIRMS_MAP_KEY di bagian KONFIGURASI. "
            "Daftar gratis di https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if FETCH_FIRMS:
        print("=== Mengambil data FIRMS: Sumatra ===")
        df_sumatra = fetch_firms_range(BBOX_SUMATRA, START_DATE, END_DATE)
        df_sumatra.to_csv(f"{OUTPUT_DIR}/firms_sumatra.csv", index=False)
        print(f"Total titik panas Sumatra: {len(df_sumatra)}")

        print("\n=== Mengambil data FIRMS: Kalimantan ===")
        df_kalimantan = fetch_firms_range(BBOX_KALIMANTAN, START_DATE, END_DATE)
        df_kalimantan.to_csv(f"{OUTPUT_DIR}/firms_kalimantan.csv", index=False)
        print(f"Total titik panas Kalimantan: {len(df_kalimantan)}")

        df_hotspots = pd.concat([df_sumatra, df_kalimantan], ignore_index=True)
        df_hotspots.to_csv(f"{OUTPUT_DIR}/firms_all.csv", index=False)
        print(f"\nTotal gabungan: {len(df_hotspots)} titik panas -> data_raw/firms_all.csv")
    else:
        print("=== Melewati proses unduhan data FIRMS (karena FETCH_FIRMS = False) ===")

    if FETCH_WEATHER:
        print("\n=== Mengambil data cuaca (Open-Meteo) ===")
        weather_dfs = []
        for name, (lat, lon) in WEATHER_POINTS.items():
            print(f"  {name}...")
            df_w = fetch_openmeteo_weather(lat, lon, START_DATE, END_DATE)
            df_w["region"] = name
            weather_dfs.append(df_w)
            time.sleep(1)

        df_weather = pd.concat(weather_dfs, ignore_index=True)
        df_weather.to_csv(f"{OUTPUT_DIR}/weather_all.csv", index=False)
        print(f"Selesai -> data_raw/weather_all.csv")

    print("\n=== SELESAI ===")
    print("Cek dulu jumlah baris firms_all.csv sebelum lanjut ke tahap 2.")
    print("Kalau terlalu tipis (<500 baris), perlebar BBOX atau perpanjang START_DATE.")


if __name__ == "__main__":
    main()