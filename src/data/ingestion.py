# ============================================================
# AeroGuard — Data Ingestion
#
# Kaam: NGAFID dataset ko load karna
#
# Do cheezein load hongi:
#
#   1. all_flights/flight_header.csv
#      → poora dataset metadata (28,935 flights)
#      → EDA, transformation aur labeling ke liye
#
#   2. all_flights/one_parq
#      → actual sensor time-series (100M+ rows, 4.3GB)
#      → Dask lazy load — RAM safe
#
# Dask kyun?
#   Sensor dataset ~4.3GB hai.
#   Pandas se load karo → poori file RAM mein → crash.
#   Dask lazy evaluation karta hai:
#     - sirf metadata read hota hai load pe
#     - data tab aata hai jab compute() call ho
#
# Usage:
#   from src.data.ingestion import load_data
#   data = load_data()
#   data["header_full"]  → pd.DataFrame (28,935 flights)
#   data["sensor_data"]  → dd.DataFrame (lazy, 100M+ rows)
#   data["config"]       → dict
# ============================================================

import os
import pandas as pd
import dask.dataframe as dd
import yaml

from src.logger import logger
from src.exception import DataIngestionException


# ============================================================
# CONFIG LOADER
# ============================================================

def load_config() -> dict:
    """
    config.yaml load karta hai.

    Returns:
        dict: saari config settings

    Raises:
        DataIngestionException: agar config file nahi mili
    """
    try:
        config_path = "configs/config.yaml"

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"config.yaml nahi mili: {config_path}"
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.debug("config.yaml successfully load hua")

        return config

    except Exception as e:
        raise DataIngestionException(
            e, context="Loading config.yaml"
        )


# ============================================================
# FLIGHT HEADER LOADER
# ============================================================

def load_flight_header_full(header_path: str) -> pd.DataFrame:
    """
    flight_header.csv load karta hai — poora dataset.

    Yeh file har flight ka metadata contain karti hai:
      - before_after : maintenance se pehle ya baad
      - date_diff    : maintenance se kitne din pehle/baad
      - flight_length: flight duration (seconds)
      - label        : maintenance issue type (36 types)
      - hierarchy    : fault system category
      - number_flights_before: sequence position

    Master Index = primary key — sensor data se join hoga.

    Args:
        header_path: flight_header.csv ka path

    Returns:
        pd.DataFrame: 28,935 flights ka metadata

    Raises:
        DataIngestionException: agar file nahi mili
    """
    try:
        logger.info(
            f"Full flight header load ho raha hai: {header_path}"
        )

        if not os.path.exists(header_path):
            raise FileNotFoundError(
                f"Full flight header nahi mila: {header_path}\n"
                f"Dataset download karo: "
                f"https://doi.org/10.5281/zenodo.6624956"
            )

        # Master Index = flight ka unique ID
        # Sensor data ke saath join key hai
        df = pd.read_csv(header_path, index_col="Master Index")

        logger.info(
            f"Full flight header load hua — Shape: {df.shape}"
        )
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Total flights: {len(df):,}")

        # before_after distribution
        if "before_after" in df.columns:
            ba_dist = df['before_after'].value_counts()
            for val, cnt in ba_dist.items():
                logger.info(
                    f"  {val}: {cnt:,} flights "
                    f"({cnt/len(df)*100:.1f}%)"
                )

        return df

    except Exception as e:
        raise DataIngestionException(
            e, context="Loading full flight_header.csv"
        )


# ============================================================
# SENSOR DATA LOADER (DASK)
# ============================================================

def load_flight_sensor_data(parquet_path: str) -> dd.DataFrame:
    """
    Flight sensor data Dask mein lazy load karta hai.

    4.3GB dataset — 100M+ rows — 23 sensors + timestep + cluster.
    Pandas mein load karna = RAM overflow = crash.
    Dask lazy loading = sirf schema read hota hai.

    23 Sensors:
      Electrical : volt1, volt2, amp1, amp2
      Fuel       : FQtyL, FQtyR, E1 FFlow
      Engine     : E1 OilT, E1 OilP, E1 RPM
      Cylinders  : E1 CHT1-4, E1 EGT1-4
      Flight     : OAT, IAS, VSpd, NormAc, AltMSL

    Extra columns:
      timestep : row order within flight (0 = first second)
      cluster  : maintenance label (will be dropped later)

    Args:
        parquet_path: one_parq folder ka path

    Returns:
        dd.DataFrame: lazy loaded Dask DataFrame

    Raises:
        DataIngestionException: agar folder nahi mila
    """
    try:
        logger.info(
            f"Flight sensor data lazy load ho raha hai: {parquet_path}"
        )

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Sensor data folder nahi mila: {parquet_path}\n"
                f"Dataset download karo: "
                f"https://doi.org/10.5281/zenodo.6624956"
            )

        # Lazy load — abhi RAM mein nahi aayega
        # Sirf schema aur metadata read hoga
        df = dd.read_parquet(parquet_path)

        logger.info("Sensor data lazy load hua")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Total partitions: {df.npartitions}")

        return df

    except Exception as e:
        raise DataIngestionException(
            e, context="Loading flight sensor parquet data"
        )


# ============================================================
# MAIN DATA INGESTION PIPELINE
# ============================================================

def load_data() -> dict:
    """
    Main function — poora dataset load karta hai.

    Pipeline:
      config load → header load → sensor lazy load

    Returns:
        dict:
          "header_full"  → pd.DataFrame (28,935 flights)
          "sensor_data"  → dd.DataFrame (lazy, 100M+ rows)
          "config"       → dict (loaded config)

    Raises:
        DataIngestionException: koi bhi step fail ho toh
    """
    try:
        logger.info("=" * 55)
        logger.info("AEROGUARD DATA INGESTION SHURU")
        logger.info("=" * 55)

        # Step 1 — config load
        config = load_config()

        # Step 2 — full flight header
        header_full = load_flight_header_full(
            config["data"]["flight_header_full"]
        )

        # Step 3 — sensor data (lazy)
        sensor_data = load_flight_sensor_data(
            config["data"]["raw_flight_data"]
        )

        logger.info("=" * 55)
        logger.info("DATA INGESTION COMPLETE ✓")
        logger.info("=" * 55)

        return {
            "header_full" : header_full,
            "sensor_data" : sensor_data,
            "config"      : config,
        }

    except Exception as e:
        raise DataIngestionException(
            e, context="Main data loading pipeline"
        )