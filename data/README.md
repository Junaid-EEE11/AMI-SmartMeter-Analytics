# Low Carbon London Smart Meter Dataset Guide

## 1. Official Dataset Details

- **Official Name**: SmartMeter Energy Consumption Data in London Households (Low Carbon London Project)
- **Producing Organization**: UK Power Networks (UKPN), led as part of the Low Carbon London Low Carbon Networks Fund (LCNF) trial.
- **Official Portals**:
  - London Datastore: [https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
  - UK Data Service / UKPN Open Data Portal
- **License**: UK Open Government Licence v3.0 (OGL) / Crown Copyright and UK Power Networks.
- **Volume & Characteristics**:
  - Sample Size: 5,567 London households
  - Time Coverage: November 2011 through February 2014 (~167 million half-hourly meter readings)
  - Raw Storage: ~10 GB uncompressed CSV files (~1.2 GB compressed / Parquet)
  - Experimental Tariff Groups:
    - ~1,100 households subjected to the Dynamic Time-of-Use (`dToU`) tariff trial in 2013.
    - ~4,400 households under the Flat / Standard (`Std`) tariff serving as the comparison population.
    - Acorn demographic classifications (Affluent, Comfortable, Adversity).

---

## 2. Directory Structure

Place raw data files in `data/raw/`. The pipeline processes raw records into columnar Parquet files in `data/interim/` and `data/processed/`.

```
data/
├── raw/                      # Unmodified CSV files from UKPN / London Datastore
│   ├── CC_LCL-FullData.csv   # Or halfhourly_dataset/block_*.csv
│   └── informations_households.csv
├── interim/                  # Cleaned, standardized, schema-mapped Parquet chunks
│   └── lcl_cleaned.parquet
├── processed/                # Chronologically aggregated cohort and feeder series
│   ├── cohort_dtou_halfhourly.parquet
│   ├── cohort_std_halfhourly.parquet
│   └── cohort_total_halfhourly.parquet
└── README.md                 # This documentation file
```

---

## 3. Schema Mapping & Canonical Internal Names

Different releases of the Low Carbon London dataset (e.g., London Datastore block CSVs vs single consolidated files) format header names slightly differently. The project utilizes a configuration-driven schema mapping layer (`configs/main.yaml`) to map raw columns to canonical internal names:

| Canonical Internal Name | Description | Example Raw Column Names |
| :--- | :--- | :--- |
| `household_id` | Unique anonymized household identifier | `LCLid`, `MAC000002`, `ClientID` |
| `timestamp` | Start or end timestamp of half-hour reading | `DateTime`, `reading_datetime`, `tstamp` |
| `energy_kwh` | Half-hourly electricity consumption in kWh | `KWH/hh (per half hour)`, `energy(kWh/hh)`, `kwh` |
| `tariff_group` | Tariff cohort identifier | `stdorToU`, `Acorn_Group`, `tariff_group` (`Std` or `ToU`/`dToU`) |
| `tariff_level` | (Optional) Dynamic tariff price level signal | `tariff_level`, `price_signal` |
| `acorn_group` | (Optional) Demographic category | `Acorn`, `Acorn_grouped` |

---

## 4. Downloading and Reproducing

1. Download the raw CSV files from the official London Datastore link:
   ```bash
   # Example: Download to data/raw/
   # https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households
   ```
2. Place the downloaded CSV file(s) or folder inside `data/raw/`.
3. Verify the path and column names in `configs/main.yaml` under the `schema_mapping` block.
4. Execute the automated data preparation script:
   ```bash
   python scripts/prepare_data.py --config configs/main.yaml
   ```

---

## 5. Development with Synthetic Test Fixture

If real Low Carbon London data files are not present in `data/raw/`, `scripts/prepare_data.py` will automatically generate a deterministic synthetic AMI test fixture with labeled distribution shifts (`data/processed/synthetic_ami.parquet`). 

> **Important Scientific Rule**: Any experiment run on the synthetic test fixture is strictly for pipeline verification, unit testing, and smoke testing. Results are stamped `SYNTHETIC — FOR PIPELINE VALIDATION ONLY` and are never presented as empirical findings from the Low Carbon London dataset.
