# 🌾 Project: Crop Yield Prediction in Syria

### FTL Syria – AI for Climate: Machine Learning Challenge (Group 4)

---

## 📖 Project Overview

This project is part of the **Frontier Tech Leaders (FTL) Syria Programme – AI for Climate** Machine Learning Challenge. Our team, **Group 4**, was assigned **Problem 1: Crop Yield Prediction**.

The goal is to build machine learning models that can predict agricultural crop yield in Syria using historical climate data (from NASA POWER) and agricultural production data (from FAOSTAT). This analysis will help us understand which climate variables—temperature, rainfall, or solar radiation—most influence crop yields, supporting the development of **climate-resilient agriculture** strategies.

### 🎯 Research Question
> Can historical weather conditions help us predict crop yields, and which machine learning model produces more accurate predictions?

---

## 👥 Team Members (Group 4)

| Name | Email Address | Field |
| :--- | :--- | :--- |
| **Nimra Afzaal** | nimraafzaal1998@gmail.com | Electrical and Power Engineering |
| **Sular Albalkhi** | sularalbalkhi@gmail.com | Electronics and Communications Engineering |
| **Zaid Aldeen Mohammad Kiki** | kikizaid2@gmail.com | Electrical and Power Engineering |
| **Ahmad Deeb** | ahmaddeebdev@gmail.com | Electronics and Communications Engineering |

---

## 📚 Datasets Downloaded

We collected two primary datasets for this project:

### 1. FAOSTAT – Crop Production Data
**Source:** [FAOSTAT – Crops and Livestock Products (QCL)](https://www.fao.org/faostat/en/#data/QCL)

This dataset provides historical agricultural production statistics for Syria (Country Code: 760). It contains records from **1961 to 2025** for various crops, including area harvested, yield, and production volume.

**File:** `FAOSTAT_crop_data_syria.csv`

#### Column Descriptions

| Column | Description |
| :--- | :--- |
| **Domain Code** | A short code identifying the data domain. For this dataset, it is always `QCL`, which stands for "Crops and Livestock Products". |
| **Domain** | The full name of the data domain: "Crops and livestock products". This indicates the dataset contains agricultural production statistics. |
| **Area Code (M49)** | The United Nations M49 numerical code for the country. For Syria, this is `760`. |
| **Area** | The name of the country or region: "Syrian Arab Republic". |
| **Element Code** | A code representing the type of measurement. Key codes include: <br>• `5312` = Area harvested <br>• `5412` = Yield <br>• `5510` = Production |
| **Element** | The full name of the measurement type (e.g., "Area harvested", "Yield", "Production"). |
| **Item Code (CPC)** | The Central Product Classification code for the crop. For example, `1371` = Almonds, in shell. |
| **Item** | The name of the crop (e.g., "Almonds, in shell", "Wheat", "Olives"). |
| **Year Code** | The year the data was recorded (e.g., `1961`, `2020`). |
| **Year** | The year as a numeric value. |
| **Unit** | The unit of measurement for the value. Common units include: <br>• `ha` = Hectares (for area) <br>• `kg/ha` = Kilograms per hectare (for yield) <br>• `t` = Tonnes (for production) |
| **Value** | The numerical value of the measurement (e.g., `2300` hectares, `739.1` kg/ha). |
| **Flag** | A single-letter code indicating the status or quality of the data point (see below). |
| **Flag Description** | A human-readable description of the flag. |
| **Note** | Any additional notes or comments about the data point (often empty). |

#### Data Flags Explained

| Flag | Description | Meaning |
| :--- | :--- | :--- |
| **A** | Official value | Data reported directly by the national government or official statistical agency. |
| **E** | Estimated value | Data estimated by FAO when official data is unavailable or unreliable. |
| **P** | Provisional value | Data that is preliminary and subject to revision. |
| **I** | Imputed value | Data filled in by FAO using interpolation or other methods. |

> **Note for our project:** We will primarily use records where `Element = "Yield"` as our **target variable**. The `Flag` column helps us understand data reliability; we may choose to exclude `I` (imputed) values in sensitivity analyses.

---

### 2. NASA POWER – Climate Data
**Source:** [NASA POWER Data Access Viewer](https://power.larc.nasa.gov/data-access-viewer/)

This dataset provides daily meteorological data for all 13 Syrian governorates, downloaded using the NASA POWER API. Data spans from **1991 to 2025**.

**Files:**
- `syria_governorates_reference.csv` – Metadata for each governorate
- `nasa_daily_{governorate}.csv` – Raw per-governorate data
- `syria_climate_all_governorates.csv` – Combined long-format dataset
- `syria_climate_country_level.csv` – Aggregated country-level data

#### Variables Downloaded

| Variable | Column Name | Unit | Description |
| :--- | :--- | :--- | :--- |
| **T2M** | `t2m` | °C | Average air temperature at 2 meters above the surface. This is a key driver of crop growth and development. |
| **T2M_MAX** | `t2m_max` | °C | Maximum daily air temperature at 2 meters. Essential for identifying heat stress periods. |
| **T2M_MIN** | `t2m_min` | °C | Minimum daily air temperature at 2 meters. Useful for tracking cold stress and frost risk. |
| **PRECTOTCORR** | `prectotcorr` | mm/day | Corrected total precipitation. This is the primary variable for crop water availability. The "corrected" version adjusts for known biases in satellite estimates. |
| **RH2M** | `rh2m` | % | Relative humidity at 2 meters. Influences evapotranspiration and plant stress. |
| **ALLSKY_SFC_SW_DWN** | `solar_kwh_m2` | kWh/m²/day | All-sky surface shortwave downward irradiance. This represents the total solar energy reaching the surface, a key driver of photosynthesis. |
| **WS2M** | `ws2m` | m/s | Wind speed at 2 meters. Affects evapotranspiration and microclimate. |

#### Governorate Reference File

| Column | Description |
| :--- | :--- |
| `governorate` | The English name of the governorate (e.g., "Aleppo", "Damascus"). |
| `governorate_ar` | The Arabic name of the governorate (e.g., "حلب", "دمشق"). |
| `latitude` | The approximate latitude of the governorate's centroid (in decimal degrees). |
| `longitude` | The approximate longitude of the governorate's centroid (in decimal degrees). |

#### Combined Governorate Dataset (`syria_climate_all_governorates.csv`)

This long-format file contains one row per governorate per day:

| Column | Description |
| :--- | :--- |
| `governorate` | English name of the governorate. |
| `governorate_ar` | Arabic name of the governorate. |
| `latitude` | Latitude of the governorate centroid. |
| `longitude` | Longitude of the governorate centroid. |
| `date` | The observation date (YYYY-MM-DD format). |
| `t2m` – `ws2m` | The climate variables as described in the table above. |

#### Country-Level Dataset (`syria_climate_country_level.csv`)

This file contains daily country-level averages (mean across all 13 governorates) and is ready to be merged with FAOSTAT data.

| Column | Description |
| :--- | :--- |
| `date` | The observation date. |
| `t2m` – `ws2m` | Country-level daily averages of the climate variables. |

---

## 🔗 Merging the Datasets

The FAOSTAT data is **annual** (one value per year per crop), while the NASA POWER data is **daily**. To build our machine learning models, we will:

1.  **Aggregate** the daily NASA POWER data to **annual or seasonal statistics** (e.g., average temperature, total rainfall, average solar radiation).
2.  **Merge** the aggregated climate data with the FAOSTAT crop yield data using the **Year** as the key.

This will produce a final training dataset with:
- **Target:** Crop yield (kg/ha) from FAOSTAT
- **Features:** Climate variables (temperature, rainfall, solar radiation, humidity, wind) from NASA POWER

---

## 📋 Important Data Considerations

### Avoiding Data Leakage
As instructed in the challenge brief:
> *"Do not use production volume or harvested area as predictors when predicting yield unless you can justify their availability at prediction time."*

Since **Yield = Production / Area Harvested**, using either of these as features would leak the target variable. We will only use **climate features** and **crop type/year** as predictors.

### Temporal Consistency
We must ensure that climate data used for a given year is representative of the **growing season** for the crop of interest. For example, wheat in Syria is typically grown from November to June, so we may focus on rainfall during that period.

### Data Quality
- **FAOSTAT flags:** We will monitor the proportion of `E` (estimated) vs. `A` (official) values. In sensitivity analyses, we may test models with only `A`-flagged data.
- **NASA POWER fill values:** Missing data points are coded as `-999.0` and have been replaced with `NaN` in our processing pipeline.

---

## 📂 Project Status

| Task | Status |
| :--- | :---: |
| Data Collection (FAOSTAT + NASA POWER) | ✅ Done |
| Data Cleaning & Preprocessing | 🔄 In Progress |
| Exploratory Data Analysis (EDA) | ⏳ Not Started |
| Feature Engineering (Seasonal Aggregation) | ⏳ Not Started |
| Model Training (Decision Tree, XGBoost) | ⏳ Not Started |
| Model Evaluation (MAE, RMSE, R²) | ⏳ Not Started |
| Feature Importance Analysis | ⏳ Not Started |
| Final Report & Presentation | ⏳ Not Started |

---

## 🛠️ Tools and Technologies

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab
- **Data Sources:** FAOSTAT, NASA POWER

---

## 🙏 Acknowledgements

We would like to thank the **FTL Syria Programme**, **UNDP**, **Damascus University**, and **BMZ/KfW** for this learning opportunity, and the open data providers—**FAO** and **NASA**—for making their datasets publicly available.

---

## 📬 Contact

For questions about this project, please reach out to any team member:

- Ahmad Deeb – [ahmaddeebdev@gmail.com](mailto:ahmaddeebdev@gmail.com) – [GitHub](https://github.com/AhmedGDeeb)