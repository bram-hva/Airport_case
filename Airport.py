"""
Vluchten, luchthavens en weer ophalen en samenvoegen tot één dataframe.

De drie bronnen:
  1. Vluchtschema Zürich Airport  - uit jullie eigen GitHub-repo (of lokaal)
  2. Luchthavens wereldwijd       - via de Kaggle-API
  3. Dagelijks weer Zürich-Kloten - via Meteostat (station 06670)

Resultaat: 'vluchten_compleet.csv', klaar voor het Streamlit-dashboard.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import kagglehub


# ======================================================================
# WAAR WORDT ALLES OPGESLAGEN?
# ======================================================================
# Python slaat standaard op in de map waar je terminal toevallig staat, en
# dat is bij VS Code vaak de installatiemap van VS Code zelf. Deze regel
# zoekt op waar DIT script staat, zodat alles in je streamlit-map belandt.

MAP = Path(__file__).parent
print("Ik werk in de map:", MAP)


# ======================================================================
# HIER VUL JE IETS IN  (de rest hoef je niet aan te raken)
# ======================================================================

# 1) De ruwe link naar het vluchtschema in jullie GitHub-repo.
#    Zo kom je eraan: open het bestand op github.com, klik rechtsboven op
#    'Raw', en kopieer de URL uit de adresbalk. Vervang hieronder alleen
#    REPONAAM door de naam van jullie repo.
#    Staat het bestand er nog niet? Laat het dan staan: het script valt
#    vanzelf terug op het bestand op je eigen laptop.
SCHEDULE_URL = "https://raw.githubusercontent.com/bram-hva/Airport_case/main/schedule_airport.csv.gz"

# 2) Zet op True om het ingepakte bestand voor GitHub te maken.
#    Draai het script, upload het resultaat, en zet hem daarna weer op False.
MAAK_ZIP = False


# Deze twee hoef je niet te veranderen.
SCHEDULE_LOKAAL = MAP / "schedule_airport.csv"
WEER_URL = "https://bulk.meteostat.net/v2/daily/06670.csv.gz"


# ======================================================================
# 1. VLUCHTEN OPHALEN
# ======================================================================
# Eerst GitHub proberen. Lukt dat niet (bestand nog niet geüpload, geen
# internet, typfout in de link), dan pakt hij het bestand op je laptop.
# Zo kun je altijd doorwerken.

try:
    vluchten = pd.read_csv(
        SCHEDULE_URL,
        encoding="utf-8-sig",     # haalt het onzichtbare BOM-teken voor 'STD' weg
        na_values=["-"],          # streepjes zijn ontbrekende waarden
    )
    print("Vluchtschema opgehaald van GitHub")
except Exception as fout:
    print("GitHub lukte niet, ik gebruik het lokale bestand.")
    print("   reden:", fout)
    vluchten = pd.read_csv(
        SCHEDULE_LOKAAL,
        encoding="utf-8-sig",
        na_values=["-"],
    )

# Het lokale bestand schrijft 01-03-2019, het GitHub-bestand 2019-03-01.
# Met format="mixed" leest pandas allebei goed.
vluchten["STD"] = pd.to_datetime(vluchten["STD"], dayfirst=True, format="mixed")

print("Vluchten opgehaald:", vluchten.shape)
print("Periode:", vluchten["STD"].min().date(), "tot", vluchten["STD"].max().date())


# ======================================================================
# 1b. INGEPAKTE KOPIE MAKEN VOOR GITHUB
# ======================================================================
# schedule_airport.csv is 32 MB en GitHub neemt via de website maar 25 MB
# per bestand aan. Ingepakt is het ongeveer 5 MB. pandas pakt een .gz
# vanzelf weer uit, dus de code hierboven hoeft niet te veranderen.

if MAAK_ZIP:
    zip_pad = MAP / "schedule_airport.csv.gz"
    vluchten.to_csv(zip_pad, index=False, compression="gzip")
    print("KLAAR: upload dit bestand naar GitHub ->", zip_pad)


# ======================================================================
# 2. VLUCHTEN OPSCHONEN
# ======================================================================

# Drie Identifiers komen dubbel voor; één rij per vlucht is genoeg.
voor = len(vluchten)
vluchten = vluchten.drop_duplicates(subset=["Identifier"], keep="first")
print(f"Dubbele vluchten verwijderd: {voor - len(vluchten)}")

# Leesbare kolomnamen voor de velden die we echt gebruiken.
vluchten = vluchten.rename(columns={
    "STD": "Datum",
    "FLT": "Vluchtnummer",
    "STA_STD_ltc": "Gepland",
    "ATA_ATD_ltc": "Werkelijk",
    "LSV": "Richting",
    "ACT": "Vliegtuigtype",
    "RWY": "Baan",
    "RWC": "Baanconfiguratie",
    "Org/Des": "ICAO_ander",
})

# L = landing (aankomst), S = start (vertrek)
vluchten["Richting"] = vluchten["Richting"].map({"L": "Aankomst", "S": "Vertrek"})


# ======================================================================
# 3. AFGELEIDE VARIABELE: DE ECHTE VERTRAGING IN MINUTEN
# ======================================================================
# LET OP: de kolommen DL1 en DL2 zijn GEEN minuten maar IATA-vertragingscodes
# (0 t/m 99, bijvoorbeeld 93 = vertraagd toestel uit de vorige vlucht). Daar
# mag je dus niet mee rekenen. De echte vertraging is het verschil tussen de
# geplande en de werkelijke tijd, en die rekenen we hier zelf uit.

gepland = pd.to_datetime(vluchten["Datum"].dt.strftime("%Y-%m-%d") + " " + vluchten["Gepland"])
werkelijk = pd.to_datetime(vluchten["Datum"].dt.strftime("%Y-%m-%d") + " " + vluchten["Werkelijk"])

verschil = (werkelijk - gepland).dt.total_seconds() / 60

# Een vlucht van 23:50 die om 00:10 vertrekt is 20 minuten te laat, niet
# 1420 minuten te vroeg. Deze twee regels corrigeren die middernacht-sprong.
verschil = np.where(verschil < -720, verschil + 1440, verschil)
verschil = np.where(verschil > 720, verschil - 1440, verschil)

vluchten["Vertraging_min"] = np.round(verschil, 1)

# Afgeleide variabelen voor de filters en grafieken in het dashboard
vluchten["Vertraagd"] = vluchten["Vertraging_min"] > 15      # luchtvaartnorm
vluchten["Uur"] = gepland.dt.hour
vluchten["Maand"] = vluchten["Datum"].dt.month
vluchten["Jaar"] = vluchten["Datum"].dt.year
vluchten["Weekdag"] = vluchten["Datum"].dt.day_name()
vluchten["Seizoen"] = vluchten["Maand"].map({12: "Winter", 1: "Winter", 2: "Winter",
                                             3: "Lente", 4: "Lente", 5: "Lente",
                                             6: "Zomer", 7: "Zomer", 8: "Zomer",
                                             9: "Herfst", 10: "Herfst", 11: "Herfst"})

# Een vertraging van meer dan 12 uur is bijna zeker een fout in de data. We
# zetten die op leeg in plaats van de rij weg te gooien, zodat de vlucht wel
# meetelt bij het aantal vluchten.
onmogelijk = vluchten["Vertraging_min"].abs() > 720
print(f"Onmogelijke vertragingen op leeg gezet: {onmogelijk.sum()}")
vluchten.loc[onmogelijk, "Vertraging_min"] = np.nan


# ======================================================================
# 4. LUCHTHAVENS OPHALEN VIA DE KAGGLE-API
# ======================================================================

pad = kagglehub.dataset_download("open-flights/airports-train-stations-and-ferry-terminals")

kolommen_air = ["ID", "Naam", "Stad", "Land", "IATA", "ICAO",
                "Latitude", "Longitude", "Hoogte_ft", "Tijdzone_offset",
                "Zomertijd", "Tijdzone_naam", "Type", "Bron"]

airports = pd.read_csv(f"{pad}/airports-extended.csv",
                       header=None,            # dit bestand heeft geen kopregel
                       names=kolommen_air,
                       na_values=["\\N"])      # zo worden de \N's herkend als leeg

# Alleen echte luchthavens met een ICAO-code; stations en havens laten we weg.
airports = airports[airports["Type"] == "airport"].dropna(subset=["ICAO"])
airports = airports.drop_duplicates(subset=["ICAO"])

airports = airports[["ICAO", "Naam", "Stad", "Land", "Latitude", "Longitude"]].rename(
    columns={"Naam": "Luchthaven", "Latitude": "Lat", "Longitude": "Lon"}
)

print("Luchthavens met ICAO-code:", len(airports))


# ======================================================================
# 5. WEER OPHALEN VIA METEOSTAT
# ======================================================================

kolommen_weer = ["datum", "tavg", "tmin", "tmax", "neerslag", "sneeuw",
                 "windrichting", "windsnelheid", "windstoot", "luchtdruk", "zonuren"]

# pandas pakt de .gz vanzelf uit en leest een URL net zo makkelijk als een bestand.
weer = pd.read_csv(WEER_URL,
                   header=None,              # Meteostat levert geen kopregel
                   names=kolommen_weer,
                   parse_dates=["datum"])

# Deze twee kolommen zijn voor dit station 100% leeg.
weer = weer.drop(columns=["sneeuw", "zonuren"])

# Het weerbestand loopt vanaf 1973; we houden alleen de vluchtperiode over.
weer = weer[weer["datum"].between("2019-01-01", "2020-12-31")]

weer = weer.rename(columns={"datum": "Datum", "tavg": "Temp_gem", "tmin": "Temp_min",
                            "tmax": "Temp_max", "neerslag": "Neerslag_mm",
                            "windrichting": "Windrichting", "windsnelheid": "Wind_kmh",
                            "windstoot": "Windstoot_kmh", "luchtdruk": "Luchtdruk_hPa"})

# Afgeleide weervariabelen: handiger om op te filteren dan losse getallen.
weer["Regendag"] = weer["Neerslag_mm"] > 1
weer["Harde_wind"] = weer["Wind_kmh"] > 20
weer["Vorst"] = weer["Temp_min"] < 0

print("Weerdagen in 2019-2020:", len(weer), "van de 731")


# ======================================================================
# 6. ALLES SAMENVOEGEN
# ======================================================================

rijen_start = len(vluchten)

# Eerst de luchthaven van herkomst of bestemming erbij, via de ICAO-code.
data = vluchten.merge(airports, left_on="ICAO_ander", right_on="ICAO", how="left")
print(f"Na merge met luchthavens: {rijen_start} -> {len(data)}")

# Dan het weer van die dag erbij, via de datum.
data = data.merge(weer, on="Datum", how="left")
print(f"Na merge met weer:        {rijen_start} -> {len(data)}")

data = data.drop(columns=["ICAO"])   # dubbele kolom na de merge


# ======================================================================
# 7. CONTROLE: IS DE KOPPELING GELUKT?
# ======================================================================

print("\n--- Controle ---")
print("Vluchten zonder luchthavengegevens:", data["Land"].isna().sum())
print("Vluchten zonder weergegevens:      ", data["Temp_gem"].isna().sum())
print("Vluchten zonder vertraging:        ", data["Vertraging_min"].isna().sum())

niet_gevonden = data.loc[data["Land"].isna(), "ICAO_ander"].value_counts().head(10)
if not niet_gevonden.empty:
    print("\nICAO-codes die niet in de luchthavenlijst staan:")
    print(niet_gevonden)

print("\n--- Vertraging in minuten ---")
print(data["Vertraging_min"].describe().round(1))

print("\n--- De tien vaakst voorkomende vertragingscodes (DL1) ---")
print(data["DL1"].value_counts().head(10))


# ======================================================================
# 8. OPSLAAN VOOR HET DASHBOARD
# ======================================================================

uit_pad = MAP / "vluchten_compleet.csv"
data.to_csv(uit_pad, index=False)
print("\nOpgeslagen:", uit_pad, data.shape)