"""
Dashboard: vluchten en vertraging op Zurich Airport, 2019-2020.

Draaien met:   python -m streamlit run dashboard_Airport5.py

Staat 'vluchten_compleet.csv' naast dit bestand, dan wordt die gelezen.
Staat hij er niet, dan haalt het dashboard de drie bronnen zelf van
internet en voegt ze samen. Zo draait een verse kopie van de repo zonder
dat iemand eerst Airport.py hoeft te starten.

De kaart is gemaakt met Folium, zoals in de demo van hoorcollege 6:
CircleMarker, een straal die met de wortel schaalt, en FeatureGroup plus
LayerControl om klassen aan en uit te zetten.
"""

import os
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Vluchten en vertraging op Zurich", layout="wide")

MAP = Path(__file__).parent          # map waar dit bestand staat

# De drie bronnen, allemaal van internet.
SCHEDULE_URL = ("https://raw.githubusercontent.com/bram-hva/Airport_case/"
                "main/schedule_airport.csv.gz")
WEER_URL = "https://bulk.meteostat.net/v2/daily/06670.csv.gz"
KAGGLE_SET = "open-flights/airports-train-stations-and-ferry-terminals"

BLAUW = "#2a78d6"
ORANJE = "#eb6834"
LICHTGRIJS = "#c8d0d6"
# Vijf tinten licht naar donker. Een reeks getallen hoort geen regenboog.
TRAPPEN = ["#cfe0f5", "#9dc1eb", "#6ba2e0", "#3a83d6", "#1a5ba8"]

# Bleke ondergrond zonder API-sleutel. Esri wil {z}/{y}/{x}, niet {z}/{x}/{y}.
ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")


# ======================================================================
# DATA: lezen als het kan, zelf bouwen als het moet
# ======================================================================

def bouw_dataset():
    """Haalt de drie bronnen op en voegt ze samen tot een dataframe."""
    import kagglehub

    # Kaggle wil inloggegevens. Op Streamlit Cloud zet je die bij Secrets;
    # lokaal staan ze in je kaggle.json. st.secrets klapt eruit als er
    # helemaal geen secrets zijn, vandaar de try.
    try:
        if "KAGGLE_USERNAME" in st.secrets:
            os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
    except Exception:
        pass

    # --- 1. vluchten ---
    v = pd.read_csv(SCHEDULE_URL, encoding="utf-8-sig", na_values=["-"])
    v["Datum"] = pd.to_datetime(v["STD"], dayfirst=True, format="mixed")
    v = v.drop_duplicates(subset=["Identifier"], keep="first")
    v = v.rename(columns={"FLT": "Vluchtnummer", "STA_STD_ltc": "Gepland",
                          "ATA_ATD_ltc": "Werkelijk", "LSV": "Richting",
                          "ACT": "Vliegtuigtype", "RWY": "Baan",
                          "RWC": "Baanconfiguratie", "Org/Des": "ICAO_ander"})
    v["Richting"] = v["Richting"].map({"L": "Aankomst", "S": "Vertrek"})

    # De echte vertraging: DL1 en DL2 zijn IATA-codes, geen minuten.
    gepland = pd.to_datetime(v["Datum"].dt.strftime("%Y-%m-%d") + " " + v["Gepland"])
    werkelijk = pd.to_datetime(v["Datum"].dt.strftime("%Y-%m-%d") + " " + v["Werkelijk"])
    verschil = (werkelijk - gepland).dt.total_seconds() / 60
    # Een vlucht van 23:55 die om 00:10 landt is 15 min te laat, niet 23 uur te vroeg.
    verschil = np.where(verschil < -720, verschil + 1440, verschil)
    verschil = np.where(verschil > 720, verschil - 1440, verschil)
    v["Vertraging_min"] = np.round(verschil, 1)
    v.loc[v["Vertraging_min"].abs() > 720, "Vertraging_min"] = np.nan

    v["Vertraagd"] = v["Vertraging_min"] > 15      # luchtvaartnorm
    v["Uur"] = gepland.dt.hour
    v["Maand"] = v["Datum"].dt.month
    v["Jaar"] = v["Datum"].dt.year
    v["Weekdag"] = v["Datum"].dt.day_name()
    v["Seizoen"] = v["Maand"].map({12: "Winter", 1: "Winter", 2: "Winter",
                                   3: "Lente", 4: "Lente", 5: "Lente",
                                   6: "Zomer", 7: "Zomer", 8: "Zomer",
                                   9: "Herfst", 10: "Herfst", 11: "Herfst"})

    # --- 2. luchthavens via de Kaggle-API ---
    pad = kagglehub.dataset_download(KAGGLE_SET)
    kolommen = ["ID", "Naam", "Stad", "Land", "IATA", "ICAO", "Latitude",
                "Longitude", "Hoogte_ft", "Tijdzone_offset", "Zomertijd",
                "Tijdzone_naam", "Type", "Bron"]
    a = pd.read_csv(f"{pad}/airports-extended.csv", header=None,
                    names=kolommen, na_values=["\\N"])
    a = a[a["Type"] == "airport"].dropna(subset=["ICAO"])
    a = a.drop_duplicates(subset=["ICAO"])
    a = a[["ICAO", "Naam", "Stad", "Land", "Latitude", "Longitude"]].rename(
        columns={"Naam": "Luchthaven", "Latitude": "Lat", "Longitude": "Lon"})

    # --- 3. weer via Meteostat ---
    kol_weer = ["Datum", "Temp_gem", "Temp_min", "Temp_max", "Neerslag_mm",
                "sneeuw", "Windrichting", "Wind_kmh", "Windstoot_kmh",
                "Luchtdruk_hPa", "zonuren"]
    w = pd.read_csv(WEER_URL, header=None, names=kol_weer, parse_dates=["Datum"])
    w = w.drop(columns=["sneeuw", "zonuren"])      # voor dit station 100% leeg
    w = w[w["Datum"].between("2019-01-01", "2020-12-31")]
    w["Regendag"] = w["Neerslag_mm"] > 1
    w["Harde_wind"] = w["Wind_kmh"] > 20
    w["Vorst"] = w["Temp_min"] < 0

    # --- samenvoegen ---
    df = v.merge(a, left_on="ICAO_ander", right_on="ICAO", how="left")
    df = df.merge(w, on="Datum", how="left").drop(columns=["ICAO"])
    return df


@st.cache_data(show_spinner="Data ophalen en samenvoegen, even geduld...")
def laad():
    lokaal = MAP / "vluchten_compleet.csv"
    if lokaal.exists():
        df = pd.read_csv(lokaal, parse_dates=["Datum"])
    else:
        df = bouw_dataset()
    # Drukte: hoeveel bewegingen in hetzelfde geplande uur? Deze kolom staat
    # in geen van de drie bronbestanden; die maken we zelf.
    blok = df["Datum"].dt.strftime("%Y-%m-%d") + " " + df["Uur"].astype(str)
    df["Drukte"] = blok.map(blok.value_counts())

    # Geheugen opruimen. Streamlit Cloud geeft 1 GB en dit bestand is met
    # losse tekstkolommen 383 MB. Kolommen die we nergens gebruiken gaan
    # eruit, en kolommen met weinig verschillende waarden worden 'category':
    # pandas bewaart dan de woorden een keer en per rij alleen een nummer.
    # Samen scheelt dat een factor vijf.
    df = df.drop(columns=[k for k in ["Identifier", "STD", "TAR", "GAT",
                                      "IX1", "IX2"] if k in df.columns])
    for k in ["Richting", "Seizoen", "Weekdag", "Vliegtuigtype", "Baan",
              "Baanconfiguratie", "ICAO_ander", "Luchthaven", "Stad", "Land",
              "Gepland", "Werkelijk"]:
        if k in df.columns:
            df[k] = df[k].astype("category")
    return df


def straal(waarde, laagste, hoogste, klein=4, groot=20):
    """Waarde naar straal in pixels. Via de wortel, want het oog
    vergelijkt oppervlakte en geen straal."""
    if hoogste == laagste:
        return (klein + groot) / 2
    deel = min(max((waarde - laagste) / (hoogste - laagste), 0.0), 1.0)
    return klein + (groot - klein) * deel ** 0.5


data = laad()

# ======================================================================
# FILTERS
# ======================================================================

with st.sidebar:
    st.header("Filters")
    st.caption("Alles staat open: het beeld hiernaast klopt ook zonder "
               "dat je iets aanraakt.")
    jaar = st.multiselect("Jaar", sorted(data["Jaar"].unique()),
                          default=sorted(data["Jaar"].unique()))
    richting = st.multiselect("Aankomst of vertrek", ["Aankomst", "Vertrek"],
                              default=["Aankomst", "Vertrek"])
    st.divider()
    st.caption("Bronnen: vluchtschema Zurich 2019-2020 (school), luchthavens "
               "via de Kaggle-API (OpenFlights), dagelijks weer van Meteostat "
               "station 06670 (Zurich-Kloten).")

sel = data[data["Jaar"].isin(jaar) & data["Richting"].isin(richting)].copy()

if sel.empty:
    st.warning("Geen vluchten in deze selectie.")
    st.stop()


# ======================================================================
# EERSTE LAAG: meteen zichtbaar waar het over gaat
# ======================================================================

st.title("Vluchten en vertraging op Zurich Airport")
st.subheader("Plant de luchthaven zijn eigen vertraging, of komt die van buiten?")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Vluchten", f"{len(sel):,}".replace(",", "."))
k2.metric("Mediane vertraging", f"{sel['Vertraging_min'].median():.1f} min")
k3.metric("Te laat (meer dan 15 min)", f"{sel['Vertraagd'].mean() * 100:.1f}%")
k4.metric("Bestemmingen", sel["ICAO_ander"].nunique())

# Alle dagen in de periode, ook die zonder vluchten. Die worden NaN en dus
# een gat in de lijn, in plaats van een rechte streep die suggereert dat er
# gewoon doorgevlogen werd.
dagen = pd.date_range(sel["Datum"].min(), sel["Datum"].max(), freq="D")
per_week = (sel.groupby("Datum").size().reindex(dagen)
            .resample("W").sum(min_count=1).reset_index())
per_week.columns = ["Week", "Vluchten"]

if len(jaar) == 2:
    kop = (f"Van {per_week['Vluchten'].max():.0f} vluchten in de drukste week "
           f"naar {per_week['Vluchten'].min():.0f} in de stilste")
else:
    kop = "Vluchten per week"

fig0 = px.line(per_week, x="Week", y="Vluchten", title=kop,
               labels={"Week": "", "Vluchten": "Vluchten per week"})
fig0.update_traces(line=dict(color=BLAUW, width=2), connectgaps=False)
fig0.update_layout(height=300, margin=dict(t=60, b=20))
st.plotly_chart(fig0, use_container_width=True)
st.caption("Weken zonder vluchten zijn onderbrekingen, geen rechte "
           "verbinding. Dat verschil is het hele voorjaar van 2020.")

st.divider()


# ======================================================================
# TWEEDE LAAG: detail, pas als je erom vraagt
# ======================================================================

tab_tijd, tab_kaart, tab_drukte, tab_weer, tab_voorspel, tab_data = st.tabs(
    ["Door de tijd", "Op de kaart", "Drukte", "Weer en baan",
     "Voorspelling", "Over de data"])


# ---------------------------------------------------------------- TIJD
with tab_tijd:
    kol1, kol2 = st.columns(2)
    stap = kol1.radio("Per", ["Dag", "Week", "Maand"], index=1, horizontal=True)
    maat = kol2.radio("Toon", ["Aantal vluchten", "Aandeel te laat"],
                      horizontal=True)
    regel = {"Dag": "D", "Week": "W", "Maand": "MS"}[stap]
    basis = sel.set_index("Datum")

    if maat == "Aantal vluchten":
        reeks = (basis.groupby("Richting").resample(regel).size()
                 .rename("waarde").reset_index())
        # Een nul betekent hier "geen vluchten", en dat is echt een gat.
        reeks["waarde"] = reeks["waarde"].replace(0, np.nan)
        y_titel = f"Vluchten per {stap.lower()}"
        titel = "Aankomsten en vertrekken lopen bijna gelijk op"
    else:
        reeks = (basis.groupby("Richting")["Vertraagd"].resample(regel)
                 .mean().mul(100).rename("waarde").reset_index())
        y_titel = "Aandeel te laat (%)"
        top = reeks["waarde"].max()
        titel = f"De slechtste {stap.lower()} haalde {top:.0f}% te late vluchten"

    fig1 = px.line(reeks, x="Datum", y="waarde", color="Richting",
                   color_discrete_map={"Aankomst": BLAUW, "Vertrek": ORANJE},
                   title=titel,
                   labels={"waarde": y_titel, "Datum": "", "Richting": ""})
    fig1.update_traces(connectgaps=False, line=dict(width=2))
    fig1.update_layout(height=430)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(f"Samengevat per {stap.lower()}. Per dag zie je het weekritme, "
               "per maand de seizoenen. Die keuze bepaalt welk patroon je ziet.")

    # ------------------------------------------------------------------
    # VERTRAGING PER UUR VAN DE DAG
    # ------------------------------------------------------------------
    st.divider()

    # Percentage te laat per gepland uur, per jaar en per richting.
    per_uur = (sel.groupby(["Jaar", "Uur", "Richting"], observed=True)
               .agg(vluchten=("Vertraagd", "size"), te_laat=("Vertraagd", "mean"))
               .reset_index())
    per_uur["te_laat"] = (per_uur["te_laat"] * 100).round(1)
    # Uren met weinig vluchten weglaten (5 uur is één vaste vlucht van 05:45).
    per_uur = per_uur[per_uur["vluchten"] >= 300]

    fig_uur = px.bar(per_uur, x="Uur", y="te_laat", color="Richting",
                     barmode="group",
                     facet_col="Jaar",              # 2019 en 2020 naast elkaar
                     color_discrete_map={"Aankomst": BLAUW, "Vertrek": ORANJE},
                     title=("In 2019 stapelt vertraging zich op over de dag, "
                            "in het rustige coronajaar bijna niet"
                            if sel["Jaar"].nunique() == 2
                            else f"Te laat per uur in {sel['Jaar'].iloc[0]}"),
                     labels={"Uur": "Gepland uur (vertrek of landing in Zürich)",
                             "te_laat": "Te laat (%)", "Richting": ""},
                     hover_data={"vluchten": True})
    fig_uur.update_xaxes(dtick=1)
    fig_uur.update_layout(height=430)
    st.plotly_chart(fig_uur, use_container_width=True)
    st.caption("Te laat = meer dan 15 minuten na de geplande tijd. Uren met "
               "minder dan 300 vluchten zijn weggelaten, zoals 5 uur: dat is "
               "één vaste vlucht van 05:45. Door het nachtvluchtverbod begint "
               "de dag in Zürich pas om 6 uur.")

# --------------------------------------------------------------- KAART
with tab_kaart:
    kol1, kol2 = st.columns(2)
    gebied = kol1.radio("Gebied", ["Europa", "Wereld"], horizontal=True)
    drempel = kol2.slider("Minimaal aantal vluchten", 1, 500, 25,
                          help="Een bestemming met drie vluchten heeft een "
                               "mediaan die nergens op slaat.")

    # Mediaan en geen gemiddelde: de verdeling van vertraging heeft een
    # lange staart naar rechts.
    per_best = (sel.groupby("ICAO_ander")
                .agg(luchthaven=("Luchthaven", "first"), stad=("Stad", "first"),
                     land=("Land", "first"), lat=("Lat", "first"),
                     lon=("Lon", "first"), vluchten=("Vertraging_min", "size"),
                     mediaan=("Vertraging_min", "median"),
                     te_laat=("Vertraagd", "mean"))
                .reset_index())
    per_best["te_laat"] = (per_best["te_laat"] * 100).round(1)
    geen_plek = per_best["lat"].isna().sum()

    op_kaart = (per_best[per_best["vluchten"] >= drempel]
                .dropna(subset=["lat", "lon", "mediaan"]).copy())
    europa = op_kaart["lon"].between(-25, 45) & op_kaart["lat"].between(34, 72)
    if gebied == "Europa":
        op_kaart, midden, zoom = op_kaart[europa], [50, 10], 4
    else:
        midden, zoom = [25, 10], 2

    if op_kaart.empty:
        st.warning("Geen bestemmingen over. Zet de drempel lager.")
    else:
        # Vijf even grote groepen. Zo gaan we om met de scheve verdeling:
        # elke kleur krijgt evenveel bestemmingen, dus een uitschieter kan
        # niet de hele schaal opeisen.
        op_kaart["klasse"] = pd.qcut(op_kaart["mediaan"],
                                     q=min(5, op_kaart["mediaan"].nunique()),
                                     labels=False, duplicates="drop")
        grenzen = op_kaart.groupby("klasse")["mediaan"].agg(["min", "max"]).round(0)
        laagste, hoogste = op_kaart["vluchten"].min(), op_kaart["vluchten"].max()

        kaart = folium.Map(location=midden, zoom_start=zoom,
                           tiles=ESRI, attr="Tiles &copy; Esri")

        # Een laag per kleurklasse: zo is het lagenmenu meteen je legenda
        # en kan de lezer klassen aan- en uitzetten.
        lagen = {k: folium.FeatureGroup(
                    name=f"{grenzen.loc[k, 'min']:.0f} tot "
                         f"{grenzen.loc[k, 'max']:.0f} min")
                 for k in sorted(op_kaart["klasse"].unique())}

        for _, r in op_kaart.iterrows():
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=straal(r["vluchten"], laagste, hoogste),
                color="white", weight=1, fill=True,
                fill_color=TRAPPEN[int(r["klasse"])], fill_opacity=0.85,
                tooltip=f"{r['stad']} — {r['mediaan']:.0f} min",
                popup=folium.Popup(
                    f"<b>{r['luchthaven']}</b><br>{r['stad']}, {r['land']}<br>"
                    f"{int(r['vluchten'])} vluchten<br>"
                    f"mediaan {r['mediaan']:.1f} min<br>"
                    f"{r['te_laat']:.0f}% te laat", max_width=230),
            ).add_to(lagen[r["klasse"]])

        for laag in lagen.values():
            laag.add_to(kaart)
        folium.LayerControl(collapsed=False).add_to(kaart)

        slechtste = op_kaart.nlargest(1, "mediaan").iloc[0]
        st.markdown(f"**{slechtste['stad']} heeft de langste mediane "
                    f"vertraging: {slechtste['mediaan']:.0f} minuten over "
                    f"{int(slechtste['vluchten'])} vluchten.**")
        # Een Folium-kaart is gewoon HTML, dus die geven we aan Streamlit door.
        st.components.v1.html(kaart._repr_html_(), height=560)
        st.caption(
            "Grootte is het aantal vluchten, via de wortel. Kleur is de "
            "mediane vertraging in vijf even grote groepen, licht naar donker. "
            "Klik een stip aan voor details; zet klassen aan en uit in het menu "
            f"rechtsboven. {geen_plek} bestemmingen hebben geen coordinaat "
            "(BER, ISL) en staan niet op de kaart.")


# -------------------------------------------------------------- DRUKTE
with tab_drukte:
    # ------------------------------------------------------------------
    # HET NATUURLIJKE EXPERIMENT
    # ------------------------------------------------------------------
    # Elke stip is een maand. Corona haalde de drukte weg en liet het weer
    # staan. Als drukte de oorzaak is van vertraging, dan moeten de lege
    # maanden van 2020 linksonder liggen en de volle maanden van 2019
    # rechtsboven. Dat is een voorspelling die de data kan weerleggen.

    per_mnd = (sel.groupby(sel["Datum"].dt.to_period("M"))
               .agg(vluchten=("Vertraging_min", "size"),
                    te_laat=("Vertraagd", "mean"))
               .reset_index())
    per_mnd["te_laat"] *= 100
    per_mnd["maand"] = per_mnd["Datum"].dt.strftime("%b %Y")
    per_mnd["jaar"] = per_mnd["Datum"].dt.year.astype(str)

    if len(per_mnd) > 3:
        r = per_mnd["vluchten"].corr(per_mnd["te_laat"])

        # Alleen de vier uitersten krijgen een naam; 24 labels wordt soep.
        uitersten = pd.concat([per_mnd.nlargest(2, "te_laat"),
                               per_mnd.nsmallest(2, "te_laat")]).index
        per_mnd["label"] = ""
        per_mnd.loc[uitersten, "label"] = per_mnd.loc[uitersten, "maand"]

        fig2 = px.scatter(
            per_mnd, x="vluchten", y="te_laat", color="jaar", text="label",
            color_discrete_map={"2019": BLAUW, "2020": ORANJE},
            title=f"Hoe voller de maand, hoe later de vluchten (r = {r:.2f})",
            labels={"vluchten": "Vluchten in die maand",
                    "te_laat": "Te laat (%)", "jaar": ""},
            hover_name="maand")
        # Rechte lijn door de punten met polyfit: geen extra pakket nodig.
        a, b = np.polyfit(per_mnd["vluchten"], per_mnd["te_laat"], 1)
        x_lijn = np.linspace(per_mnd["vluchten"].min(),
                             per_mnd["vluchten"].max(), 50)
        fig2.add_scatter(x=x_lijn, y=a * x_lijn + b, mode="lines",
                         line=dict(color="#8a8880", width=2, dash="dash"),
                         name="trend", hoverinfo="skip")
        fig2.update_traces(marker=dict(size=13), textposition="top center",
                           selector=dict(mode="markers+text"))
        fig2.update_layout(height=430)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            f"Elke stip is een maand. Per 1.000 extra vluchten komt er "
            f"{a * 1000:.1f} procentpunt te late vluchten bij. Corona is hier "
            "een natuurlijk experiment: de drukte viel weg, het weer bleef, en "
            "de vertraging viel mee weg. Let wel op december 2020: weinig "
            "vluchten en toch 19% te laat. Drukte verklaart veel, niet alles.")

        st.divider()

    sel["Drukteklasse"] = pd.cut(sel["Drukte"], [0, 20, 30, 40, 50, 60, 1000],
                                 labels=["tot 20", "20-30", "30-40",
                                         "40-50", "50-60", "60+"])
    per_klasse = (sel.dropna(subset=["Drukteklasse"])
                  .groupby("Drukteklasse", observed=True)
                  .agg(vluchten=("Vertraging_min", "size"),
                       te_laat=("Vertraagd", "mean"),
                       mediaan=("Vertraging_min", "median")).reset_index())
    per_klasse["te_laat"] = (per_klasse["te_laat"] * 100).round(1)

    titel3 = (f"Van {per_klasse['te_laat'].iloc[0]:.0f}% naar "
              f"{per_klasse['te_laat'].iloc[-1]:.0f}% te late vluchten "
              "als het druk wordt") if len(per_klasse) > 1 else "Drukte"

    fig3 = px.bar(per_klasse, x="Drukteklasse", y="te_laat", title=titel3,
                  text="te_laat",
                  labels={"Drukteklasse": "Bewegingen in dat uur",
                          "te_laat": "Te laat (%)"})
    fig3.update_traces(marker_color=BLAUW, texttemplate="%{text:.0f}%",
                       textposition="outside")
    fig3.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("'Drukte' is een kolom die we zelf hebben gemaakt: het aantal "
               "bewegingen in hetzelfde geplande uur. Die staat in geen van de "
               "drie bronbestanden.")


# ----------------------------------------------------------------- WEER
with tab_weer:
    per_config = (sel.dropna(subset=["Baanconfiguratie"])
                  .groupby("Baanconfiguratie")
                  .agg(vluchten=("Vertraging_min", "size"),
                       mediaan=("Vertraging_min", "median"),
                       te_laat=("Vertraagd", "mean")).reset_index())
    per_config["te_laat"] = (per_config["te_laat"] * 100).round(1)
    per_config = per_config[per_config["vluchten"] >= 200]

    # 'Night' hoort hier niet. De configuratie wordt vastgelegd op het moment
    # van de WERKELIJKE beweging. Die vluchten stonden gepland rond 22 uur en
    # landden na 23 uur: ze kregen dat label omdat ze te laat waren, niet
    # andersom. Night voorspelt vertraging niet, het meet hem.
    per_config = per_config[per_config["Baanconfiguratie"] != "Night"]
    per_config = per_config.sort_values("te_laat")

    if not per_config.empty:
        slecht, goed = per_config.iloc[-1], per_config.iloc[0]
        fig5 = px.bar(per_config, x="te_laat", y="Baanconfiguratie",
                      orientation="h",
                      title=(f"Bij {slecht['Baanconfiguratie']} is "
                             f"{slecht['te_laat']:.0f}% te laat, bij "
                             f"{goed['Baanconfiguratie']} {goed['te_laat']:.0f}%"),
                      labels={"te_laat": "Te laat (%)", "Baanconfiguratie": ""},
                      hover_data={"vluchten": True, "mediaan": ":.1f"})
        fig5.update_traces(marker_color=BLAUW)
        fig5.update_layout(height=380)
        st.plotly_chart(fig5, use_container_width=True)

    st.info(
        "**Bise** is de koude noordoostenwind in Zwitserland. Waait die, dan "
        "moet Zurich landen op banen met minder capaciteit. De oorzaak staat "
        "in het weerbestand, het gevolg in het vluchtbestand.\n\n"
        "De configuratie **Night** staat hier niet bij. Die haalt 67% te laat, "
        "maar dat is een cirkelredenering: de kolom legt vast welke "
        "configuratie draaide op het moment van de *werkelijke* beweging. "
        "Die vluchten kregen dat label omdat ze te laat waren.")

    # Per vlucht kan de configuratie door de vertraging zijn bepaald. Per dag
    # niet: een dag is of een Bise-dag of niet, en dan tellen alle vluchten
    # van die dag mee, ook die van voor en na de Bise-uren.
    per_datum = sel.groupby("Datum").agg(vluchten=("Vertraging_min", "size"),
                                         te_laat=("Vertraagd", "mean"))
    bise = (sel[sel["Baanconfiguratie"].astype(str).str.startswith("Bise")]
            .groupby("Datum").size())
    per_datum["soort"] = np.where(
        bise.reindex(per_datum.index).fillna(0) >= 20, "Bise-dag", "gewone dag")
    per_datum["te_laat"] = per_datum["te_laat"] * 100
    samen = per_datum.groupby("soort")["te_laat"].agg(["size", "mean"])

    if len(samen) == 2:
        b, g = samen.loc["Bise-dag", "mean"], samen.loc["gewone dag", "mean"]
        fig5b = px.box(per_datum.reset_index(), x="soort", y="te_laat",
                       color="soort", points=False,
                       color_discrete_map={"Bise-dag": ORANJE,
                                           "gewone dag": BLAUW},
                       title=(f"Ook per dag blijft het verschil staan: "
                              f"{b:.0f}% tegen {g:.0f}% te late vluchten"),
                       labels={"te_laat": "Te laat die dag (%)", "soort": ""})
        fig5b.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig5b, use_container_width=True)
        st.caption(
            f"Een Bise-dag is een dag met minstens 20 vluchten onder een "
            f"Bise-configuratie ({int(samen.loc['Bise-dag', 'size'])} dagen). "
            "Per vlucht gemeten is het verschil 21 procentpunt, per dag nog 5. "
            "Het eerlijke getal is het kleinere: die vergelijking kan niet door "
            "de vertraging zelf zijn beinvloed.")


# ---------------------------------------------------------- VOORSPELLING
with tab_voorspel:
        # ------------------------------------------------------------------
    # VOORSPELLING VAN DE VERTRAGING
    # Opgesteld met hulp van Claude (AI) en aangepast aan onze data.
    # ------------------------------------------------------------------
    st.subheader("Kun je voorspellen hoeveel vluchten op een dag te laat zijn?")
    st.write("Het model leert op **70% van de dagen** en wordt getoetst op de "
             "**andere 30%**, dagen die het nooit heeft gezien. Zo is de toets eerlijk.")

    # 1. Eén rij per dag: hoeveel procent te laat, hoe druk, en het weer.
    dag = (sel.groupby("Datum")
           .agg(te_laat=("Vertraagd", "mean"),
                vluchten=("Vertraagd", "size"),
                wind=("Wind_kmh", "first"),
                regen=("Neerslag_mm", "first"),
                temp_min=("Temp_min", "first"))
           .dropna()
           .reset_index())
    dag["te_laat"] = dag["te_laat"] * 100
    dag["vorst"] = (dag["temp_min"] < 0).astype(int)     # 1 = vorst, 0 = geen vorst
    dag["jaar"] = dag["Datum"].dt.year.astype(str)
    # Dagen met minder dan 50 vluchten weglaten: een percentage over een
    # handvol vluchten springt alle kanten op.
    dag = dag[dag["vluchten"] >= 50]

    # 2. De gebruiker kiest zelf welke factoren het model mag gebruiken.
    FACTOREN = {"Drukte (vluchten per dag)": "vluchten",
                "Wind (km/u)": "wind",
                "Regen (mm)": "regen",
                "Vorst (ja/nee)": "vorst"}
    gekozen = st.multiselect("Welke factoren gebruikt het model?",
                             list(FACTOREN), default=list(FACTOREN),
                             help="Haal een factor weg en kijk hoeveel slechter "
                                  "het model wordt. Zo zie je wat vertraging voorspelt.")

    # Vaste willekeurige verdeling (seed 42), zodat iedereen hetzelfde ziet.
    is_train = np.random.default_rng(42).random(len(dag)) < 0.7
    train = dag[is_train]
    toets = dag[~is_train].copy()

    if not gekozen:
        st.warning("Kies minstens één factor.")
    else:
        kol = [FACTOREN[f] for f in gekozen]

        # 3. Lineaire regressie: de beste rechte lijn door alle factoren tegelijk.
        #    np.linalg.lstsq zoekt de gewichten waarbij de fout het kleinst is.
        #    De kolom met enen is het startgetal (het snijpunt).
        X_train = np.column_stack([np.ones(len(train)), train[kol]])
        gewichten, *_ = np.linalg.lstsq(X_train, train["te_laat"], rcond=None)

        X_toets = np.column_stack([np.ones(len(toets)), toets[kol]])
        toets["voorspeld"] = np.clip(X_toets @ gewichten, 0, 100)
        toets["fout"] = toets["voorspeld"] - toets["te_laat"]

        # 4. Hoe goed is hij? Vergelijk met steeds het gemiddelde gokken.
        fout_model = toets["fout"].abs().mean()
        fout_gok = (toets["te_laat"] - train["te_laat"].mean()).abs().mean()

        # Welke factor weegt het zwaarst? Gewicht maal de spreiding van die factor,
        # zodat km/u, mm en vluchten met elkaar te vergelijken zijn.
        effect = {f: abs(g) * train[FACTOREN[f]].std()
                  for f, g in zip(gekozen, gewichten[1:])}
        zwaarst = max(effect, key=effect.get)

        m1, m2, m3 = st.columns(3)
        m1.metric("Gemiddelde fout van het model", f"{fout_model:.1f} procentpunt")
        m2.metric("Fout als je steeds het gemiddelde gokt",
                  f"{fout_gok:.1f} procentpunt")
        m3.metric("Zwaarste factor", zwaarst.split(" (")[0])

        # 5. Grafiek: elke stip is een toetsdag. Op de stippellijn is de
        #    voorspelling precies goed; hoe verder ervan af, hoe groter de fout.
        slecht = toets.assign(abs_fout=toets["fout"].abs()).nlargest(5, "abs_fout")
        toets["label"] = ""
        toets.loc[slecht.index, "label"] = toets.loc[slecht.index, "Datum"].dt.strftime("%d %b %Y")

        fig8 = px.scatter(toets, x="voorspeld", y="te_laat", color="jaar",
                          text="label",
                          color_discrete_map={"2019": BLAUW, "2020": ORANJE},
                          hover_data={"Datum": "|%d %b %Y", "vluchten": True,
                                      "label": False},
                          title=(f"Het model zit gemiddeld {fout_model:.1f} procentpunt "
                                 f"naast; gokken zou {fout_gok:.1f} naast zitten"),
                          labels={"voorspeld": "Voorspeld te laat (%)",
                                  "te_laat": "Werkelijk te laat (%)", "jaar": ""})
        grens = toets["voorspeld"].max() + 5
        fig8.add_scatter(x=[0, grens], y=[0, grens], mode="lines",
                         line=dict(color="#8a8880", dash="dash"),
                         name="perfecte voorspelling", hoverinfo="skip")
        fig8.update_traces(textposition="top center", marker=dict(size=8),
                           selector=dict(mode="markers+text"))
        fig8.update_layout(height=480)
        st.plotly_chart(fig8, use_container_width=True)
        st.caption("Elke stip is een dag die het model niet heeft gezien. Boven de "
                   "stippellijn: het werd erger dan voorspeld. Eronder: het viel mee. "
                   "De vijf grootste missers hebben een datum. Haal hierboven een "
                   "factor weg om te zien hoeveel die bijdraagt.")

        # 6. Waar zit hij ernaast? De vijf dagen met de grootste fout.
        with st.expander("De vijf dagen waarop het model er het verst naast zat"):
            tabel = slecht[["Datum", "vluchten", "wind", "regen", "vorst",
                            "te_laat", "voorspeld", "fout"]].copy()
            tabel["Datum"] = tabel["Datum"].dt.strftime("%d-%m-%Y")
            tabel = tabel.rename(columns={
                "vluchten": "Vluchten", "wind": "Wind (km/u)", "regen": "Regen (mm)",
                "vorst": "Vorst", "te_laat": "Werkelijk (%)",
                "voorspeld": "Voorspeld (%)", "fout": "Fout (pp)"})
            st.dataframe(tabel.round(1), use_container_width=True, hide_index=True)
            st.caption("Negatieve fout: het werd erger dan het model verwachtte, "
                       "bijvoorbeeld door iets dat niet in de data staat, zoals een "
                       "storing of staking. Positief: het viel mee.")

    st.divider()


    st.subheader("Hoe snel zou Zurich herstellen van de eerste golf?")
    st.write("We trekken een rechte lijn door een paar maanden van 2020 en "
             "kijken wat die voorspelt voor de maanden die er niet in zaten. "
             "Die maanden zijn dus een echte toets.")

    NAAM = ["jan", "feb", "mrt", "apr", "mei", "jun",
            "jul", "aug", "sep", "okt", "nov", "dec"]
    per_maand = (data[data["Jaar"] == 2020].groupby("Maand").size()
                 .reindex(range(1, 13), fill_value=0))
    niveau_2019 = data[data["Jaar"] == 2019].groupby("Maand").size().mean()

    start, eind = st.select_slider(
        "Op welke maanden baseer je de trend?", options=list(range(1, 13)),
        value=(4, 8), format_func=lambda m: NAAM[m - 1])

    if eind - start < 1:
        st.warning("Kies minstens twee maanden.")
    else:
        # polyfit met graad 1 geeft de helling en het snijpunt van de beste
        # rechte lijn door die punten. Meer is lineaire regressie niet.
        xt = np.arange(start, eind + 1)
        yt = per_maand.loc[start:eind].to_numpy()
        helling, snijpunt = np.polyfit(xt, yt, 1)
        maanden = np.arange(1, 13)
        # Bandbreedte: hoe ver lagen de trainpunten zelf van de lijn af?
        rest = yt - (helling * xt + snijpunt)
        band = 1.96 * rest.std(ddof=1) if len(rest) > 2 else 0.0

        vgl = pd.DataFrame({
            "maand": [NAAM[m - 1] for m in maanden], "nr": maanden,
            "werkelijk": per_maand.to_numpy(),
            "voorspeld": (helling * maanden + snijpunt).round(0),
            "rol": ["gebruikt voor de trend" if start <= m <= eind else "toets"
                    for m in maanden]})
        vgl["fout"] = (vgl["voorspeld"] - vgl["werkelijk"]).round(0)

        v1, v2, v3 = st.columns(3)
        v1.metric("Groei volgens de trend",
                  f"{helling:+,.0f} per maand".replace(",", "."))
        v2.metric("Niveau 2019", f"{niveau_2019:,.0f} per maand".replace(",", "."))
        v3.metric("Maanden tot 2019-niveau",
                  f"{(niveau_2019 - yt[-1]) / helling:.0f}" if helling > 0
                  else "trend daalt")

        na = vgl[(vgl["rol"] == "toets") & (vgl["nr"] > eind)]
        if helling > 0 and len(na):
            e = na.loc[na["fout"].idxmax()]
            titel7 = (f"De trend voorspelt {int(e['voorspeld']):,} vluchten in "
                      f"{e['maand']}, het werden er "
                      f"{int(e['werkelijk']):,}").replace(",", ".")
        else:
            titel7 = "Vluchten per maand in 2020, met de doorgetrokken lijn"

        fig7 = px.bar(vgl, x="maand", y="werkelijk", color="rol", title=titel7,
                      category_orders={"maand": NAAM,
                                       "rol": ["gebruikt voor de trend", "toets"]},
                      color_discrete_map={"gebruikt voor de trend": BLAUW,
                                          "toets": LICHTGRIJS},
                      labels={"werkelijk": "Vluchten", "maand": "Maand 2020",
                              "rol": ""})
        fig7.add_scatter(x=vgl["maand"], y=vgl["voorspeld"], mode="lines",
                         line=dict(color=ORANJE, width=3), name="trend")
        if band > 0:
            for kant, toon in [(band, True), (-band, False)]:
                fig7.add_scatter(x=vgl["maand"], y=vgl["voorspeld"] + kant,
                                 mode="lines", showlegend=toon,
                                 name="bandbreedte (95%)",
                                 line=dict(color=ORANJE, width=1, dash="dot"))
        fig7.update_layout(height=430)
        st.plotly_chart(fig7, use_container_width=True)

        st.markdown(
            "**Wat deze voorspelling breekt:** de tweede coronagolf van "
            "oktober 2020. Een rechte lijn gaat ervan uit dat de "
            "omstandigheden gelijk blijven. Dat is de aanname.")

        with st.expander("Maand voor maand"):
            st.dataframe(vgl[["maand", "werkelijk", "voorspeld", "fout", "rol"]],
                         use_container_width=True, hide_index=True)


# ----------------------------------------------------------- OVER DE DATA
with tab_data:
    st.markdown("""
**Weggehaald**

- **3 dubbele vluchten.** Drie Identifiers kwamen twee keer voor.
- **Vertragingen van meer dan 12 uur** zijn op leeg gezet in plaats van de rij
  weg te gooien, zodat de vlucht wel meetelt als vlucht.

**Expres laten staan**

- **Vertragingen tussen een en twaalf uur.** Die zien er extreem uit, maar de
  vertragingscodes erbij (41 technisch, 96 bemanning) laten zien dat het echte
  gebeurtenissen zijn.
- **Bestemmingen zonder coordinaat**, vooral BER (geopend eind 2020) en ISL.
  Niet op de kaart, wel in alle cijfers.

**Twee valkuilen**

- **DL1 en DL2 zijn geen minuten** maar IATA-codes van 0 tot 99. Code 93,
  "vertraagd toestel uit de vorige vlucht", staat bij 54.937 vluchten. Het
  gemiddelde van die kolom is betekenisloos; de vertraging hebben we zelf
  berekend uit het verschil tussen de geplande en de werkelijke tijd.
- **Middernacht.** Een vlucht gepland om 23:55 die om 00:10 landt is 15 minuten
  te laat, niet 23 uur te vroeg. Zonder correctie zitten er sprongen van 1440
  minuten in je data.
""")

    st.markdown("**Ontbrekende waarden per kolom**")
    leeg = (data.isna().mean() * 100).round(1)
    st.dataframe(leeg[leeg > 0].sort_values(ascending=False).rename("Leeg (%)"),
                 use_container_width=True)

# ======================================================================
# CONCLUSIE: het antwoord op de vraag bovenaan
# ======================================================================
st.divider()
st.subheader("Conclusie: plant Zurich zijn eigen vertraging?")
st.markdown("""
**Grotendeels wel.** De drukte op de luchthaven bepaalt het meest hoeveel
vluchten te laat zijn. Het weer maakt het erger, en een deel is met deze data
niet te verklaren.

- **Drukte is de grootste oorzaak.** Hoe voller de maand, hoe meer te late
  vluchten (r = 0,91). Corona was een natuurlijk experiment: de drukte viel
  weg, het weer bleef, en de vertraging viel mee weg.
- **Vertraging stapelt zich op over de dag.** In 2019 liep het aandeel te laat
  op van 15% om 6 uur naar 41% rond 13 uur. In het rustige 2020 gebeurde dat
  bijna niet.
- **Van buiten: het weer.** Bij Bise-wind landt Zurich op banen met minder
  capaciteit. Op Bise-dagen is 24% van de vluchten te laat, op gewone dagen 18%.
- **Voorspelbaar, maar niet helemaal.** Drukte en weer voorspellen de
  vertraging van een dag beter dan gokken (8 tegen 10 procentpunt fout). De
  grootste missers zijn dagen met iets wat niet in de data staat, zoals een
  storing of staking.
""")
st.caption("Cijfers over 2019 en 2020 samen, zonder filters.")