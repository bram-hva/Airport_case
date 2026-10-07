"""
Dashboard: vluchten en vertraging op Zurich Airport, 2019-2020.

Leest 'vluchten_compleet.csv', het bestand dat Airport.py maakt.
Draaien met:   python -m streamlit run dashboard.py

De kaart is gemaakt met Folium, zoals in de demo van hoorcollege 6:
CircleMarker, een straal die met de wortel schaalt, en FeatureGroup plus
LayerControl om klassen aan en uit te zetten.
"""

from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Vluchten en vertraging op Zurich", layout="wide")

MAP = Path(__file__).parent          # map waar dit bestand staat

BLAUW = "#2a78d6"
ORANJE = "#eb6834"
LICHTGRIJS = "#c8d0d6"
# Vijf tinten licht naar donker. Een reeks getallen hoort geen regenboog.
TRAPPEN = ["#cfe0f5", "#9dc1eb", "#6ba2e0", "#3a83d6", "#1a5ba8"]

# Bleke ondergrond zonder API-sleutel. Esri wil {z}/{y}/{x}, niet {z}/{x}/{y}.
ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")


@st.cache_data
def laad():
    df = pd.read_csv(MAP / "vluchten_compleet.csv", parse_dates=["Datum"])
    # Drukte: hoeveel bewegingen in hetzelfde geplande uur? Deze kolom staat
    # in geen van de drie bronbestanden; die maken we zelf.
    blok = df["Datum"].dt.strftime("%Y-%m-%d") + " " + df["Uur"].astype(str)
    df["Drukte"] = blok.map(blok.value_counts())
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
