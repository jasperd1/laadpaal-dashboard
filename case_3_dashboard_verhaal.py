import warnings
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_folium import st_folium
import requests

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
st.set_page_config(page_title="EV Dashboard Presentatie", page_icon="⚡", layout="wide")

WEEKDAY_NL = {
    0: "Maandag",
    1: "Dinsdag",
    2: "Woensdag",
    3: "Donderdag",
    4: "Vrijdag",
    5: "Zaterdag",
    6: "Zondag",
}
WEEKDAY_ORDER = ["Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag", "Zaterdag", "Zondag"]


# ---------- Helper functies ----------
def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.30)


def format_int(value):
    return f"{int(round(value)):,}".replace(",", ".")


def format_pct(value):
    return f"{value:.1f}%"


@st.cache_data
def load_data():
    laad_df = pd.read_csv("laadpaaldata_schoon_verrijkt.csv")
    cars_df = pd.read_csv("cars_schoon_verrijkt.csv")

    # ---- Laadpaaldata ----
    laad_df["Started"] = pd.to_datetime(laad_df["Started"], errors="coerce")
    laad_df["Ended"] = pd.to_datetime(laad_df["Ended"], errors="coerce")
    laad_df["TotalEnergy"] = pd.to_numeric(laad_df["TotalEnergy"], errors="coerce")
    laad_df["ConnectedTime"] = pd.to_numeric(laad_df["ConnectedTime"], errors="coerce")
    laad_df["ChargeTime"] = pd.to_numeric(laad_df["ChargeTime"], errors="coerce")
    laad_df["MaxPower"] = pd.to_numeric(laad_df["MaxPower"], errors="coerce")

    laad_df = laad_df.dropna(subset=["Started", "Ended", "TotalEnergy", "ConnectedTime", "ChargeTime", "MaxPower"]).copy()
    laad_df = laad_df[
        (laad_df["TotalEnergy"] >= 0)
        & (laad_df["ConnectedTime"] >= 0)
        & (laad_df["ChargeTime"] >= 0)
        & (laad_df["MaxPower"] >= 0)
    ].copy()

    laad_df["TotalEnergy_kWh"] = laad_df["TotalEnergy"] / 1000
    laad_df["MaxPower_kW"] = laad_df["MaxPower"] / 1000
    laad_df["hour"] = laad_df["Started"].dt.hour
    laad_df["weekday_num"] = laad_df["Started"].dt.weekday
    laad_df["weekday_nl"] = laad_df["weekday_num"].map(WEEKDAY_NL)
    laad_df["day_type_nl"] = np.where(laad_df["weekday_num"] < 5, "Werkdag", "Weekend")
    laad_df["idle_time"] = laad_df["ConnectedTime"] - laad_df["ChargeTime"]
    laad_df["idle_time"] = laad_df["idle_time"].clip(lower=0)
    laad_df["efficiency"] = np.where(laad_df["ConnectedTime"] > 0, laad_df["ChargeTime"] / laad_df["ConnectedTime"], np.nan)
    laad_df["avg_power_kW"] = np.where(laad_df["ChargeTime"] > 0, laad_df["TotalEnergy_kWh"] / laad_df["ChargeTime"], np.nan)
    laad_df["jaar_maand"] = laad_df["Started"].dt.to_period("M").astype(str)
    laad_df["jaar_maand_dt"] = pd.to_datetime(laad_df["jaar_maand"] + "-01", errors="coerce")

    # ---- Voertuigdata ----
    if "jaar_maand" not in cars_df.columns and {"jaar", "maand"}.issubset(cars_df.columns):
        cars_df["jaar_maand"] = (
            cars_df["jaar"].astype("Int64").astype(str)
            + "-"
            + cars_df["maand"].astype("Int64").astype(str).str.zfill(2)
        )

    cars_df["jaar_maand_dt"] = pd.to_datetime(cars_df["jaar_maand"] + "-01", errors="coerce")
    cars_df["catalogusprijs"] = pd.to_numeric(cars_df["catalogusprijs"], errors="coerce")
    cars_df["merk_clean"] = cars_df["merk"].astype(str).str.upper().str.strip()
    cars_df["model_clean"] = cars_df["handelsbenaming"].astype(str).str.upper().str.strip()

    return laad_df, cars_df


@st.cache_data
def load_local_map_data():
    for file_name in ["laadpunten_nederland.csv", "ocm_nederland.csv"]:
        file_path = Path(file_name)
        if file_path.exists():
            df = pd.read_csv(file_path)
            if {"latitude", "longitude"}.issubset(df.columns):
                if "state" not in df.columns:
                    df["state"] = "Onbekend"
                if "town" not in df.columns:
                    df["town"] = "Onbekend"
                if "title" not in df.columns:
                    df["title"] = "Laadpunt"
                return df
    return pd.DataFrame()


# ---------- Data laden ----------
laad_df, cars_df = load_data()
map_df = load_local_map_data()

# ---------- Standaardinstellingen zonder sidebar ----------
start_date = laad_df["Started"].dt.date.min()
end_date = laad_df["Started"].dt.date.max()
selected_day_type = "Alles"
problem_idle_threshold = 1.0
story_mode = "Management"

# Hele dataset gebruiken
laad_filtered = laad_df[
    (laad_df["Started"].dt.date >= start_date) &
    (laad_df["Started"].dt.date <= end_date)
].copy()

if selected_day_type != "Alles":
    laad_filtered = laad_filtered[
        laad_filtered["day_type_nl"] == selected_day_type
    ].copy()

cars_filtered = cars_df.copy()

if laad_filtered.empty:
    st.warning("Geen laadsessies gevonden.")
    st.stop()

# ---------- Kernberekeningen ----------
peak_hour_counts = laad_filtered.groupby("hour").size().reindex(range(24), fill_value=0)
peak_hour = int(peak_hour_counts.idxmax())

heat_counts = laad_filtered.groupby(["weekday_nl", "hour"]).size().reset_index(name="count")
peak_heat = heat_counts.loc[heat_counts["count"].idxmax()]

mean_charge = laad_filtered["ChargeTime"].mean()
mean_connected = laad_filtered["ConnectedTime"].mean()
mean_idle = laad_filtered["idle_time"].mean()
mean_efficiency = laad_filtered["efficiency"].mean() * 100
problem_idle_share = (laad_filtered["idle_time"] > problem_idle_threshold).mean() * 100
any_idle_share = (laad_filtered["idle_time"] > 0).mean() * 100
idle_gap = mean_connected - mean_charge

# ---------- Pagina ----------
st.title("⚡ Dashboard Elektrisch Laden en EV-markt")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Overzicht",
    "2. Gebruikspatroon",
    "3. Laden vs bezetten",
    "4. EV-markt",
    "5. Kaart",
    "6. Scenario & voorspelling",
])

with tab1:
    st.subheader("KPI's")
    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)

    c1.metric("Laadsessies", format_int(len(laad_filtered)))
    c2.metric("Totale energie", f"{laad_filtered['TotalEnergy_kWh'].sum():,.0f} kWh".replace(",", "."))
    c3.metric("Gem. laadtijd", f"{mean_charge:.2f} uur")
    c4.metric("Gem. verbonden tijd", f"{mean_connected:.2f} uur")
    c5.metric("Gem. idle time", f"{mean_idle:.2f} uur")
    c6.metric("Efficiency", format_pct(mean_efficiency))
    c7.metric("Drukste uur", f"{peak_hour}:00")
    c8.metric(f"Idle time > {problem_idle_threshold:.1f} uur", format_pct(problem_idle_share))

with tab2:
    st.subheader("Laadpaalgebruik")
    chart_choice = st.radio(
    "",
    ["Aantal sessies", "Totaal geladen energie"],
    horizontal=True,
    key="usage_metric",
)

    col1, col2 = st.columns(2)

    with col1:
        if chart_choice == "Aantal sessies":
            series = laad_filtered.groupby("hour").size().reindex(range(24), fill_value=0)
            ylabel = "Aantal sessies"
            title = "Laadsessies per uur"
        else:
            series = laad_filtered.groupby("hour")["TotalEnergy_kWh"].sum().reindex(range(24), fill_value=0)
            ylabel = "Totaal geladen kWh"
            title = "Geladen energie per uur"

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(series.index, series.values, marker="o", linewidth=2)
        ax.axvline(peak_hour, linestyle="--", linewidth=2, alpha=0.8)
        ax.set_xticks(range(24))
        style_ax(ax, title, "Uur van de dag", ylabel)
        st.pyplot(fig)

    with col2:
        if chart_choice == "Aantal sessies":
            heatmap_data = laad_filtered.pivot_table(
                index="weekday_nl",
                columns="hour",
                values="Started",
                aggfunc="count",
            ).fillna(0)
            title = "Heatmap: aantal sessies"
        else:
            heatmap_data = laad_filtered.pivot_table(
                index="weekday_nl",
                columns="hour",
                values="TotalEnergy_kWh",
                aggfunc="sum",
            ).fillna(0)
            title = "Heatmap: geladen energie"

        heatmap_data = heatmap_data.reindex(WEEKDAY_ORDER)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Uur van de dag")
        ax.set_ylabel("Weekdag")
        st.pyplot(fig)


with tab3:
    st.subheader("Laden vs bezetten")

    schaal = st.selectbox("X-as schaal scatterplot", ["Normaal", "Logaritmisch"], key="scale_tab3")

    plot_df = laad_filtered[
        (laad_filtered["ConnectedTime"] > 0)
        & (laad_filtered["ChargeTime"] > 0)
        & (laad_filtered["idle_time"] >= 0)
    ].copy()

    plot_df["problematisch"] = np.where(plot_df["idle_time"] > problem_idle_threshold, "Ja", "Nee")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for label, alpha in [("Nee", 0.18), ("Ja", 0.45)]:
            deel = plot_df[plot_df["problematisch"] == label]
            ax.scatter(deel["ConnectedTime"], deel["ChargeTime"], alpha=alpha, label=f"Problematisch: {label}")

        max_axis = max(plot_df["ConnectedTime"].max(), plot_df["ChargeTime"].max())
        ax.plot([0.01, max_axis], [0.01, max_axis], linestyle="--", linewidth=2, label="Perfect: laden = bezetten")

        if schaal == "Logaritmisch":
            ax.set_xscale("log")

        style_ax(ax, "ConnectedTime versus ChargeTime", "ConnectedTime (uur)", "ChargeTime (uur)")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(plot_df["idle_time"].dropna(), bins=35, edgecolor="black", alpha=0.80)
        ax.axvline(problem_idle_threshold, linestyle="--", linewidth=2, label=f"Drempel: {problem_idle_threshold:.1f} uur")
        ax.axvline(plot_df["idle_time"].mean(), linestyle=":", linewidth=2, label="Gemiddelde")
        style_ax(ax, "Verdeling idle time", "Idle time (uur)", "Aantal sessies")
        ax.legend()
        st.pyplot(fig)

    k1, k2, k3 = st.columns(3)
    k1.metric("Gem. verschil", f"{idle_gap:.2f} uur")
    k2.metric("Sessies met enige idle time", format_pct(any_idle_share))
    k3.metric(f"Sessies > {problem_idle_threshold:.1f} uur idle", format_pct(problem_idle_share))

with tab4:
    st.subheader("Waarom dit belangrijker wordt: de EV-markt groeit")
    top_n = st.slider("Top N merken", 5, 15, 10, key="top_n_brands")

    voertuigen_per_maand = cars_filtered.groupby("jaar_maand_dt").size().sort_index()
    cumulatief = voertuigen_per_maand.cumsum()
    top_merken = cars_filtered["merk_clean"].value_counts().head(top_n).sort_values()
    prijs_per_maand = cars_filtered.groupby("jaar_maand_dt")["catalogusprijs"].mean().dropna()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(cumulatief.index, cumulatief.values, marker="o", linewidth=2)
        style_ax(ax, "Cumulatieve groei EV-registraties", "Maand", "Cumulatief aantal voertuigen")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.barh(top_merken.index, top_merken.values)
        style_ax(ax, f"Top {top_n} merken", "Aantal voertuigen", "Merk")
        st.pyplot(fig)

with tab5:
    st.subheader("Kaart van laadpunten in Nederland")
 
    @st.cache_data(ttl=3600)
    def load_ocm_data():
        url = "https://api.openchargemap.io/v3/poi/"
        headers = {
            "X-API-Key": "f3765ace-d3ab-4f82-92b5-156b758a3030",
            "User-Agent": "school-dashboard"
        }
        params = {
            "output": "json",
            "countrycode": "NL",
            "maxresults": 300,   # lager maken helpt vaak
            "compact": True,
            "verbose": False
        }
 
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
 
        except requests.exceptions.ReadTimeout:
            st.warning("De kaartdata van Open Charge Map laadt te langzaam. Probeer het zo opnieuw.")
            return pd.DataFrame()
 
        except requests.exceptions.ConnectionError:
            st.warning("Geen verbinding met Open Charge Map.")
            return pd.DataFrame()
 
        except requests.exceptions.HTTPError as e:
            st.warning(f"HTTP-fout bij ophalen kaartdata: {e}")
            return pd.DataFrame()
 
        except ValueError:
            st.warning("De API gaf geen geldige JSON terug.")
            return pd.DataFrame()
 
        except Exception as e:
            st.warning(f"Onverwachte fout bij ophalen kaartdata: {e}")
            return pd.DataFrame()
 
        rows = []
        for item in data:
            address = item.get("AddressInfo", {})
            rows.append({
                "title": address.get("Title"),
                "latitude": address.get("Latitude"),
                "longitude": address.get("Longitude"),
                "town": address.get("Town"),
                "state": address.get("StateOrProvince")
            })
 
        df = pd.DataFrame(rows)
 
        if df.empty:
            return df
 
        df = df.dropna(subset=["latitude", "longitude"])
        df = df.drop_duplicates()
        df["town"] = df["town"].fillna("Onbekend")
        df["state"] = df["state"].fillna("Onbekend").astype(str).str.strip()
 
        df["state"] = df["state"].replace({
            "NB": "Noord-Brabant",
            "NH": "Noord-Holland",
            "ZH": "Zuid-Holland",
            "UT": "Utrecht",
            "UTRECHT": "Utrecht",
            "Nederland": "Onbekend",
            "North Brabant": "Noord-Brabant",
            "North-Holland": "Noord-Holland",
            "North Holland": "Noord-Holland",
            "Noord Holland": "Noord-Holland",
            "South Holland": "Zuid-Holland",
            "Zuid Holland": "Zuid-Holland",
            "Seeland": "Zeeland",
            "Nordholland": "Noord-Holland",
            "Fryslân": "Friesland"
        })
 
        geldige_provincies = [
            "Drenthe", "Flevoland", "Friesland", "Gelderland",
            "Groningen", "Limburg", "Noord-Brabant", "Noord-Holland",
            "Overijssel", "Utrecht", "Zeeland", "Zuid-Holland"
        ]
 
        df = df[df["state"].isin(geldige_provincies)].copy()
        return df
 
    df = load_ocm_data()
 
    if df.empty:
        st.info("Geen kaartdata beschikbaar op dit moment.")
    else:
        provincies = ["Alle provincies"] + sorted(df["state"].unique().tolist())
        gekozen_provincie = st.selectbox("Kies een provincie", provincies)
 
        if gekozen_provincie == "Alle provincies":
            filtered_df = df.copy()
            center_lat, center_lon, zoom = 52.2, 5.3, 7
        else:
            filtered_df = df[df["state"] == gekozen_provincie].copy()
            if not filtered_df.empty:
                center_lat = filtered_df["latitude"].mean()
                center_lon = filtered_df["longitude"].mean()
                zoom = 9
            else:
                center_lat, center_lon, zoom = 52.2, 5.3, 7
 
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
 
        for _, row in filtered_df.iterrows():
            popup_text = f"""
            <b>Naam:</b> {row['title']}<br>
            <b>Plaats:</b> {row['town']}<br>
            <b>Provincie:</b> {row['state']}
            """
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4,
                popup=folium.Popup(popup_text, max_width=250),
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
 
        st_folium(m, width=1000, height=600)
        st.write(f"Aantal getoonde laadpunten: {len(filtered_df)}")

with tab6:
    st.subheader("Lineaire cumulatieve voorspelling elektrische auto's tot 2030")
 
    # Historische instroom per maand
    voorspelling_df = cars_df.groupby("jaar_maand_dt").size().reset_index(name="aantal_voertuigen")
    voorspelling_df = voorspelling_df.sort_values("jaar_maand_dt").copy()
 
    # Alleen vanaf januari 2022
    start_datum = pd.Timestamp("2022-01-01")
    voorspelling_df = voorspelling_df[voorspelling_df["jaar_maand_dt"] >= start_datum].copy()
 
    # Eventueel laatste onvolledige maand verwijderen
    if len(voorspelling_df) >= 3:
        median_laatste_6 = voorspelling_df["aantal_voertuigen"].iloc[:-1].tail(6).median()
        laatste_waarde = voorspelling_df["aantal_voertuigen"].iloc[-1]
 
        if laatste_waarde < 0.5 * median_laatste_6:
            voorspelling_df = voorspelling_df.iloc[:-1].copy()
 
    voorspelling_df = voorspelling_df.reset_index(drop=True)
 
    if voorspelling_df.empty:
        st.error("Er is geen voertuigdata beschikbaar vanaf januari 2022.")
    else:
        # Historische cumulatieve lijn laten starten op 200.000 in jan 2022
        start_waarde = 200000
        voorspelling_df["cumulatief"] = start_waarde + voorspelling_df["aantal_voertuigen"].cumsum().shift(fill_value=0)
 
        # Tijdindex op cumulatieve lijn
        voorspelling_df["t"] = np.arange(len(voorspelling_df))
 
        x = voorspelling_df["t"].values
        y = voorspelling_df["cumulatief"].values
 
        # Lineaire regressie DIRECT op cumulatieve waarden
        coeffs = np.polyfit(x, y, 1)
        lineaire_functie = np.poly1d(coeffs)
 
        # Toekomst tot eind 2030
        laatste_datum = voorspelling_df["jaar_maand_dt"].max()
        einddatum = pd.Timestamp("2030-12-01")
 
        future_dates = pd.date_range(
            start=laatste_datum + pd.DateOffset(months=1),
            end=einddatum,
            freq="MS"
        )
 
        future_t = np.arange(len(voorspelling_df), len(voorspelling_df) + len(future_dates))
        future_cumulatief = lineaire_functie(future_t)
 
        # Zorg dat voorspelling niet onder laatste historische waarde komt
        future_cumulatief = np.maximum(future_cumulatief, voorspelling_df["cumulatief"].iloc[-1])
 
        future_df = pd.DataFrame({
            "jaar_maand_dt": future_dates,
            "cumulatief": future_cumulatief
        })
 
        # KPI's
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Startwaarde januari 2022", f"{start_waarde:,}".replace(",", "."))

    with col2:
        st.metric("Laatste historische waarde", f"{int(voorspelling_df['cumulatief'].iloc[-1]):,}".replace(",", "."))

    with col3:
        st.metric("Voorspeld aantal in 2030", f"{int(future_df['cumulatief'].iloc[-1]):,}".replace(",", "."))
 
        # Grafiek
        fig, ax = plt.subplots(figsize=(11, 5))
 
        ax.plot(
            voorspelling_df["jaar_maand_dt"],
            voorspelling_df["cumulatief"],
            linewidth=2,
            marker="o",
            label="Historisch cumulatief"
        )
 
        ax.plot(
            future_df["jaar_maand_dt"],
            future_df["cumulatief"],
            linewidth=2,
            linestyle="--",
            label="Lineair doorgetrokken tot 2030"
        )
 
        ax.scatter(
            pd.Timestamp("2022-01-01"),
            start_waarde,
            s=80,
            zorder=5,
            label="Startpunt: 200.000"
        )
 
        ax.axvline(
            voorspelling_df["jaar_maand_dt"].max(),
            linestyle=":",
            linewidth=2,
            label="Start voorspelling"
        )
 
        style_ax(
            ax,
            "Cumulatieve groei elektrische auto's (lineair doorgetrokken tot 2030)",
            "Jaar",
            "Cumulatief aantal voertuigen"
        )
        ax.legend()
        plt.xticks(rotation=45)
 
        st.pyplot(fig)
 
        st.info(
            "De voorspelling is hier direct lineair doorgetrokken op de cumulatieve aantallen. "
            "Daardoor is de voorspellingslijn zelf een rechte lijn tot 2030."
        )