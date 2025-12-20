# ==================== IMPORTACIÓN DE LIBRERÍAS ====================
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splev, splrep

# ==================== CONSTANTES GLOBALES ====================
EPSILON = 0.62198   # relación masas agua/aire seco
TOL = 1e-6          # tolerancia numérica

# ==================== CONFIGURACIÓN DE LA PÁGINA ====================
st.set_page_config(
    page_title="Método de Mickley - Torres de Enfriamiento",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title('🌡️ Simulación de Torres de Enfriamiento - Método de Mickley ❄️')
st.write(
    'Esta aplicación calcula la evolución del aire en una torre de enfriamiento y '
    'determina sus parámetros de diseño mediante el método de Mickley.'
)

# ==================== FUNCIONES AUXILIARES ====================

def get_units_config(units_option: str):
    """Devuelve diccionario con datos y propiedades para SI o Inglés."""
    if units_option == 'Sistema Inglés':
        return dict(
            teq=np.array([32, 40, 60, 80, 100, 120, 140]),
            Heq_data=np.array([4.074, 7.545, 18.780, 36.020, 64.090, 112.0, 198.0]),
            Cp_water=1.0,
            temp_unit="°F",
            enthalpy_unit="BTU/lb aire seco",
            flow_unit="lb/(h ft²)",
            length_unit="ft",
            h_temp_ref=32.0,
            h_latent_ref=1075.8,
            h_cp_air_dry=0.24,
            h_cp_vapor=0.45,
            kya_unit="lb/(h ft³ ΔY)",
            cp_unit="BTU/(lb agua °F)",
            Y_unit="lb agua/lb aire seco",
            psychrometric_constant=0.000367,  # psi^-1
            pressure_factor=14.696,          # atm → psi
            psat_units='psi'
        )
    # Sistema Internacional
    return dict(
        teq=np.array([0, 10, 20, 30, 40, 50, 60]),
        Heq_data=np.array([9479, 29360, 57570, 100030, 166790, 275580, 461500]),
        Cp_water=4186.0,
        temp_unit="°C",
        enthalpy_unit="J/kg aire seco",
        flow_unit="kg/(s m²)",
        length_unit="m",
        h_temp_ref=0.0,
        h_latent_ref=2501e3,
        h_cp_air_dry=1005.0,
        h_cp_vapor=1880.0,
        kya_unit="kg/(s m³ ΔY)",
        cp_unit="J/(kg agua °C)",
        Y_unit="kg agua/kg aire seco",
        psychrometric_constant=0.000662,    # kPa^-1
        pressure_factor=101.325,           # atm → kPa
        psat_units='kPa'
    )


def enthalpy_moist_air(T, Y, props):
    """Entalpía del aire húmedo (H) en función de T y Y."""
    dT = T - props["h_temp_ref"]
    return (props["h_cp_air_dry"] + props["h_cp_vapor"] * Y) * dT + props["h_latent_ref"] * Y


def humidity_ratio_from_H_t(H, T, props):
    """Humedad absoluta Y a partir de entalpía H y temperatura T."""
    dT = T - props["h_temp_ref"]
    denom = (props["h_cp_vapor"] * dT + props["h_latent_ref"])
    if abs(denom) < TOL:
        raise ValueError("Denominador ~0 al calcular Y(H,T). Verificar rango de temperaturas.")
    return (H - props["h_cp_air_dry"] * dT) / denom


def sat_vapor_pressure_magnus(T, units_psat):
    """
    Presión de vapor de saturación usando Magnus.
    units_psat = 'kPa' → T en °C, P_ws en kPa.
    units_psat = 'psi' → T en °F, se convierte internamente.
    """
    if units_psat == 'kPa':
        return 0.61094 * np.exp((17.625 * T) / (T + 243.04))
    T_c = (T - 32.0) * 5.0 / 9.0
    P_ws_kPa = 0.61094 * np.exp((17.625 * T_c) / (T_c + 243.04))
    return P_ws_kPa / 6.89476  # kPa → psi


def humidity_ratio_from_wet_bulb(t_db, t_wb, P_atm, props):
    """Calcula Y a partir de bulbo seco, bulbo húmedo y presión total."""
    P_total = P_atm * props["pressure_factor"]
    P_ws = sat_vapor_pressure_magnus(t_wb, props["psat_units"])
    Pv = P_ws - props["psychrometric_constant"] * P_total * (t_db - t_wb)
    Pv = max(0.0, min(Pv, 0.99 * P_total))
    if P_total <= Pv:
        return None
    return EPSILON * Pv / (P_total - Pv)


def humidity_ratio_from_RH(t_db, RH_percent, P_atm, props):
    """Calcula Y a partir de bulbo seco, HR (%) y presión total."""
    P_total = P_atm * props["pressure_factor"]
    P_ws = sat_vapor_pressure_magnus(t_db, props["psat_units"])
    Pv = (RH_percent / 100.0) * P_ws
    Pv = max(0.0, min(Pv, 0.99 * P_total))
    if P_total <= Pv:
        return None
    return EPSILON * Pv / (P_total - Pv)


def build_equilibrium_functions(teq, Heq_data):
    """
    Construye:
    - H_star_spline: curva de equilibrio suave (cúbica) para Mickley e integración
    - H_star_lin: versión lineal para el cálculo robusto de Gs_min
    """
    H_star_spline = interp1d(teq, Heq_data, kind='cubic', fill_value
