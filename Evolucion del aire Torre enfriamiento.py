# ==================== IMPORTACIÓN DE LIBRERÍAS ====================
import streamlit as st  # Importa la librería Streamlit
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splev, splrep  # Import splev and splrep
from scipy.optimize import fsolve  # Para resolver numéricamente el punto de pellizco

# ==================== CONFIGURACIÓN DE LA PÁGINA (OPCIONAL) ====================
st.set_page_config(
    page_title="Método de Mickley - Torres de Enfriamiento",
    layout="centered",  # o "wide" para más espacio
    initial_sidebar_state="auto"
)

# ==================== TÍTULO DE LA APLICACIÓN ====================
st.title('🌡️ Simulación de Torres de Enfriamiento - Método de Mickley ❄️')
st.write('Esta aplicación calcula la evolución del aire en una torre de enfriamiento y determina sus parámetros de diseño.')

# ==================== DATOS DE EQUILIBRIO (MANTENER FIJOS O PERMITIR SELECCIÓN) ====================
st.subheader('Datos de la Curva de Equilibrio H*(t)')
opcion_unidades = st.radio(
    "Seleccione el sistema de unidades:",
    ('Sistema Inglés', 'Sistema Internacional')
)

if opcion_unidades == 'Sistema Inglés':
    teq = np.array([32, 40, 60, 80, 100, 120, 140])  # °F
    Heq_data = np.array([4.074, 7.545, 18.780, 36.020, 64.090, 112.0, 198.0])  # BTU/lb aire seco
    Cp_default = 1.0  # calor específico del agua, Btu/(lb °F)
    temp_unit = "°F"
    enthalpy_unit = "BTU/lb aire seco"
    flow_unit = "lb/(h ft²)"  # Especificación de unidades de flujo de agua y aire
    length_unit = "ft"
    h_temp_ref = 32
    h_latent_ref = 1075.8
    h_cp_air_dry = 0.24
    h_cp_vapor = 0.45
    kya_unit = "lb/(h ft² DY)"  # Especificación de unidades de KYa
    cp_unit = "BTU/(lb agua °F)"  # Especificación de unidades de Cp
    Y_unit = "lb agua/lb aire seco"  # Especificación de unidades de Y
    psychrometric_constant = 0.000367  # psi^-1 (para presión en psi)
else:  # Sistema Internacional
    teq = np.array([0, 10, 20, 30, 40, 50, 60])  # °C
    Heq_data = np.array([9479, 29360, 57570, 100030, 166790, 275580, 461500])  # J/kg aire seco
    Cp_default = 4186       # calor específico del agua, J/(kg °C)
    temp_unit = "°C"
    enthalpy_unit = "J/kg aire seco"  # Especificado "aire seco"
    flow_unit = "kg/(s m²)"  # Especificación de unidades de flujo de agua y aire
    length_unit = "m"
    h_temp_ref = 0  # Referencia para °C
    h_latent_ref = 2501e3  # A 0°C, J/kg
    h_cp_air_dry = 1005  # J/kg°C
    h_cp_vapor = 1880  # J/kg°C (puede variar un poco)
    kya_unit = "kg/(s m² DY)"  # Especificación de unidades de KYa
    cp_unit = "J/(kg agua °C)"  # Especificación de unidades de Cp
    Y_unit = "kg agua/kg aire seco"  # Especificación de unidades de Y
    psychrometric_constant = 0.000662  # kPa^-1 (para presión en kPa)

# ==================== FUNCIONES TERMODINÁMICAS ====================

def calcular_entalpia_aire(t, Y, temp_ref, latent_ref, cp_air_dry, cp_vapor):
    """Entalpía del aire húmedo."""
    return (cp_air_dry + cp_vapor * Y) * (t - temp_ref) + latent_ref * Y

def calcular_Y(H, t, temp_ref, latent_ref, cp_air_dry, cp_vapor):
    """Humedad absoluta Y a partir de H y t."""
    return (H - cp_air_dry * (t - temp_ref)) / (cp_vapor * (t - temp_ref) + latent_ref)

def get_saturation_vapor_pressure(temperature, units_system):
    """
    Calcula la presión de vapor de saturación del agua (Magnus).
    """
    if units_system == 'Sistema Internacional':  # Temperatura en °C, P_ws en kPa
        return 0.61094 * np.exp((17.625 * temperature) / (temperature + 243.04))
    else:  # Temperatura en °F, P_ws en psi
        temp_c = (temperature - 32) * 5/9
        P_ws_kPa = 0.61094 * np.exp((17.625 * temp_c) / (temp_c + 243.04))
        return P_ws_kPa / 6.89476  # kPa → psi

def calculate_Y_from_wet_bulb(t_dry_bulb, t_wet_bulb, total_pressure_atm, units_system, psych_const):
    """Calcula Y a partir de bulbo seco, bulbo húmedo y P total."""
    if units_system == 'Sistema Internacional':
        P_total = total_pressure_atm * 101.325  # kPa
    else:
        P_total = total_pressure_atm * 14.696  # psi

    P_ws_tw = get_saturation_vapor_pressure(t_wet_bulb, units_system)
    Pv = P_ws_tw - psych_const * P_total * (t_dry_bulb - t_wet_bulb)

    if Pv < 0:
        Pv = 0.0

    if (P_total - Pv) <= 0:
        return float('inf')
    Y = 0.62198 * (Pv / (P_total - Pv))
    return Y

def calculate_Y_from_relative_humidity(t_dry_bulb, relative_humidity_percent, total_pressure_atm, units_system):
    """Calcula Y a partir de bulbo seco, HR (%) y P total."""
    if units_system == 'Sistema Internacional':
        P_total = total_pressure_atm * 101.325  # kPa
    else:
        P_total = total_pressure_atm * 14.696  # psi

    P_ws_tdb = get_saturation_vapor_pressure(t_dry_bulb, units_system)
    Pv = (relative_humidity_percent / 100.0) * P_ws_tdb

    if (P_total - Pv) <= 0:
        return float('inf')
    Y = 0.62198 * (Pv / (P_total - Pv))
    return Y

# ==================== ENTRADA DE DATOS DEL PROBLEMA ====================
st.sidebar.header('Parámetros del Problema')

P = st.sidebar.number_input('Presión de operación (P, atm)', value=1.0, format="%.2f")
L = st.sidebar.number_input(f'Flujo de agua (L, {flow_unit})', value=2200.0, format="%.2f")
G = st.sidebar.number_input(f'Flujo de aire (G, {flow_unit})', value=2000.0, format="%.2f")
tfin = st.sidebar.number_input(f'Temperatura de entrada del agua (tfin, {temp_unit})', value=105.0, format="%.2f")
tini = st.sidebar.number_input(f'Temperatura de salida del agua (tini, {temp_unit})', value=85.0, format="%.2f")

Y1_source_option = st.sidebar.radio(
    "Fuente de Humedad Absoluta (Y1):",
    ('Ingresar Y1 directamente', 'Calcular Y1 a partir de Bulbo Húmedo', 'Calcular Y1 a partir de Humedad Relativa')
)

Y1 = 0.016  # Valor por defecto inicial

if Y1_source_option == 'Ingresar Y1 directamente':
    tG1 = st.sidebar.number_input(f
