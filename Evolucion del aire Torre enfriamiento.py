# ==================== IMPORTACIÓN DE LIBRERÍAS ====================
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splev, splrep 
from scipy.optimize import fsolve 

# ==================== CONFIGURACIÓN DE LA PÁGINA ====================
st.set_page_config(
    page_title="Torres de Enfriamiento OU3 FICA-UNSL",
    layout="centered",
    initial_sidebar_state="auto"
)

# ==================== TÍTULO DE LA APLICACIÓN ====================
st.title('🌡️ Simulación de Torres de Enfriamiento OU3 FICA-UNSL ❄️')
st.write('Esta aplicación calcula la evolución del aire en una torre de enfriamiento y estima sus parámetros de diseño.')

# ==================== DATOS DE EQUILIBRIO ====================
opcion_unidades = st.radio(
    "Seleccione el sistema de unidades:",
    ('Sistema Inglés', 'Sistema Internacional')
)

if opcion_unidades == 'Sistema Inglés':
    teq = np.array([32, 40, 60, 80, 100, 120, 140]) # °F
    Heq_data = np.array([4.074, 7.545, 18.780, 36.020, 64.090, 112.0, 198.0]) 
    Cp_default = 1.0 
    temp_unit = "°F"
    enthalpy_unit = "BTU/lb aire seco"
    flow_unit = "lb/(h ft²)" 
    length_unit = "ft"
    h_temp_ref = 32
    h_latent_ref = 1075.8
    h_cp_air_dry = 0.24
    h_cp_vapor = 0.45
    kya_unit = "lb/(h ft² DY)" 
    cp_unit = "BTU/(lb agua °F)" 
    Y_unit = "lb agua/lb aire seco" 
    psychrometric_constant = 0.000367 
    Gs_unit = "lb aire seco/(h ft²)"
else: # Sistema Internacional
    teq = np.array([0, 10, 20, 30, 40, 50, 60]) # °C
    Heq_data = np.array([9479, 29360, 57570, 100030, 166790, 275580, 461500]) 
    Cp_default = 4186 
    temp_unit = "°C"
    enthalpy_unit = "J/kg aire seco"
    flow_unit = "kg/(s m²)" 
    length_unit = "m"
    h_temp_ref = 0 
    h_latent_ref = 2501e3 
    h_cp_air_dry = 1005 
    h_cp_vapor = 1880 
    kya_unit = "kg/(s m² DY)" 
    cp_unit = "J/(kg agua °C)" 
    Y_unit = "kg agua/kg aire seco" 
    psychrometric_constant = 0.000662 
    Gs_unit = "kg aire seco/(s m²)"

# ==================== FUNCIONES TERMODINÁMICAS ====================

def calcular_entalpia_aire(t, Y, temp_ref, latent_ref, cp_air_dry, cp_vapor):
    return (cp_air_dry + cp_vapor * Y) * (t - temp_ref) + latent_ref * Y

def calcular_Y(H, t, temp_ref, latent_ref, cp_air_dry, cp_vapor):
    return (H - cp_air_dry * (t - temp_ref)) / (cp_vapor * (t - temp_ref) + latent_ref)

def get_saturation_vapor_pressure(temperature, units_system):
    if units_system == 'Sistema Internacional': 
        return 0.61094 * np.exp((17.625 * temperature) / (temperature + 243.04))
    else: 
        temp_c = (temperature - 32) * 5/9
        P_ws_kPa = 0.61094 * np.exp((17.625 * temp_c) / (temp_c + 243.04))
        return P_ws_kPa / 6.89476 

def calculate_Y_from_wet_bulb(t_dry_bulb, t_wet_bulb, total_pressure_atm, units_system, psych_const):
    if units_system == 'Sistema Internacional':
        P_total = total_pressure_atm * 101.325 
    else:
        P_total = total_pressure_atm * 14.696 
    P_ws_tw = get_saturation_vapor_pressure(t_wet_bulb, units_system)
    Pv = P_ws_tw - psych_const * P_total * (t_dry_bulb - t_wet_bulb)
    if Pv < 0: Pv = 0.0
    if (P_total - Pv) <= 0: return float('inf')
    return 0.62198 * (Pv / (P_total - Pv))

def calculate_Y_from_relative_humidity(t_dry_bulb, relative_humidity_percent, total_pressure_atm, units_system):
    if units_system == 'Sistema Internacional':
        P_total = total_pressure_atm * 101.325 
    else:
        P_total = total_pressure_atm * 14.696 
    P_ws_tdb = get_saturation_vapor_pressure(t_dry_bulb, units_system)
    Pv = (relative_humidity_percent / 100.0) * P_ws_tdb
    if (P_total - Pv) <= 0: return float('inf')
    return 0.62198 * (Pv / (P_total - Pv))

# ==================== ENTRADA DE DATOS ====================
st.sidebar.header('Datos del Problema')

P = st.sidebar.number_input('Presión de operación (P, atm)', value=1.0, format="%.2f")
L = st.sidebar.number_input(f'Flujo de agua (L, {flow_unit})', value=2200.0, format="%.2f")
G = st.sidebar.number_input(f'Flujo de aire (G, {flow_unit})', value=2000.0, format="%.2f")
tfin = st.sidebar.number_input(f'Temp. entrada agua (tfin, {temp_unit})', value=105.0, format="%.2f")
tini = st.sidebar.number_input(f'Temp. salida agua (tini, {temp_unit})', value=85.0, format="%.2f")

Y1_source_option = st.sidebar.radio(
    "Fuente de Y1:",
    ('Ingresar Y1 directamente', 'Calcular Y1 a partir de Bulbo Húmedo', 'Calcular Y1 a partir de Humedad Relativa')
)

Y1 = 0.016 
tG1 = 90.0 # Valor por defecto

if Y1_source_option == 'Ingresar Y1 directamente':
    tG1 = st.sidebar.number_input(f'Bulbo seco aire (tG1, {temp_unit})', value=90.0)
    Y1 = st.sidebar.number_input(f'Y1 ({Y_unit})', value=0.016, format="%.5f")
elif Y1_source_option == 'Calcular Y1 a partir de Bulbo Húmedo':
    tG1 = st.sidebar.number_input(f'Bulbo seco aire (tG1, {temp_unit})', value=90.0)
    tw1 = st.sidebar.number_input(f'Bulbo húmedo aire (tw1, {temp_unit})', value=76.0)
    Y1 = calculate_Y_from_wet_bulb(tG1, tw1, P, opcion_unidades, psychrometric_constant)
elif Y1_source_option == 'Calcular Y1 a partir de Humedad Relativa':
    tG1 = st.sidebar.number_input(f'Bulbo seco aire (tG1, {temp_unit})', value=90.0)
    hr = st.sidebar.number_input('HR (%)', value=50.0)
    Y1 = calculate_Y_from_relative_humidity(tG1, hr, P, opcion_unidades)

KYa = st.sidebar.number_input(f'KYa ({kya_unit})', value=850.0)

# ==================== CÁLCULOS Y GRÁFICOS ====================
try:
    y1_frac = Y1 / (1 + Y1)
    Gs = G * (1 - y1_frac)
    Hini = calcular_entalpia_aire(tG1, Y1, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)
    Hfin = (L * Cp_default / Gs) * (tfin - tini) + Hini
    
    # Curva equilibrio
    H_star_func = interp1d(teq, Heq_data, kind='cubic', fill_value='extrapolate')
    
    # Simulación Mickley simplificada para el gráfico
    t_water_array = np.linspace(tini, tfin, 20)
    H_op_array = (L * Cp_default / Gs) * (t_water_array - tini) + Hini
    
    # Cálculo NtoG por integración trapezoidal
    H_range = np.linspace(Hini, Hfin, 100)
    # T_water correspondiente a cada H en la línea de operación:
    T_water_range = (H_range - Hini) * (tfin - tini) / (Hfin - Hini) + tini
    H_star_range = H_star_func(T_water_range)
    integrando = 1 / (H_star_range - H_range)
    NtoG = np.trapz(integrando, H_range)
    
    HtoG = Gs / KYa
    Z = HtoG * NtoG

    # Resultados en pantalla
    st.subheader("📊 Resultados del Diseño")
    col1, col2 = st.columns(2)
    col1.metric("Altura del Relleno (Z)", f"{Z:.2f} {length_unit}")
    col2.metric("NtoG", f"{NtoG:.2f}")
    
    # Gráfico
    fig, ax = plt.subplots()
    t_plot = np.linspace(min(teq), max(teq), 100)
    ax.plot(t_plot, H_star_func(t_plot), label='Equilibrio H*')
    ax.plot(t_water_array, H_op_array, 'r-o', label='Línea de Operación')
    ax.set_xlabel(f'Temperatura ({temp_unit})')
    ax.set_ylabel(f'Entalpía ({enthalpy_unit})')
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error en los cálculos: {e}")

st.markdown("---")
st.caption("Final del reporte de simulación - 2025")
