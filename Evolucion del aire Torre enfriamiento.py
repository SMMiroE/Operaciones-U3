# ==================== IMPORTACIÓN DE LIBRERÍAS ====================
import streamlit as st  
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splev, splrep  
from scipy.optimize import fsolve  

# ==================== CONFIGURACIÓN DE LA PÁGINA ====================
st.set_page_config(
    page_title="Método de Mickley - Torres de Enfriamiento",
    layout="centered",
    initial_sidebar_state="auto"
)

# ==================== TÍTULO DE LA APLICACIÓN ====================
st.title('🌡️ Simulación de Torres de Enfriamiento - Método de Mickley ❄️')
st.write('Esta aplicación calcula la evolución del aire en una torre de enfriamiento y determina sus parámetros de diseño.')

# ==================== DATOS DE EQUILIBRIO ====================
st.subheader('Datos de la Curva de Equilibrio H*(t)')
opcion_unidades = st.radio(
    "Seleccione el sistema de unidades:",
    ('Sistema Inglés', 'Sistema Internacional')
)

if opcion_unidades == 'Sistema Inglés':
    teq = np.array([32, 40, 60, 80, 100, 120, 140])  # °F
    Heq_data = np.array([4.074, 7.545, 18.780, 36.020, 64.090, 112.0, 198.0])  # BTU/lb aire seco
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
else:  # Sistema Internacional
    teq = np.array([0, 10, 20, 30, 40, 50, 60])  # °C
    Heq_data = np.array([9479, 29360, 57570, 100030, 166790, 275580, 461500])  # J/kg aire seco
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
    Y = 0.62198 * (Pv / (P_total - Pv)) if (P_total - Pv) > 0 else float('inf')
    return Y

def calculate_Y_from_relative_humidity(t_dry_bulb, relative_humidity_percent, total_pressure_atm, units_system):
    if units_system == 'Sistema Internacional':
        P_total = total_pressure_atm * 101.325
    else:
        P_total = total_pressure_atm * 14.696
    P_ws_tdb = get_saturation_vapor_pressure(t_dry_bulb, units_system)
    Pv = (relative_humidity_percent / 100.0) * P_ws_tdb
    Y = 0.62198 * (Pv / (P_total - Pv)) if (P_total - Pv) > 0 else float('inf')
    return Y

# ==================== ENTRADA DE DATOS DEL PROBLEMA ====================
st.sidebar.header('Parámetros del Problema')
P = st.sidebar.number_input('Presión de operación (P, atm)', value=1.0, format="%.2f")
L = st.sidebar.number_input(f'Flujo de agua (L, {flow_unit})', value=2200.0, format="%.2f")
G = st.sidebar.number_input(f'Flujo de aire (G, {flow_unit})', value=2000.0, format="%.2f")
tfin = st.sidebar.number_input(f'Temperatura de entrada del agua (tfin, {temp_unit})', value=105.0, format="%.2f")
tini = st.sidebar.number_input(f'Temperatura de salida del agua (tini, {temp_unit})', value=85.0, format="%.2f")

Y1_source_option = st.sidebar.radio("Fuente de Humedad Absoluta (Y1):", 
    ('Ingresar Y1 directamente', 'Calcular Y1 a partir de Bulbo Húmedo', 'Calcular Y1 a partir de Humedad Relativa'))

if Y1_source_option == 'Ingresar Y1 directamente':
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    tw1 = st.sidebar.number_input(f'Bulbo húmedo del aire a la entrada (tw1, {temp_unit})', value=76.0, format="%.2f")
    Y1 = st.sidebar.number_input(f'Humedad absoluta del aire a la entrada (Y1, {Y_unit})', value=0.016, format="%.5f")
elif Y1_source_option == 'Calcular Y1 a partir de Bulbo Húmedo':
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    tw1 = st.sidebar.number_input(f'Bulbo húmedo del aire a la entrada (tw1, {temp_unit})', value=76.0, format="%.2f")
    Y1 = calculate_Y_from_wet_bulb(tG1, tw1, P, opcion_unidades, psychrometric_constant)
else:
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    rh = st.sidebar.number_input('Humedad Relativa (%)', value=50.0)
    Y1 = calculate_Y_from_relative_humidity(tG1, rh, P, opcion_unidades)

KYa = st.sidebar.number_input(f'KYa ({kya_unit})', value=850.0, format="%.2f")

# ==================== CÁLCULOS BASE ====================
try:
    y1 = Y1 / (1 + Y1)
    Gs = G * (1 - y1)
    Hini = calcular_entalpia_aire(tG1, Y1, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)
    Hfin = (L * Cp_default / Gs) * (tfin - tini) + Hini
    H_star_func = interp1d(teq, Heq_data, kind='cubic', fill_value='extrapolate')

    # ==================== CÁLCULO DEL FLUJO MÍNIMO (MODIFICACIÓN TANGENCIA) ====================
    st.subheader('Cálculo del Flujo Mínimo de Aire (Método de Tangencia)')
    
    # El punto pivote es la base de la torre: Agua fría (tini) y Aire de entrada (Hini)
    t_search = np.linspace(tini + 0.01, tfin, 1000)
    H_star_search = H_star_func(t_search)
    
    # Pendientes desde (tini, Hini) a cada punto de la curva de equilibrio
    pendientes = (H_star_search - Hini) / (t_search - tini)
    
    # El pinch ocurre en la pendiente MÁXIMA que toca la curva dentro del rango
    idx_pinch = np.argmax(pendientes)
    m_max_global = pendientes[idx_pinch]
    t_pinch_global = t_search[idx_pinch]
    H_pinch_global = H_star_search[idx_pinch]
    
    Gs_min = (L * Cp_default) / m_max_global
    G_min = Gs_min / (1 - y1)
    
    st.success(f"✅ Punto de pinch calculado en T = {t_pinch_global:.2f} {temp_unit}")
    st.write(f"- **Gs mínimo:** {Gs_min:.2f} | **G mínimo:** {G_min:.2f}")

    if G <= G_min:
        st.error("Error: El flujo de aire es insuficiente para el enfriamiento deseado.")
        st.stop()

    # ==================== MÉTODO DE MICKLEY ======================
    DH = (Hfin - Hini) / 20
    t_air, H_air, t_op, H_star = [tG1], [Hini], [tini], [H_star_func(tini)]
    Y_air = [Y1]
    segmentos = []

    for _ in range(20):
        H_next = H_air[-1] + DH
        if H_next > Hfin: H_next = Hfin
        t_op_next = (H_next - Hini) * (tfin - tini) / (Hfin - Hini) + tini
        
        # Pendiente de Mickley (t_op - t_air)/(H* - H)
        den = H_star[-1] - H_air[-1]
        t_next = DH * ((t_op[-1] - t_air[-1]) / den) + t_air[-1] if abs(den) > 1e-6 else t_air[-1]
        
        H_air.append(H_next)
        t_air.append(t_next)
        Y_air.append(calcular_Y(H_next, t_next, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor))
        t_op.append(t_op_next)
        H_star.append(H_star_func(t_op_next))
        
        segmentos.append(((t_next, H_next), (t_op_next, H_next)))
        segmentos.append(((t_op_next, H_next), (t_op_next, H_star[-1])))

    # ==================== CÁLCULO DE NtoG, HtoG, Z ====================
    t_wat_int = np.linspace(tini, tfin, 101)
    H_op_int = np.interp(t_wat_int, [tini, tfin], [Hini, Hfin])
    H_st_int = H_star_func(t_wat_int)
    NtoG = np.trapz(1 / (H_st_int - H_op_int), H_op_int)
    HtoG = Gs / KYa
    Z_total = HtoG * NtoG

    st.subheader('Resultados de Diseño')
    st.info(f"NtoG: {NtoG:.2f} | HtoG: {HtoG:.2f} {length_unit} | Altura Total Z: {Z_total:.2f} {length_unit}")

    # ==================== GRÁFICO FINAL ====================
    fig, ax = plt.subplots(figsize=(10, 7))
    T_plot = np.linspace(min(teq), max(teq), 200)
    ax.plot(T_plot, H_star_func(T_plot), 'b-', label='Equilibrio H*', linewidth=2)
    ax.plot([tini, tfin], [Hini, Hfin], 'r-', label='Línea Operación', linewidth=2)
    ax.plot(t_air, H_air, 'ko-', label='Evolución Aire (Mickley)', markersize=3)
    
    # Línea de G_min (Tangente)
    ax.plot([tini, t_pinch_global], [Hini, H_pinch_global], 'r--', alpha=0.6, label='Tangente G_min')
    ax.scatter(t_pinch_global, H_pinch_global, color='red', s=100, zorder=5, label='Pinch Point')

    for seg in segmentos:
        (x1, y1), (x2, y2) = seg
        ax.plot([x1, x2], [y1, y2], 'gray', linestyle='--', linewidth=0.5)

    ax.set_xlabel(f'Temperatura ({temp_unit})')
    ax.set_ylabel(f'Entalpía ({enthalpy_unit})')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error en los cálculos: {e}")
