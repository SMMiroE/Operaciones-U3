# ==================== IMPORTACIÓN DE LIBRERÍAS ====================
import streamlit as st # Importa la librería Streamlit
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splev, splrep # Import splev and splrep
from scipy.optimize import fsolve # Para resolver numéricamente el punto de pellizco

# ==================== CONFIGURACIÓN DE LA PÁGINA (OPCIONAL) ====================
st.set_page_config(
    page_title="Torres de Enfriamiento OU3 FICA-UNSL",
    layout="centered", # o "wide" para más espacio
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
    Heq_data = np.array([4.074, 7.545, 18.780, 36.020, 64.090, 112.0, 198.0]) # BTU/lb aire seco
    Cp_default = 1.0 # calor específico del agua, Btu/(lb °F)
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
    Heq_data = np.array([9479, 29360, 57570, 100030, 166790, 275580, 461500]) # J/kg aire seco
    Cp_default = 4186 # calor específico del agua, J/(kg °C)
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
    """Entalpía del aire húmedo."""
    return (cp_air_dry + cp_vapor * Y) * (t - temp_ref) + latent_ref * Y

def calcular_Y(H, t, temp_ref, latent_ref, cp_air_dry, cp_vapor):
    """Humedad absoluta Y a partir de H y t."""
    return (H - cp_air_dry * (t - temp_ref)) / (cp_vapor * (t - temp_ref) + latent_ref)

def get_saturation_vapor_pressure(temperature, units_system):
    """Calcula la presión de vapor de saturación del agua (Magnus)."""
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
    if Pv < 0:
        Pv = 0.0
    if (P_total - Pv) <= 0:
        return float('inf')
    return 0.62198 * (Pv / (P_total - Pv))

def calculate_Y_from_relative_humidity(t_dry_bulb, relative_humidity_percent, total_pressure_atm, units_system):
    if units_system == 'Sistema Internacional':
        P_total = total_pressure_atm * 101.325 
    else:
        P_total = total_pressure_atm * 14.696 

    P_ws_tdb = get_saturation_vapor_pressure(t_dry_bulb, units_system)
    Pv = (relative_humidity_percent / 100.0) * P_ws_tdb
    if (P_total - Pv) <= 0:
        return float('inf')
    return 0.62198 * (Pv / (P_total - Pv))

# ==================== ENTRADA DE DATOS DEL PROBLEMA ====================
st.sidebar.header('Datos del Problema')

P = st.sidebar.number_input('Presión de operación (P, atm)', value=1.0, format="%.2f")
L = st.sidebar.number_input(f'Flujo de agua (L, {flow_unit})', value=2200.0, format="%.2f")
G = st.sidebar.number_input(f'Flujo de aire (G, {flow_unit})', value=2000.0, format="%.2f")
tfin = st.sidebar.number_input(f'Temperatura de entrada del agua (tfin, {temp_unit})', value=105.0, format="%.2f")
tini = st.sidebar.number_input(f'Temperatura de salida del agua (tini, {temp_unit})', value=85.0, format="%.2f")

Y1_source_option = st.sidebar.radio(
    "Fuente de Humedad Absoluta (Y1):",
    ('Ingresar Y1 directamente', 'Calcular Y1 a partir de Bulbo Húmedo', 'Calcular Y1 a partir de Humedad Relativa')
)

Y1 = 0.016 

if Y1_source_option == 'Ingresar Y1 directamente':
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    tw1 = st.sidebar.number_input(f'Bulbo húmedo del aire a la entrada (tw1, {temp_unit})', value=76.0, format="%.2f")
    Y1 = st.sidebar.number_input(f'Humedad absoluta del aire a la entrada (Y1, {Y_unit})', value=0.016, format="%.5f")

elif Y1_source_option == 'Calcular Y1 a partir de Bulbo Húmedo':
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    tw1 = st.sidebar.number_input(f'Bulbo húmedo del aire a la entrada (tw1, {temp_unit})', value=76.0, format="%.2f")
    try:
        calculated_Y1 = calculate_Y_from_wet_bulb(tG1, tw1, P, opcion_unidades, psychrometric_constant)
        if calculated_Y1 == float('inf'):
            st.sidebar.error("Error en Y1. Ajuste temperaturas.")
            Y1 = 0.016
        else:
            Y1 = calculated_Y1
            st.sidebar.info(f"Y1 calculado: **{Y1:.5f}**")
    except:
        Y1 = 0.016

elif Y1_source_option == 'Calcular Y1 a partir de Humedad Relativa':
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    relative_humidity = st.sidebar.number_input('Humedad Relativa (%)', value=50.0, min_value=0.0, max_value=100.0)
    try:
        calculated_Y1 = calculate_Y_from_relative_humidity(tG1, relative_humidity, P, opcion_unidades)
        if calculated_Y1 == float('inf'):
            st.sidebar.error("Error en Y1. Ajuste parámetros.")
            Y1 = 0.016
        else:
            Y1 = calculated_Y1
            st.sidebar.info(f"Y1 calculado: **{Y1:.5f}**")
    except:
        Y1 = 0.016

KYa = st.sidebar.number_input(f'KYa ({kya_unit})', value=850.0, format="%.2f")

# ==================== CÁLCULOS BASE ====================
try:
    y1 = Y1 / (1 + Y1)
    Gs = G * (1 - y1)
    Hini = calcular_entalpia_aire(tG1, Y1, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)

    if Gs == 0:
        st.error("Gs no puede ser cero.")
        st.stop()

    Hfin = (L * Cp_default / Gs) * (tfin - tini) + Hini

    # ==================== CURVA H*(t) ====================
    H_star_func = interp1d(teq, Heq_data, kind='cubic', fill_value='extrapolate')
    tck = splrep(teq, Heq_data, k=3)

    # ==================== FLUJO MÍNIMO ====================
    t_rango_check = np.linspace(tini - 5, tfin + 10, 1000)
    from scipy.optimize import minimize

    def objetivo_tangencia(m):
        h_op = Hini + m * (t_rango_check - tini)
        h_eq = H_star_func(t_rango_check)
        distancia_minima = np.min(h_eq - h_op)
        return distancia_minima**2

    res_m = minimize(objetivo_tangencia, x0=[(H_star_func(tfin)-Hini)/(tfin-tini)], bounds=[(0.01, None)])
    m_tangente = res_m.x[0]
    
    m_tope = (H_star_func(tfin) - Hini) / (tfin - tini)
    if (H_star_func(tfin) - (Hini + m_tangente * (tfin - tini))) < 0:
        m_max_global = m_tope
        t_pinch_global = tfin
    else:
        m_max_global = m_tangente
        h_op_tangente = Hini + m_tangente * (t_rango_check - tini)
        idx_pinch = np.argmin(H_star_func(t_rango_check) - h_op_tangente)
        t_pinch_global = t_rango_check[idx_pinch]

    H_pinch_global = H_star_func(t_pinch_global)
    Gs_min = (L * Cp_default) / m_max_global

    # ==================== MÉTODO DE MICKLEY ======================
    DH = (Hfin - Hini) / 20
    t_air, H_air, Y_air, t_op, H_op, H_star = [tG1], [Hini], [Y1], [tini], [Hini], [H_star_func(tini)]
    segmentos = []

    for _ in range(1000):
        H_next = H_air[-1] + DH
        if H_next >= Hfin:
            H_next = Hfin
            t_op_next = (H_next - Hini) * (tfin - tini) / (Hfin - Hini) + tini
            t_next = (H_next - H_air[-1]) * ((t_op[-1] - t_air[-1]) / (H_star[-1] - H_air[-1])) + t_air[-1] if abs(H_star[-1]-H_air[-1])>1e-6 else t_air[-1]
            H_air.append(H_next); t_air.append(t_next); Y_air.append(calcular_Y(H_next, t_next, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor))
            t_op.append(t_op_next); H_op.append(H_next); H_star.append(H_star_func(t_op_next))
            break
        
        t_op_next = (H_next - Hini) * (tfin - tini) / (Hfin - Hini) + tini
        t_next = DH * ((t_op[-1] - t_air[-1]) / (H_star[-1] - H_air[-1])) + t_air[-1] if abs(H_star[-1]-H_air[-1])>1e-6 else t_air[-1]
        
        H_air.append(H_next); t_air.append(t_next); Y_air.append(calcular_Y(H_next, t_next, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor))
        t_op.append(t_op_next); H_op.append(H_next); H_star.append(H_star_func(t_op_next))
        segmentos.append(((t_next, H_next), (t_op_next, H_next)))
        segmentos.append(((t_op_next, H_next), (t_op_next, H_star_func(t_op_next))))

    # ==================== NtoG y DISEÑO ====================
    t_int = np.linspace(tini, tfin, 100)
    H_op_int = np.interp(t_int, [tini, tfin], [Hini, Hfin])
    H_star_int = H_star_func(t_int)
    NtoG = np.trapz(1/(H_star_int - H_op_int), H_op_int)
    HtoG = Gs / KYa
    Z_total = HtoG * NtoG
    Lrep = Gs * (Y_air[-1] - Y1)

    # ==================== RESULTADOS ====================
    st.markdown("### 📊 Resultados")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Altura Z:** {Z_total:.2f} {length_unit}")
        st.write(f"**NtoG:** {NtoG:.2f}")
    with c2:
        st.write(f"**Gs Mínimo:** {Gs_min:.1f} {Gs_unit}")
        st.write(f"**L Reposición:** {Lrep:.2f} {flow_unit}")

    fig, ax = plt.subplots()
    T_p = np.linspace(min(teq), max(teq), 100)
    ax.plot(T_p, H_star_func(T_p), label='Equilibrio')
    ax.plot([tini, tfin], [Hini, Hfin], 'r-', label='Op.')
    ax.plot(t_air, H_air, 'k--', label='Aire (Mickley)')
    ax.legend(); ax.grid(True)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error en cálculos: {e}")

with st.expander("📚 Ver mas información"):
    st.info("Modelo: Estado Estacionario, Adiabático, L/G Constante, Resistencia en fase gas.")
    st.latex(r"Z = H_{toG} \times N_{toG}")
