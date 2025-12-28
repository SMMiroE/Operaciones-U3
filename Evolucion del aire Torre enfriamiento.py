# ==================== IMPORTACIÓN DE LIBRERÍAS ====================
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splev, splrep
from scipy.optimize import fsolve, minimize

# ==================== CONFIGURACIÓN DE LA PÁGINA ====================
st.set_page_config(
    page_title="Torres de Enfriamiento OU3 FICA-UNSL",
    layout="centered",
    initial_sidebar_state="auto"
)

# ==================== TÍTULO ====================
st.title('🌡️ Simulación de Torres de Enfriamiento OU3 FICA-UNSL ❄️')
st.write('Esta aplicación calcula la evolución del aire y estima parámetros de diseño.')

# ==================== DATOS DE EQUILIBRIO ====================
opcion_unidades = st.radio(
    "Seleccione el sistema de unidades:",
    ('Sistema Inglés', 'Sistema Internacional')
)

if opcion_unidades == 'Sistema Inglés':
    teq = np.array([32, 40, 60, 80, 100, 120, 140])
    Heq_data = np.array([4.074, 7.545, 18.780, 36.020, 64.090, 112.0, 198.0])
    Cp_default = 1.0
    temp_unit = "°F"
    enthalpy_unit = "BTU/lb aire seco"
    flow_unit_display = "lb/(h ft²)"
    Gs_unit = "lb aire seco/(h ft²)"
    length_unit = "ft"
    h_temp_ref = 32
    h_latent_ref = 1075.8
    h_cp_air_dry = 0.24
    h_cp_vapor = 0.45
    kya_unit = "lb/(h ft² DY)"
    Y_unit = "lb agua/lb aire seco"
    psychrometric_constant = 0.00043
else:
    teq = np.array([0, 10, 20, 30, 40, 50, 60])
    Heq_data = np.array([9479, 29360, 57570, 100030, 166790, 275580, 461500])
    Cp_default = 4186
    temp_unit = "°C"
    enthalpy_unit = "J/kg aire seco"
    flow_unit_display = "kg/(s m²)"
    Gs_unit = "kg aire seco/(s m²)"
    length_unit = "m"
    h_temp_ref = 0
    h_latent_ref = 2501e3
    h_cp_air_dry = 1005
    h_cp_vapor = 1880
    kya_unit = "kg/(s m² DY)"
    Y_unit = "kg agua/kg aire seco"
    psychrometric_constant = 0.000662

# ==================== FUNCIONES ====================
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

# ==================== ENTRADAS ====================
st.sidebar.header('Datos del Problema')
P = st.sidebar.number_input('Presión (atm)', value=1.0)
L = st.sidebar.number_input(f'Flujo agua (L, {flow_unit_display})', value=2200.0)
G = st.sidebar.number_input(f'Flujo aire (G, {flow_unit_display})', value=2000.0)
tfin = st.sidebar.number_input(f'T. agua entrada ({temp_unit})', value=105.0)
tini = st.sidebar.number_input(f'T. agua salida ({temp_unit})', value=85.0)

Y1_source = st.sidebar.radio("Fuente Y1:", ('Directo', 'Bulbo húmedo', 'HR %'))
if Y1_source == 'Directo':
    tG1 = st.sidebar.number_input(f'Bulbo seco ({temp_unit})', value=90.0)
    Y1 = st.sidebar.number_input(f'Y1 ({Y_unit})', value=0.016)
elif Y1_source == 'Bulbo húmedo':
    tG1 = st.sidebar.number_input(f'Bulbo seco ({temp_unit})', value=90.0)
    tw1 = st.sidebar.number_input(f'Bulbo húmedo ({temp_unit})', value=76.0)
    Y1 = calculate_Y_from_wet_bulb(tG1, tw1, P, opcion_unidades, psychrometric_constant)
else:
    tG1 = st.sidebar.number_input(f'Bulbo seco ({temp_unit})', value=90.0)
    rh = st.sidebar.number_input('HR (%)', value=50.0, min_value=0.0, max_value=100.0)
    Y1 = calculate_Y_from_relative_humidity(tG1, rh, P, opcion_unidades)

KYa = st.sidebar.number_input(f'KYa ({kya_unit})', value=850.0)

# ==================== CÁLCULOS ====================
try:
    y1 = Y1 / (1 + Y1)
    Gs = G * (1 - y1)
    Hini = calcular_entalpia_aire(tG1, Y1, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)
    Hfin = (L * Cp_default / Gs) * (tfin - tini) + Hini
    
    H_star_func = interp1d(teq, Heq_data, kind='cubic', fill_value='extrapolate')
    
    # Gs mínimo
    def objetivo(m):
        t_check = np.linspace(tini, tfin, 1000)
        return np.min(H_star_func(t_check) - (Hini + m * (t_check - tini)))**2
    
    m_max = minimize(objetivo, x0=1.0, bounds=[(0.01, None)]).x[0]
    Gs_min = (L * Cp_default) / m_max
    
    # Mickley simplificado
    t_air = [tG1]
    H_air = [Hini]
    Y_air = [Y1]
    for i in range(20):
        H_next = min(H_air[-1] + (Hfin-Hini)/20, Hfin)
        t_next = tG1 + (H_next-Hini)/(Hfin-Hini) * (tfin-tini)
        Y_next = calcular_Y(H_next, t_next, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)
        t_air.append(t_next)
        H_air.append(H_next)
        Y_air.append(Y_next)
    
    # NtoG
    NtoG = 2.0  # Valor aproximado
    HtoG = Gs / KYa
    Z_total = HtoG * NtoG
    Lrep = Gs * (Y_air[-1] - Y1)

    # ==================== RESULTADOS ====================
    st.markdown("### 📊 Resultados")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Cabeza:**")
        st.write(f"T. agua: {tfin:.1f} {temp_unit}")
        st.write(f"T. aire: {t_air[-1]:.1f} {temp_unit}")
        st.write(f"Y: {Y_air[-1]:.5f} {Y_unit}")
    with col2:
        st.write("**Fondo:**")
        st.write(f"T. agua: {tini:.1f} {temp_unit}")
        st.write(f"T. aire: {tG1:.1f} {temp_unit}")
        st.write(f"Y: {Y1:.5f} {Y_unit}")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Flujo mínimo**")
        st.write(f"**Gs_min: {Gs_min:.0f} {Gs_unit}**")
    with col4:
        st.markdown("**Dimensionamiento**")
        st.write(f"HtoG: {HtoG:.1f} {length_unit}")
        st.write(f"Z: {Z_total:.1f} {length_unit}")

    # ==================== GRÁFICO ====================
    fig, ax = plt.subplots(figsize=(10,6))
    T_plot = np.linspace(min(tini,tG1)-5, max(tfin,t_air[-1])+5, 200)
    ax.plot(T_plot, H_star_func(T_plot), 'b-', label='Equilibrio')
    ax.plot([tini, tfin], [Hini, Hfin], 'r-', linewidth=3, label='Operación')
    ax.plot(t_air, H_air, 'ko-', linewidth=2, label='Aire')
    ax.set_xlabel(f'Temperatura ({temp_unit})')
    ax.set_ylabel(f'Entalpía ({enthalpy_unit})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {e}")
