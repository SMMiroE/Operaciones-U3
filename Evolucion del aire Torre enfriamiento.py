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

# ==================== DATOS DE EQUILIBRIO - CORREGIDO ====================
opcion_unidades = st.radio(
    "Seleccione el sistema de unidades:",
    ('Sistema Inglés', 'Sistema Internacional')
)

if opcion_unidades == 'Sistema Inglés':
    teq = np.array([32, 40, 60, 80, 100, 120, 140])  # °F
    Heq_data = np.array([4.074, 7.545, 18.780, 36.020, 64.090, 112.0, 198.0])  # BTU/lb aire seco
    Cp_default = 1.0  # Btu/(lb °F)
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
    cp_unit = "BTU/(lb agua °F)"
    Y_unit = "lb agua/lb aire seco"
    psychrometric_constant = 0.00043  # CORREGIDO psi/°F
else:  # Sistema Internacional
    teq = np.array([0, 10, 20, 30, 40, 50, 60])  # °C
    Heq_data = np.array([9479, 29360, 57570, 100030, 166790, 275580, 461500])  # J/kg aire seco
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
    cp_unit = "J/(kg agua °C)"
    Y_unit = "kg agua/kg aire seco"
    psychrometric_constant = 0.000662  # kPa^-1

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
        return P_ws_kPa / 6.89476  # kPa a psi

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
P = st.sidebar.number_input('Presión (atm)', value=1.0, format="%.2f")
L = st.sidebar.number_input(f'Flujo agua (L, {flow_unit_display})', value=2200.0, format="%.2f")
G = st.sidebar.number_input(f'Flujo aire (G, {flow_unit_display})', value=2000.0, format="%.2f")
tfin = st.sidebar.number_input(f'T. agua entrada (tfin, {temp_unit})', value=105.0, format="%.2f")
tini = st.sidebar.number_input(f'T. agua salida (tini, {temp_unit})', value=85.0, format="%.2f")

Y1_source_option = st.sidebar.radio("Fuente Y1:", ('Ingresar Y1', 'Bulbo Húmedo', 'Humedad Relativa'))
Y1 = 0.016

if Y1_source_option == 'Ingresar Y1':
    tG1 = st.sidebar.number_input(f'Bulbo seco (tG1, {temp_unit})', value=90.0, format="%.2f")
    Y1 = st.sidebar.number_input(f'Humedad (Y1, {Y_unit})', value=0.016, format="%.5f")
elif Y1_source_option == 'Bulbo Húmedo':
    tG1 = st.sidebar.number_input(f'Bulbo seco (tG1, {temp_unit})', value=90.0, format="%.2f")
    tw1 = st.sidebar.number_input(f'Bulbo húmedo (tw1, {temp_unit})', value=76.0, format="%.2f")
    Y1 = calculate_Y_from_wet_bulb(tG1, tw1, P, opcion_unidades, psychrometric_constant)
    st.sidebar.info(f"Y1 calculado: {Y1:.5f} {Y_unit}")
else:
    tG1 = st.sidebar.number_input(f'Bulbo seco (tG1, {temp_unit})', value=90.0, format="%.2f")
    rh = st.sidebar.number_input('HR (%)', value=50.0, min_value=0.0, max_value=100.0)
    Y1 = calculate_Y_from_relative_humidity(tG1, rh, P, opcion_unidades)
    st.sidebar.info(f"Y1 calculado: {Y1:.5f} {Y_unit}")

KYa = st.sidebar.number_input(f'KYa ({kya_unit})', value=850.0, format="%.2f")

# ==================== CÁLCULOS PRINCIPALES ====================
try:
    y1 = Y1 / (1 + Y1)
    Gs = G * (1 - y1)
    Hini = calcular_entalpia_aire(tG1, Y1, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)
    Hfin = (L * Cp_default / Gs) * (tfin - tini) + Hini

    # Curva equilibrio
    H_star_func = interp1d(teq, Heq_data, kind='cubic', fill_value='extrapolate')
    tck = splrep(teq, Heq_data, k=3)

    # Cálculo Gs mínimo
    t_rango_check = np.linspace(tini - 5, tfin + 10, 1000)
    def objetivo_tangencia(m):
        h_op = Hini + m * (t_rango_check - tini)
        h_eq = H_star_func(t_rango_check)
        return np.min(h_eq - h_op)**2

    m_guess = (H_star_func(tfin) - Hini) / (tfin - tini)
    res_m = minimize(objetivo_tangencia, x0=[m_guess], bounds=[(0.01, None)], method='L-BFGS-B')
    m_max_global = res_m.x[0]
    
    h_op_tangente = Hini + m_max_global * (t_rango_check - tini)
    idx_pinch = np.argmin(H_star_func(t_rango_check) - h_op_tangente)
    t_pinch_global = t_rango_check[idx_pinch]
    
    if t_pinch_global > tfin:
        m_max_global = (H_star_func(tfin) - Hini) / (tfin - tini)
        t_pinch_global = tfin
    
    H_pinch_global = H_star_func(t_pinch_global)
    Gs_min = (L * Cp_default) / m_max_global

    # Método Mickley (simplificado)
    DH = (Hfin - Hini) / 20
    t_air, H_air, Y_air, t_op, H_op, H_star = [tG1], [Hini], [Y1], [tini], [Hini], [H_star_func(tini)]
    
    for i in range(20):
        H_next = min(H_air[-1] + DH, Hfin)
        t_op_next = (H_next - Hini) * (tfin - tini) / (Hfin - Hini) + tini
        H_star_next = H_star_func(t_op_next)
        if abs(H_star[-1] - H_air[-1]) < 1e-6:
            t_next = t_air[-1]
        else:
            t_next = DH * ((t_op[-1] - t_air[-1]) / (H_star[-1] - H_air[-1])) + t_air[-1]
        Y_next = calcular_Y(H_next, t_next, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)
        
        t_air.append(t_next)
        H_air.append(H_next)
        Y_air.append(Y_next)
        t_op.append(t_op_next)
        H_op.append(H_next)
        H_star.append(H_star_next)
        if H_next >= Hfin: break

    # NtoG
    n_pasos = 100
    t_integracion = np.linspace(tini, tfin, n_pasos + 1)
    H_op_int = np.interp(t_integracion, [tini, tfin], [Hini, Hfin])
    H_star_int = H_star_func(t_integracion)
    f_T = 1 / (H_star_int - H_op_int)
    dHdT = (Hfin - Hini) / (tfin - tini)
    NtoG = dHdT * np.trapz(f_T, t_integracion)

    # Resultados finales
    HtoG = Gs / KYa
    Z_total = HtoG * NtoG
    Lrep = Gs * (Y_air[-1] - Y1)

    # ==================== RESULTADOS ====================
    st.markdown("### 📊 Resultados")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cabeza (entrada agua)**")
        st.write(f"T. agua: {tfin:.1f} {temp_unit}")
        st.write(f"T. aire: {t_air[-1]:.1f} {temp_unit}")
        st.write(f"Y aire: {Y_air[-1]:.5f} {Y_unit}")
    
    with col2:
        st.markdown("**Fondo (salida agua)**")
        st.write(f"T. agua: {tini:.1f} {temp_unit}")
        st.write(f"T. aire: {tG1:.1f} {temp_unit}")
        st.write(f"Y aire: {Y1:.5f} {Y_unit}")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Flujo mínimo aire**")
        st.write(f"Gs mínimo: **{Gs_min:.0f} {Gs_unit}**")
        st.write(f"Relación L/Gs: {L/Gs_min:.1f}")
    
    with col4:
        st.markdown("**Dimensionamiento**")
        st.write(f"HtoG: {HtoG:.1f} {length_unit}")
        st.write(f"NtoG: {NtoG:.2f}")
        st.write(f"Altura Z: {Z_total:.1f} {length_unit}")

    st.write(f"💧 Agua reposición: {Lrep:.0f} {flow_unit_display} ({Lrep/L*100:.1f}%)")

    # ==================== GRÁFICO ====================
    st.subheader('Diagrama Entalpía-Temperatura')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    T_plot = np.linspace(min(tini, tG1)-5, max(tfin, max(t_air))+5, 200)
    ax.plot(T_plot, H_star_func(T_plot), 'b-', linewidth=2, label='Curva equilibrio')
    ax.plot([tini, tfin], [Hini, Hfin], 'r-', linewidth=3, label='Línea operación')
    ax.plot(t_air, H_air, 'ko-', linewidth=2, markersize=6, label='Evolución aire')
    ax.plot([tini, t_pinch_global], [Hini, H_pinch_global], 'r--', linewidth=2, label='Gs mínimo')
    
    ax.set_xlabel(f'Temperatura ({temp_unit})')
    ax.set_ylabel(f'Entalpía ({enthalpy_unit})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Método de Mickley - Torre de Enfriamiento')
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error en cálculos: {e}")
