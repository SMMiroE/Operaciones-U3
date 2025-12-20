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
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    tw1 = st.sidebar.number_input(f'Bulbo húmedo del aire a la entrada (tw1, {temp_unit})', value=76.0, format="%.2f")
    Y1 = st.sidebar.number_input(f'Humedad absoluta del aire a la entrada (Y1, {Y_unit})', value=0.016, format="%.5f")

elif Y1_source_option == 'Calcular Y1 a partir de Bulbo Húmedo':
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    tw1 = st.sidebar.number_input(f'Bulbo húmedo del aire a la entrada (tw1, {temp_unit})', value=76.0, format="%.2f")
    st.sidebar.write("Calculando Y1 a partir de Bulbo Húmedo:")
    try:
        calculated_Y1 = calculate_Y_from_wet_bulb(tG1, tw1, P, opcion_unidades, psychrometric_constant)
        if calculated_Y1 == float('inf'):
            st.sidebar.error("Error al calcular Y1: Posible saturación o datos inconsistentes. Ajuste las temperaturas de bulbo seco y húmedo.")
            Y1 = 0.016
        else:
            Y1 = calculated_Y1
            st.sidebar.info(f"Y1 calculado: **{Y1:.5f}** ({Y_unit})")
    except Exception as e:
        st.sidebar.error(f"Error en el cálculo de Y1: {e}. Usando valor por defecto.")
        Y1 = 0.016

elif Y1_source_option == 'Calcular Y1 a partir de Humedad Relativa':
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    relative_humidity = st.sidebar.number_input('Humedad Relativa a la entrada (HR, %)', value=50.0, min_value=0.0, max_value=100.0, format="%.1f")
    tw1 = 0.0
    st.sidebar.write("Calculando Y1 a partir de Humedad Relativa:")
    try:
        calculated_Y1 = calculate_Y_from_relative_humidity(tG1, relative_humidity, P, opcion_unidades)
        if calculated_Y1 == float('inf'):
            st.sidebar.error("Error al calcular Y1: Posible saturación o datos inconsistentes. Ajuste la temperatura de bulbo seco y la humedad relativa.")
            Y1 = 0.016
        else:
            Y1 = calculated_Y1
            st.sidebar.info(f"Y1 calculado: **{Y1:.5f}** ({Y_unit})")
    except Exception as e:
        st.sidebar.error(f"Error en el cálculo de Y1: {e}. Usando valor por defecto.")
        Y1 = 0.016

KYa = st.sidebar.number_input(f'Coef. volumétrico de transferencia de materia (KYa, {kya_unit})', value=850.0, format="%.2f")

# ==================== CÁLCULOS BASE ====================
try:
    y1 = Y1 / (1 + Y1)
    Gs = G * (1 - y1)

    Hini = calcular_entalpia_aire(tG1, Y1, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)

    if Gs == 0:
        st.error("Error: El flujo de aire seco (Gs) no puede ser cero. Revise el flujo de aire (G) y la humedad (Y1).")
        st.stop()

    Hfin = (L * Cp_default / Gs) * (tfin - tini) + Hini

    if tini >= tfin:
        st.warning("Advertencia: La temperatura de salida del agua (tini) debe ser menor que la de entrada (tfin) para un enfriamiento.")

    # ==================== CURVA H*(t) ====================
    # Spline cúbica para Mickley e integración
    H_star_func = interp1d(teq, Heq_data, kind='cubic', fill_value='extrapolate')
    tck = splrep(teq, Heq_data, k=3)

    def dH_star_dt_func_spline(t_val):
        t_val_clipped = np.clip(t_val, np.min(teq), np.max(teq))
        return splev(t_val_clipped, tck, der=1)

    # Versión lineal para cálculo robusto de Gs_min
    H_star_lin = interp1d(teq, Heq_data, kind='linear', fill_value='extrapolate')

    # ==================== CÁLCULO DEL FLUJO MÍNIMO DE AIRE ====================
    st.subheader('Cálculo del Flujo Mínimo de Aire')

    Gs_min = 1.0
    Hfin_min = Hini + (L * Cp_default / Gs_min) * (tfin - tini)
    t_pinch_for_Gs_min = tini
    H_pinch_value = H_star_func(tini)

    try:
        if tini >= tfin:
            st.error("Error: La temperatura de salida del agua (tini) debe ser menor que la de entrada (tfin) para calcular el flujo mínimo.")
            st.stop()

        # Entalpía de equilibrio a la entrada caliente
        H_eq_tfin = float(H_star_lin(tfin))

        # Rango de búsqueda de T_pinch entre tini y tfin
        t_search_range = np.linspace(tini + 1e-6, tfin, 500)
        t_search_range = t_search_range[(t_search_range >= np.min(teq)) & (t_search_range <= np.max(teq))]

        if t_search_range.size == 0:
            st.error("Error: El rango de temperaturas del agua no se superpone con los datos de la curva de equilibrio. Ajuste los datos de entrada o la curva de equilibrio.")
            st.stop()

        slopes_to_equilibrium = []
        for t_eq_point in t_search_range:
            H_star_at_t_eq = float(H_star_lin(t_eq_point))
            if abs(t_eq_point - tini) < 1e-9:
                continue

            slope_candidate = (H_star_at_t_eq - Hini) / (t_eq_point - tini)
            if slope_candidate <= 0:
                continue

            # entalpía de la recta candidata en tfin
            H_line_at_tfin = Hini + slope_candidate * (tfin - tini)

            # condición física: la recta mínima no debe estar por encima de H* en tfin
            if H_line_at_tfin <= H_eq_tfin + 1e-6:
                slopes_to_equilibrium.append((slope_candidate, t_eq_point, H_star_at_t_eq, H_line_at_tfin))

        if not slopes_to_equilibrium:
            st.error("No se encontraron pendientes válidas para calcular el flujo mínimo. Revise los datos de entrada o la viabilidad del diseño.")
            st.stop()

        # Pendiente máxima válida
        m_min, t_pinch_for_Gs_min, H_pinch_value, H_line_at_tfin = max(slopes_to_equilibrium, key=lambda item: item[0])

        if m_min <= 0:
            st.error("Error: La pendiente máxima calculada para el flujo mínimo es cero o negativa. Esto indica un problema con los datos o que el enfriamiento deseado es imposible.")
            st.stop()

        Gs_min = (L * Cp_default) / m_min
        Hfin_min = Hini + m_min * (tfin - tini)
        G_min = Gs_min / (1 - y1)

        st.write(f"  - Punto de pellizco (temperatura): **{t_pinch_for_Gs_min:.2f}** {temp_unit}")
        st.write(f"  - Punto de pellizco (entalpía): **{H_pinch_value:.2f}** {enthalpy_unit}")
        st.write(f"  - Flujo mínimo de aire seco (Gs_min): **{Gs_min:.2f}** {flow_unit.replace('aire', 'aire seco')}")
        st.write(f"  - Flujo mínimo de aire (G_min): **{G_min:.2f}** {flow_unit}")
        st.write(f"  - Entalpía del aire a la salida con flujo mínimo (Hfin_min): **{Hfin_min:.2f}** {enthalpy_unit}")

        if G < G_min:
            st.warning(f"Advertencia: El flujo de aire actual (G={G:.2f} {flow_unit}) es menor que el flujo mínimo requerido (G_min={G_min:.2f} {flow_unit}). Esto indica que el enfriamiento deseado es imposible con el flujo de aire actual.")
        elif G / G_min < 1.1:
            st.warning(f"Advertencia: El flujo de aire actual (G={G:.2f} {flow_unit}) está muy cerca del flujo mínimo requerido (G_min={G_min:.2f} {flow_unit}). Operar tan cerca del mínimo puede requerir una torre de enfriamiento muy grande y costosa.")

    except Exception as e:
        st.error(f"No se pudo calcular el flujo mínimo de aire. Revise los datos de entrada o la viabilidad del diseño. Detalle del error: {e}")
        Gs_min = 1.0
        Hfin_min = Hini + (L * Cp_default / Gs_min) * (tfin - tini)
        t_pinch_for_Gs_min = tini
        H_pinch_value = H_star_func(tini)

    # ==================== MÉTODO DE MICKLEY ======================
    DH = (Hfin - Hini) / 20

    if DH <= 0:
        st.error("Error: El incremento de entalpía (DH) es cero o negativo. Revise las temperaturas del agua (tini, tfin) y flujos (L, G).")
        st.stop()

    t_air = [tG1]
    H_air = [Hini]
    Y_air = [calcular_Y(Hini, tG1, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)]
    t_op = [tini]
    H_op = [Hini]
    H_star = [H_star_func(tini)]
    segmentos = []

    max_iterations = 1000
    i_loop = 0

    while True:
        i_loop += 1
        if i_loop > max_iterations:
            st.warning(f"Advertencia: Bucle de Mickley excedió {max_iterations} iteraciones. Revisar datos de entrada o divergencia.")
            break

        H_next = H_air[-1] + DH

        if H_next >= Hfin:
            H_next = Hfin
            t_op_next = (H_next - Hini) * (tfin - tini) / (Hfin - Hini) + tini
            H_star_next = H_star_func(t_op_next)

            if len(H_air) > 1 and len(t_air) > 1 and len(t_op) > 1 and len(H_star) > 1:
                t_prev = t_air[-1]
                H_prev = H_air[-1]
                t_op_prev = t_op[-1]
                H_star_prev = H_star[-1]
                DH_last_step = H_next - H_prev
                if abs(H_star_prev - H_prev) < 1e-6:
                    t_next = t_prev
                else:
                    t_next = DH_last_step * ((t_op_prev - t_prev) / (H_star_prev - H_prev)) + t_prev
            else:
                t_next = tG1

            H_star_tnext = H_star_func(t_next)
            Y_next = calcular_Y(H_next, t_next, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)

            H_air.append(H_next)
            t_air.append(t_next)
            Y_air.append(Y_next)
            t_op.append(t_op_next)
            H_op.append(H_next)
            H_star.append(H_star_next)
            break

        t_op_next = (H_next - Hini) * (tfin - tini) / (Hfin - Hini) + tini
        H_star_next = H_star_func(t_op_next)

        if abs(H_star[-1] - H_air[-1]) < 1e-6:
            t_next = t_air[-1]
        else:
            t_next = DH * ((t_op[-1] - t_air[-1]) / (H_star[-1] - H_air[-1])) + t_air[-1]

        H_star_tnext = H_star_func(t_next)
        Y_next = calcular_Y(H_next, t_next, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)

        if H_next > Hfin or (H_next - H_star_tnext) > 0:
            if H_next > Hfin:
                H_next = Hfin
                t_op_next = (H_next - Hini) * (tfin - tini) / (Hfin - Hini) + tini
                H_star_next = H_star_func(t_op_next)

                if len(H_air) > 1 and len(t_air) > 1 and len(t_op) > 1 and len(H_star) > 1:
                    t_prev = t_air[-1]
                    H_prev = H_air[-1]
                    t_op_prev = t_op[-1]
                    H_star_prev = H_star[-1]
                    DH_last_step = H_next - H_prev
                    if abs(H_star_prev - H_prev) < 1e-6:
                        t_next = t_prev
                    else:
                        t_next = DH_last_step * ((t_op_prev - t_prev) / (H_star_prev - H_prev)) + t_prev
                else:
                    t_next = tG1
                Y_next = calcular_Y(H_next, t_next, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)

                H_air.append(H_next)
                t_air.append(t_next)
                Y_air.append(Y_next)
                t_op.append(t_op_next)
                H_op.append(H_next)
                H_star.append(H_star_next)
            break

        H_air.append(H_next)
        t_air.append(t_next)
        Y_air.append(Y_next)
        t_op.append(t_op_next)
        H_op.append(H_next)
        H_star.append(H_star_next)

        segmentos.append(((t_next, H_next), (t_op_next, H_next)))
        segmentos.append(((t_op_next, H_next), (t_op_next, H_star_next)))
        segmentos.append(((t_op_next, H_star_next), (t_next, H_next)))

    if len(H_air) <= 1:
        st.warning("No se pudo generar la curva de evolución del aire. Revise las temperaturas y flujos de entrada.")
        st.stop()

    # ==================== CÁLCULO DE NtoG ====================
    n_pasos_integracion = 100
    dt_integracion = (tfin - tini) / n_pasos_integracion
    t_water_integracion = np.linspace(tini, tfin, n_pasos_integracion + 1)

    H_op_vals_integracion = np.interp(t_water_integracion, [tini, tfin], [Hini, Hfin])
    H_star_vals_integracion = H_star_func(t_water_integracion)

    f_T_integracion = []
    for i in range(len(t_water_integracion)):
        delta = H_star_vals_integracion[i] - H_op_vals_integracion[i]
        if abs(delta) < 1e-6:
            st.error(f"Error: La línea de operación está muy cerca o cruza la curva de equilibrio en t={t_water_integracion[i]:.2f}. Verifique los datos de entrada o la viabilidad del diseño. No se puede calcular NtoG.")
            st.stop()
        f_T_integracion.append(1 / delta)

    dHdT_integracion = (Hfin - Hini) / (tfin - tini)
    NtoG = 0
    for i in range(1, len(t_water_integracion)):
        NtoG += 0.5 * dt_integracion * (f_T_integracion[i] + f_T_integracion[i - 1])
    NtoG *= dHdT_integracion

    # ======== CÁLCULO DE HtoG, Z y agua de reposición ====================
    if KYa == 0:
        st.error("Error: KYa no puede ser cero. Revise el coeficiente de transferencia de masa.")
        st.stop()

    HtoG = Gs / KYa
    Z_total = HtoG * NtoG
    Lrep = Gs * (Y_air[-1] - Y1)

    # ==================== RESULTADOS ====================
    st.subheader('Resultado del Cálculo')
    st.info("**Línea de operación:**")
    st.write(f"  - Cabeza de la torre (entrada de agua): (t = {tfin:.2f} {temp_unit}, H = {Hfin:.2f} {enthalpy_unit})")
    st.write(f"  - Base de la torre (salida de agua): (t = {tini:.2f} {temp_unit}, H = {Hini:.2f} {enthalpy_unit})")
    st.info("**Parámetros de Diseño:**")
    st.write(f"  - Humedad absoluta del aire a la salida: **Y = {Y_air[-1]:.5f}** (masa vapor de agua/masa de aire seco)")
    st.write(f"  - Agua evaporada (reposición): **Lrep = {Lrep:.2f} {flow_unit}**")
    st.write(f"  - Número de unidades de transferencia (NtoG): **{NtoG:.2f}**")
    st.write(f"  - Altura de unidad de transferencia (HtoG): **{HtoG:.2f} {length_unit}**")
    st.write(f"  - Altura total del relleno (Z): **{Z_total:.2f} {length_unit}**")

    # ==================== GRÁFICO FINAL ====================
    st.subheader('Diagrama de Entalpía-Temperatura')

    fig, ax = plt.subplots(figsize=(10, 7))

    T_plot = np.linspace(min(teq), max(teq) + 10, 200)
    ax.plot(T_plot, H_star_func(T_plot), label=f'Curva de equilibrio H*({temp_unit})', linewidth=2, color='blue')
    ax.plot([tini, tfin], [Hini, Hfin], 'r-', label=f'Línea de operación Hop({temp_unit})', linewidth=2)
    ax.plot(t_air, H_air, 'ko-', label=f'Curva de evolución del aire H({temp_unit})', markersize=4, linewidth=1)

    # Dibujo del triángulo inicial
    A_plot = (tG1, Hini)
    B_plot = (tini, Hini)
    C_plot = (tini, H_star_func(tini))
    ax.plot([A_plot[0], B_plot[0]], [A_plot[1], B_plot[1]], 'gray', linestyle='--')
    ax.plot([B_plot[0], C_plot[0]], [B_plot[1], C_plot[1]], 'gray', linestyle='--')
    ax.plot([A_plot[0], C_plot[0]], [A_plot[1], C_plot[1]], 'gray', linestyle='--')

    for seg in segmentos:
        (x1, y1), (x2, y2) = seg
        ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, linestyle='--')

    ax.set_xlabel(f'Temperatura del agua ({temp_unit})')
    ax.set_ylabel(f'Entalpía del aire húmedo ({enthalpy_unit})')
    ax.set_title('Método de Mickley - Torre de Enfriamiento')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(min(tini, tG1) - 10, max(tfin, max(t_air)) + 10)
    ax.set_ylim(min(Hini, min(Heq_data)) - 10, max(Hfin, max(Heq_data)) + 30)

    st.pyplot(fig)

except Exception as e:
    st.error(f"Ha ocurrido un error en los cálculos. Por favor, revise los datos de entrada. Detalle del error: {e}")
