# ==================== IMPORTACIÓN DE LIBRERÍAS ====================
import streamlit as st  # Importa la librería Streamlit
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splev, splrep  # Import splev and splrep
from scipy.optimize import fsolve  # Para resolver numéricamente el punto de pellizco

# ==================== CONFIGURACIÓN DE LA PÁGINA (OPCIONAL) ====================
#st.write("DEBUG: El código se actualizó")
st.set_page_config(
    page_title="Torres de Enfriamiento OU3 FICA-UNSL",
    layout="centered",  # o "wide" para más espacio
    initial_sidebar_state="auto"
)

# ==================== TÍTULO DE LA APLICACIÓN ====================
st.title('🌡️ Simulación de Torres de Enfriamiento OU3 FICA-UNSL ❄️')
st.write('Esta aplicación calcula la evolución del aire en una torre de enfriamiento y estima sus parámetros de diseño.')

# ==================== DATOS DE EQUILIBRIO (MANTENER FIJOS O PERMITIR SELECCIÓN) ====================
#st.subheader('Datos de la Curva de Equilibrio H*(t)')
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

# ==================== CÁLCULO DEL FLUJO MÍNIMO DE AIRE (CON RESTRICCIÓN FÍSICA) ====================
    #st.subheader('Cálculo del Flujo Mínimo de Aire')

    # Variables de salida inicializadas por seguridad
    t_pinch_global = tini
    H_pinch_global = H_star_func(tini)
    m_max_global = 0.0
    Gs_min = 0.0

    try:
        t_start = tini
        H_start = Hini

        # 1. Definimos el rango de búsqueda matemática (amplio para el optimizador)
        t_rango_check = np.linspace(tini - 5, tfin + 10, 1000)

        def objetivo_tangencia(m):
            h_op = H_start + m * (t_rango_check - t_start)
            h_eq = H_star_func(t_rango_check)
            distancia_minima = np.min(h_eq - h_op)
            return distancia_minima**2

        from scipy.optimize import minimize

        # Pendiente inicial aproximada
        m_guess = (H_star_func(tfin) - H_start) / (tfin - t_start)
        res_m = minimize(objetivo_tangencia, x0=[m_guess], bounds=[(0.01, None)], method='L-BFGS-B')

        m_tangente = res_m.x[0]

        # 2. Identificamos dónde ocurre esa tangencia matemática
        h_op_tangente = H_start + m_tangente * (t_rango_check - t_start)
        h_eq_check = H_star_func(t_rango_check)
        idx_pinch = np.argmin(h_eq_check - h_op_tangente)
        t_pinch_calc = t_rango_check[idx_pinch]

        # 3. LÓGICA DE RESTRICCIÓN DE RANGO (Cabeza de columna)
        # Pendiente límite hacia el equilibrio en la entrada de agua (Punto crítico en T_fin)
        m_tope = (H_star_func(tfin) - H_start) / (tfin - t_start)

        if t_pinch_calc > tfin:
            # Si la tangencia es fuera de rango, el punto crítico es el tope
            m_max_global = m_tope
            t_pinch_global = tfin
            #st.warning("⚠️ Pinch detectado en la cabeza de la columna (T_entrada agua).")
        else:
            # Si la tangencia es interna, es el flujo mínimo teórico estricto
            m_max_global = m_tangente
            t_pinch_global = t_pinch_calc
            #st.success("✅ Tangencia interna detectada (Pinch intermedio).")

        H_pinch_global = H_star_func(t_pinch_global)

        # 4. Cálculos de flujo final
        Gs_min = (L * Cp_default) / m_max_global
        G_min = Gs_min / (1 - y1)

        #col_a, col_b, col_c = st.columns(3)
        #col_a.metric("Pendiente Máx (m)", f"{m_max_global:.3f}")
        #col_b.metric("Temp. Pinch", f"{t_pinch_global:.2f} {temp_unit}")
        #col_c.metric("Gs Mínimo", f"{Gs_min:.1f}")

    except Exception as e:
        st.error(f"Error en la optimización: {e}")
        m_max_global = (H_star_func(tfin) - Hini) / (tfin - tG1)
        Gs_min = (L * Cp_default) / m_max_global

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

# ==================== SECCIÓN DE RESULTADOS UNIFICADA Y COMPACTA ====================
    st.markdown("### 📊 Resultados de la Simulación")
    st.markdown("## Condicionesde operacion")
    st.write(f"**Gs:** {Gs:.2f} {flow_unit}")
    st.write(f"**Ls:** {L:.2f} {flow_unit}")
    # --- PARTE 1: Puntos de Operación ---
    st.markdown("##### 🌡️ Condiciones en los extremos de la torre")
    col_ext1, col_ext2 = st.columns(2)
    with col_ext1:
      # ==================== LÓGICA DE VALIDACIÓN DE SATURACIÓN (CABEZA) ====================
        H_final_cabeza = H_air[-1]
        T_cabeza_mickley = t_air[-1]

        # Comparamos la entalpía final con la entalpía de saturación a la temperatura de Mickley
        if H_final_cabeza > H_star_func(T_cabeza_mickley):
            # El aire se saturó: buscamos la Temperatura donde H* = H_final_cabeza
            T_cabeza_final = fsolve(lambda t: H_star_func(t) - H_final_cabeza, T_cabeza_mickley)[0]
            Y_cabeza_final = calcular_Y(H_final_cabeza, T_cabeza_final, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)
            status_sat = " ⚠️ (Saturado)"
        else:
            # El aire no llegó a saturarse: usamos los valores directos de Mickley
            T_cabeza_final = T_cabeza_mickley
            Y_cabeza_final = Y_air[-1]
            status_sat = ""

        # ==================== REPORTE DE RESULTADOS (CABEZA) ====================
        st.markdown("**Cabeza**")
        st.write(f"🌡️ **Temperatura del agua:** {tfin:.2f} {temp_unit}")
        st.write(f"🌡️ **Temperatura del aire:** {T_cabeza_final:.2f} {temp_unit}{status_sat}")
        st.write(f"💧 **Humedad del aire:** {Y_cabeza_final:.5f} {Y_unit}")
        st.write(f"🔥 **Entalpía del aire:** {H_final_cabeza:.2f} {enthalpy_unit}")
    with col_ext2:
        st.markdown("**Base**")
        st.write(f"🌡️ **Temperatura del agua:** {tini:.2f} {temp_unit}")
        st.write(f"🌡️ **Temperatura del aire:** {tG1:.2f} {temp_unit}")
        st.write(f"💧 **Humedad del aire:** {Y1:.5f} {Y_unit}")
        st.write(f"🔥 **Entalpía del aire:** {Hini:.2f} {enthalpy_unit}")

    st.markdown("---")

    # --- PARTE 2: Análisis de Flujo Crítico y Dimensionamiento ---
    # Combinamos Pinch y Diseño en una misma estructura de columnas para uniformidad
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.markdown("##### Flujo mínimo de aire")
        st.write(f"📉**Pendiente Máxima:** {m_max_global:.3f}")
        #st.write(f"📍 **Temp. Pinch:** {t_pinch_global:.2f} {temp_unit}")
        st.write(f"🌬️**Gs Mínimo:** {Gs_min:.1f}")
        #estado_txt = "Interno" if t_pinch_global < tfin else "En Cabeza"
        #st.write(f"📌 **Tipo de Pinch:** {estado_txt}")

    with col_res2:
        st.markdown("##### Dimensionamiento del Relleno")
        st.write(f"🔢**HtoG:** {HtoG:.2f} {length_unit}")
        st.write(f"🔢**NtoG:** {NtoG:.2f}")
        st.write(f"📏**Altura del relleno (Z):** {Z_total:.2f} {length_unit}")
        porcentaje_evap = (Lrep/L)*100

    st.write(f"💧 **Agua de reposición (Lrep):** {Lrep:.2f} {flow_unit} ({porcentaje_evap:.2f}%)")

    st.markdown("---")
    # ==================== GRÁFICO FINAL ====================
    st.subheader('Diagrama de Entalpía-Temperatura')

    fig, ax = plt.subplots(figsize=(10, 7))

    T_plot = np.linspace(min(teq), max(teq) + 10, 200)
    ax.plot(T_plot, H_star_func(T_plot), label=f'Curva de equilibrio H*({temp_unit})', linewidth=2, color='blue')
    ax.plot([tini, tfin], [Hini, Hfin], 'r-', label=f'Línea de operación Hop({temp_unit})', linewidth=2)
    ax.plot(t_air, H_air, 'ko-', label=f'Curva de evolución del aire H({temp_unit})', markersize=4, linewidth=1)

    # Línea tangente del pinch (RECTA ROJA)
    Hfin_min = Hini + m_max_global * (tfin - tG1)
    ax.plot([tini, t_pinch_global], 
            [Hini, H_pinch_global], 
            'r--', linewidth=3, label='Recta tangente (Gs_min)', alpha=0.8)
    ax.plot(t_pinch_global, H_pinch_global, 'ro', markersize=12, label=f'Pinch ({t_pinch_global:.1f}{temp_unit})')

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

    # ==================== SECCIÓN DE FUNDAMENTOS Y METODOLOGÍA ====================

with st.expander("📚 Ver mas información"):

    st.markdown("### 📋 Condiciones y restricciones del modelo")
    st.info("""
    1. **Estado Estacionario** 
    2. **Operación Adiabática** 
    3. **Resistencia Controlante en la fase gas** 
    4. **L/G Constante** 
    5. **Calor Específico del agua ($C_{pw}$) constante** 
    6. **Equilibrio en la interfase** 
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Metodología de Cálculo")

    st.markdown("#### 1. Flujo Mínimo de Aire ($G_{s,min}$)")
    st.write("""
    Se determina mediante la **Pendiente Máxima ($m_{max}$)** de la Línea de Operación. 
    El algoritmo busca la tangencia entre la recta que nace en $(T_{w,out}, H_{in})$ y la curva de equilibrio.
    - Si la tangencia es interna, se identifica el **Punto de Pinch**.
    - Si no hay tangencia interna, el límite se establece en la cabeza de la torre ($T_{w,in}$).
    """)

    st.markdown("#### 2. Evolución del Aire (Método de Mickley)")
    st.write("""
    Se calcula paso a paso la evolución de la entalpía ($H$) y temperatura del aire ($T_G$) resolviendo la relación:
    """)
    st.latex(r"\frac{dH}{dT_G} = \frac{H^* - H}{T_w - T_G}")
    st.write("Esto permite obtener la **Humedad Absoluta de salida ($Y_2$)** y la entalpía final.")

    st.markdown("#### 3. Altura del relleno Z")
    st.write("""
    **Número de Unidades de Transferencia ($N_{toG}$):** 
    """)
    st.latex(r"N_{toG} = \int_{H_{in}}^{H_{out}} \frac{dH}{H^* - H}")

    st.write("""
    **Altura de la Unidad de Transferencia ($H_{toG}$):** 
    """)
    st.latex(r"H_{toG} = \frac{G_s}{K_y a}")

    st.write("""
    **Altura del relleno ($Z$):** Resultado final del diseño.
    """)
    st.latex(r"Z = H_{toG} \times N_{toG}")

    st.markdown("#### 4. Agua de Reposición")
    st.write("Se calcula a partir de la diferencia de humedades absolutas entre la entrada y la salida:")
    st.latex(r"L_{rep} = G_s \cdot (Y_2 - Y_1)")
    st.markdown("---")
    st.markdown("### 📚 Bibliografía y recursos")

    st.markdown("El desarrollo del simulador se realizó en lenguaje Python 3.11 (Van Rossum & Drake, 2025), utilizando la librería Streamlit para la interfaz de usuario. El procesamiento numérico y la resolución de las ecuaciones de balance de entalpía se apoyaron en las librerías NumPy y SciPy, utilizando específicamente algoritmos de resolución no lineal (fsolve) e interpolación spline para la modelización de las curvas de equilibrio psicrométrico.")

    st.markdown("""
    * Treybal, R. E. (1980).Mass-Transfer Operations (3rd ed.). McGraw-Hill Education. 
    * Foust, A. S., Wenzel, L. A., Clump, C. W., Maus, L., & Andersen, L. B. (1980).Principles of Unit Operations (2nd ed.). John Wiley & Sons.
    * Streamlit Inc. (2025). Streamlit (Version 1.x) [Software]. https://streamlit.io
    * Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). https://doi.org/10.1038/s41586-020-2649-2
    """)

st.markdown("### 🎓 ")
st.write("**Asignatura:** Operaciones Unitarias 3 - Ingeniería Química")
st.write("**Institución:** Facultad de Ingeniería y Ciencias Agropecuarias (FICA) - Universidad Nacional de San Luis (UNSL).")
st.write("**Cita sugerida (APA):**")
st.markdown("Miró Erdmann, S. M. (2025). Simulador de Torres de Enfriamiento(v1.0) [Software]. Villa Mercedes, San Luis: FICA-UNSL._")
st.write("Este software es un recurso de acceso abierto para fines académicos y de investigación en el marco de la Universidad Nacional de San Luis.")
st.caption("Final del reporte de simulación - 2025")

# Línea final fuera del bloque para cerrar la interfaz
st.markdown("---")
