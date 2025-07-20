# ==================== IMPORTACIÓN DE LIBRERÍAS ====================
import streamlit as st # Importa la librería Streamlit
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==================== CONFIGURACIÓN DE LA PÁGINA (OPCIONAL) ====================
st.set_page_config(
    page_title="Método de Mickley - Torres de Enfriamiento",
    layout="centered", # o "wide" para más espacio
    initial_sidebar_state="auto"
)

# ==================== TÍTULO DE LA APLICACIÓN ====================
st.title('🌡️ Simulación de Torres de Enfriamiento - Método de Mickley ❄️')
st.write('Esta aplicación calcula la evolución del aire en una torre de enfriamiento y determina sus parámetros de diseño.')

# ==================== DATOS DE EQUILIBRIO (MANTENER FIJOS O PERMITIR SELECCIÓN) ====================
# Estos datos suelen ser fijos por la naturaleza del método
st.subheader('Datos de la Curva de Equilibrio H*(t)')
opcion_unidades = st.radio(
    "Seleccione el sistema de unidades:",
    ('Sistema Inglés', 'Sistema Internacional')
)

if opcion_unidades == 'Sistema Inglés':
    teq = np.array([32, 40, 60, 80, 100, 120, 140]) # °F
    Heq_data = np.array([4.074, 7.545, 18.780, 36.020, 64.090, 112.0, 198.0]) # BTU/lb aire seco
    Cp_default = 1.0    # calor específico del agua, Btu/(lb °F)
    temp_unit = "°F"
    enthalpy_unit = "BTU/lb aire seco"
    flow_unit = "lb/(h ft²)" # Especificación de unidades de flujo de agua y aire
    length_unit = "ft"
    h_temp_ref = 32
    h_latent_ref = 1075.8
    h_cp_air_dry = 0.24
    h_cp_vapor = 0.45
    kya_unit = "lb/(h ft² DY)" # Especificación de unidades de KYa
    cp_unit = "BTU/(lb agua °F)" # Especificación de unidades de Cp
    Y_unit = "lb agua/lb aire seco" # Especificación de unidades de Y
else: # Sistema Internacional
    teq = np.array([0, 10, 20, 30, 40, 50, 60])  # °C
    Heq_data = np.array([9479, 29360, 57570, 100030, 166790, 275580, 461500])  # J/kg aire seco
    Cp_default = 4186       # calor específico del agua, J/(kg °C)
    temp_unit = "°C"
    enthalpy_unit = "J/kg aire seco" # Especificado "aire seco"
    flow_unit = "kg/(s m²)" # Especificación de unidades de flujo de agua y aire
    length_unit = "m"
    h_temp_ref = 0 # Referencia para °C
    h_latent_ref = 2501e3 # A 0°C, J/kg
    h_cp_air_dry = 1005 # J/kg°C
    h_cp_vapor = 1880 # J/kg°C (puede variar un poco)
    kya_unit = "kg/(s m² DY)" # Especificación de unidades de KYa
    cp_unit = "J/(kg agua °C)" # Especificación de unidades de Cp
    Y_unit = "kg agua/kg aire seco" # Especificación de unidades de Y

# Función para calcular entalpía del aire húmedo (adaptada para ambos sistemas)
def calcular_entalpia_aire(t, Y, temp_ref, latent_ref, cp_air_dry, cp_vapor):
    return (cp_air_dry + cp_vapor * Y) * (t - temp_ref) + latent_ref * Y

# Función para calcular Y (humedad absoluta) (adaptada para ambos sistemas)
def calcular_Y(H, t, temp_ref, latent_ref, cp_air_dry, cp_vapor):
    return (H - cp_air_dry * (t - temp_ref)) / (cp_vapor * (t - temp_ref) + latent_ref)

# Función para calcular Y1 a partir de bulbo seco y bulbo húmedo
def calculate_Y_from_wet_bulb(t_dry_bulb, t_wet_bulb, total_pressure_atm, units_system):
    """
    Calcula la humedad absoluta (Y) a partir de la temperatura de bulbo seco,
    temperatura de bulbo húmedo y presión total, utilizando correlaciones psicrométricas.
    """
    if units_system == 'Sistema Internacional':
        # Convertir presión total de atm a kPa
        P_kPa = total_pressure_atm * 101.325

        # Presión de vapor de saturación a la temperatura de bulbo húmedo (en kPa)
        # Utilizando la fórmula de Magnus (aproximación para t en °C)
        P_ws_tw_kPa = 0.6108 * np.exp((17.27 * t_wet_bulb) / (t_wet_bulb + 237.3))

        # Constante psicrométrica (aproximación para presión estándar, en kPa^-1)
        psychrometric_constant = 0.000662 # kPa^-1

        # Presión de vapor (Pv)
        Pv_kPa = P_ws_tw_kPa - psychrometric_constant * P_kPa * (t_dry_bulb - t_wet_bulb)

        # Asegurar que Pv no sea negativo
        if Pv_kPa < 0:
            Pv_kPa = 0

        # Humedad absoluta (Y)
        if (P_kPa - Pv_kPa) <= 0:
            return float('inf') # Retornar infinito para indicar un estado de saturación/error
        Y = 0.62198 * (Pv_kPa / (P_kPa - Pv_kPa))
        return Y
    else: # Sistema Inglés
        # Convertir presión total de atm a psi
        P_psi = total_pressure_atm * 14.696

        # Presión de vapor de saturación a la temperatura de bulbo húmedo (en psi)
        # Aproximación para t en °F
        P_ws_tw_psi = 0.1805 * np.exp((17.27 * (t_wet_bulb - 32)) / (t_wet_bulb - 32 + 427.3))

        # Constante psicrométrica (aproximación para presión estándar, en psi^-1)
        psychrometric_constant = 0.000367 # psi^-1

        # Presión de vapor (Pv)
        Pv_psi = P_ws_tw_psi - psychrometric_constant * P_psi * (t_dry_bulb - t_wet_bulb)

        # Asegurar que Pv no sea negativo
        if Pv_psi < 0:
            Pv_psi = 0

        # Humedad absoluta (Y)
        if (P_psi - Pv_psi) <= 0:
            return float('inf')
        Y = 0.62198 * (Pv_psi / (P_psi - Pv_psi))
        return Y

# Nueva función para calcular Y1 a partir de bulbo seco y humedad relativa
def calculate_Y_from_relative_humidity(t_dry_bulb, relative_humidity_percent, total_pressure_atm, units_system):
    """
    Calcula la humedad absoluta (Y) a partir de la temperatura de bulbo seco,
    humedad relativa y presión total.
    """
    if units_system == 'Sistema Internacional':
        P_kPa = total_pressure_atm * 101.325
        # Presión de vapor de saturación a la temperatura de bulbo seco (en kPa)
        P_ws_tdb_kPa = 0.6108 * np.exp((17.27 * t_dry_bulb) / (t_dry_bulb + 237.3))
        
        # Presión de vapor (Pv)
        Pv_kPa = (relative_humidity_percent / 100.0) * P_ws_tdb_kPa

        if (P_kPa - Pv_kPa) <= 0:
            return float('inf') # Indica saturación o error
        Y = 0.62198 * (Pv_kPa / (P_kPa - Pv_kPa))
        return Y
    else: # Sistema Inglés
        P_psi = total_pressure_atm * 14.696
        # Presión de vapor de saturación a la temperatura de bulbo seco (en psi)
        P_ws_tdb_psi = 0.1805 * np.exp((17.27 * (t_dry_bulb - 32)) / (t_dry_bulb - 32 + 427.3))

        # Presión de vapor (Pv)
        Pv_psi = (relative_humidity_percent / 100.0) * P_ws_tdb_psi

        if (P_psi - Pv_psi) <= 0:
            return float('inf')
        Y = 0.62198 * (Pv_psi / (P_psi - Pv_psi))
        return Y


# ==================== ENTRADA DE DATOS DEL PROBLEMA ====================
st.sidebar.header('Parámetros del Problema')

# Uso de st.number_input para permitir al usuario ingresar los valores
L = st.sidebar.number_input(f'Flujo de agua (L, {flow_unit})', value=2200.0, format="%.2f")
G = st.sidebar.number_input(f'Flujo de aire (G, {flow_unit})', value=2000.0, format="%.2f")
tfin = st.sidebar.number_input(f'Temperatura de entrada del agua (tfin, {temp_unit})', value=105.0, format="%.2f")
tini = st.sidebar.number_input(f'Temperatura de salida del agua (tini, {temp_unit})', value=85.0, format="%.2f")

# Añadir la opción para la fuente de Y1
Y1_source_option = st.sidebar.radio(
    "Fuente de Humedad Absoluta (Y1):",
    ('Ingresar Y1 directamente', 'Calcular Y1 a partir de Bulbo Húmedo', 'Calcular Y1 a partir de Humedad Relativa')
)

Y1 = 0.016 # Valor por defecto inicial

if Y1_source_option == 'Ingresar Y1 directamente':
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    # tw1 no es necesario si se ingresa Y1 directamente, pero se mantiene para consistencia en el flujo de datos
    tw1 = st.sidebar.number_input(f'Bulbo húmedo del aire a la entrada (tw1, {temp_unit})', value=76.0, format="%.2f")
    Y1 = st.sidebar.number_input(f'Humedad absoluta del aire a la entrada (Y1, {Y_unit})', value=0.016, format="%.5f")
elif Y1_source_option == 'Calcular Y1 a partir de Bulbo Húmedo':
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    tw1 = st.sidebar.number_input(f'Bulbo húmedo del aire a la entrada (tw1, {temp_unit})', value=76.0, format="%.2f")
    P = st.sidebar.number_input('Presión de operación (P, atm)', value=1.0, format="%.2f") # P es necesario para el cálculo
    st.sidebar.write("Calculando Y1 a partir de Bulbo Húmedo:")
    try:
        calculated_Y1 = calculate_Y_from_wet_bulb(tG1, tw1, P, opcion_unidades)
        if calculated_Y1 == float('inf'):
            st.sidebar.error("Error al calcular Y1: Posible saturación o datos inconsistentes. Ajuste las temperaturas de bulbo seco y húmedo.")
            Y1 = 0.016 # Valor de respaldo en caso de error
        else:
            Y1 = calculated_Y1
            st.sidebar.info(f"Y1 calculado: **{Y1:.5f}** ({Y_unit})")
    except Exception as e:
        st.sidebar.error(f"Error en el cálculo de Y1: {e}. Usando valor por defecto.")
        Y1 = 0.016 # Valor de respaldo en caso de error
elif Y1_source_option == 'Calcular Y1 a partir de Humedad Relativa':
    tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
    relative_humidity = st.sidebar.number_input('Humedad Relativa a la entrada (HR, %)', value=50.0, min_value=0.0, max_value=100.0, format="%.1f")
    P = st.sidebar.number_input('Presión de operación (P, atm)', value=1.0, format="%.2f") # P es necesario para el cálculo
    # tw1 no es necesario para este cálculo, pero se puede mantener para evitar errores si se usa en otro lugar
    tw1 = 0.0 # Valor por defecto, no se usa en este cálculo
    st.sidebar.write("Calculando Y1 a partir de Humedad Relativa:")
    try:
        calculated_Y1 = calculate_Y_from_relative_humidity(tG1, relative_humidity, P, opcion_unidades)
        if calculated_Y1 == float('inf'):
            st.sidebar.error("Error al calcular Y1: Posible saturación o datos inconsistentes. Ajuste la temperatura de bulbo seco y la humedad relativa.")
            Y1 = 0.016 # Valor de respaldo en caso de error
        else:
            Y1 = calculated_Y1
            st.sidebar.info(f"Y1 calculado: **{Y1:.5f}** ({Y_unit})")
    except Exception as e:
        st.sidebar.error(f"Error en el cálculo de Y1: {e}. Usando valor por defecto.")
        Y1 = 0.016 # Valor de respaldo en caso de error

# La presión P y KYa y Cp se muestran aquí, fuera de los bloques condicionales de Y1
# para asegurar que siempre estén presentes y no se dupliquen si se usan en múltiples ramas.
# Si P ya se definió en un bloque anterior, Streamlit lo manejará correctamente.
if Y1_source_option != 'Calcular Y1 a partir de Bulbo Húmedo' and Y1_source_option != 'Calcular Y1 a partir de Humedad Relativa':
    P = st.sidebar.number_input('Presión de operación (P, atm)', value=1.0, format="%.2f") # Solo mostrar si no se calculó Y1

KYa = st.sidebar.number_input(f'Coef. volumétrico de transferencia de materia (KYa, {kya_unit})', value=850.0, format="%.2f")
Cp = st.sidebar.number_input(f'Calor específico del agua (Cp, {cp_unit})', value=Cp_default, format="%.2f")

# ==================== CÁLCULOS BASE ====================
try:
    y1 = Y1 / (1 + Y1)
    Gs = G * (1 - y1)
    
    # Usar la función de entalpía adaptada
    Hini = calcular_entalpia_aire(tG1, Y1, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)
    
    # Evitar división por cero si Gs es 0
    if Gs == 0:
        st.error("Error: El flujo de aire seco (Gs) no puede ser cero. Revise el flujo de aire (G) y la humedad (Y1).")
        st.stop()
        
    Hfin = (L * Cp / Gs) * (tfin - tini) + Hini

    # Validaciones iniciales
    if tini >= tfin:
        st.warning("Advertencia: La temperatura de salida del agua (tini) debe ser menor que la de entrada (tfin) para un enfriamiento.")


    # ==================== Polinomio H*(t) ====================
    H_star_func = interp1d(teq, Heq_data, kind='cubic', fill_value='extrapolate')

    # ==================== MÉTODO DE MICKLEY ======================
    DH = (Hfin - Hini) / 20
    
    # Manejo de la dirección de la entalpía para evitar bucles infinitos en casos atípicos
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
    
    # Bucle con un contador de seguridad para evitar bucles infinitos
    max_iterations = 1000 # Límite de iteraciones
    i_loop = 0

    while True:
        i_loop += 1
        if i_loop > max_iterations:
            st.warning(f"Advertencia: Bucle de Mickley excedió {max_iterations} iteraciones. Revisar datos de entrada o divergencia.")
            break

        H_next = H_air[-1] + DH
        
        # Si H_next ya supera Hfin, terminar y ajustar el último punto si es necesario
        if H_next >= Hfin: # Usar >= para incluir el punto final
            H_next = Hfin # Asegurar que el último punto no exceda Hfin
            
            # Recalcular t_op_next para este Hfin
            t_op_next = (H_next - Hini) * (tfin - tini) / (Hfin - Hini) + tini
            H_star_next = H_star_func(t_op_next)
            
            # Recalcular t_next con el último punto válido antes de Hfin para la pendiente
            if len(H_air) > 1 and len(t_air) > 1 and len(t_op) > 1 and len(H_star) > 1:
                t_prev = t_air[-1]
                H_prev = H_air[-1]
                t_op_prev = t_op[-1]
                H_star_prev = H_star[-1]
                
                # Usar el DH real para el último paso
                DH_last_step = H_next - H_prev
                
                # Evitar división por cero
                if abs(H_star_prev - H_prev) < 1e-6: # Muy cerca del equilibrio
                    t_next = t_prev # o alguna aproximación para evitar infinito
                else:
                    t_next = DH_last_step * ((t_op_prev - t_prev) / (H_star_prev - H_prev)) + t_prev
            else: # Para el primer paso si H_next ya es > Hfin (caso raro)
                t_next = tG1 # O un valor por defecto
            
            H_star_tnext = H_star_func(t_next)
            Y_next = calcular_Y(H_next, t_next, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)

            H_air.append(H_next)
            t_air.append(t_next)
            Y_air.append(Y_next)
            t_op.append(t_op_next)
            H_op.append(H_next)
            H_star.append(H_star_next)
            break # Romper después de añadir el punto final
            
        t_op_next = (H_next - Hini) * (tfin - tini) / (Hfin - Hini) + tini
        H_star_next = H_star_func(t_op_next)

        # Evitar división por cero si estamos muy cerca del equilibrio (H_star - H_air)
        if abs(H_star[-1] - H_air[-1]) < 1e-6:
            t_next = t_air[-1] # No hay cambio significativo en la temperatura del aire
        else:
            t_next = DH * ((t_op[-1] - t_air[-1]) / (H_star[-1] - H_air[-1])) + t_air[-1]
            
        H_star_tnext = H_star_func(t_next)
        Y_next = calcular_Y(H_next, t_next, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor)

        # Condición de detención original (se puede refinar)
        # Se ha ajustado la condición de H_next - Hfin para evitar un bucle extra si ya pasó el Hfin
        if H_next > Hfin or (H_next - H_star_tnext) > 0: # La segunda condición indica que la línea de operación cruza el equilibrio
            # Si H_next ya superó Hfin, o si la fuerza impulsora se invierte, terminamos.
            # Aseguramos que el último punto sea Hfin si se superó.
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
                    t_next = tG1 # Fallback
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

        # Segmentos para el gráfico
        segmentos.append(((t_next, H_next), (t_op_next, H_next)))
        segmentos.append(((t_op_next, H_next), (t_op_next, H_star_next)))
        segmentos.append(((t_op_next, H_star_next), (t_next, H_next)))
    
    # Manejar el caso donde no se generaron suficientes puntos para la evolución del aire
    if len(H_air) <= 1:
        st.warning("No se pudo generar la curva de evolución del aire. Revise las temperaturas y flujos de entrada.")
        st.stop()
        
    # ==================== CÁLCULO DE NtoG ====================
    n_pasos_integracion = 100 # Aumentar pasos para mejor precisión en la integración
    dt_integracion = (tfin - tini) / n_pasos_integracion
    t_water_integracion = np.linspace(tini, tfin, n_pasos_integracion + 1)
    
    # Calcular Hop(t) para los puntos de integración
    H_op_vals_integracion = np.interp(t_water_integracion, [tini, tfin], [Hini, Hfin])
    H_star_vals_integracion = H_star_func(t_water_integracion)

    f_T_integracion = []
    for i in range(len(t_water_integracion)):
        delta = H_star_vals_integracion[i] - H_op_vals_integracion[i]
        # Manejar el caso de delta muy pequeño (cercano a 0) que indica pinch point
        if abs(delta) < 1e-6:
            st.error(f"Error: La línea de operación está muy cerca o cruza la curva de equilibrio en t={t_water_integracion[i]:.2f}. Verifique los datos de entrada o la viabilidad del diseño. No se puede calcular NtoG.")
            st.stop() # Detiene la ejecución de Streamlit
        f_T_integracion.append(1 / delta)

    dHdT_integracion = (Hfin - Hini) / (tfin - tini) # Pendiente de la línea de operación
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
    st.subheader('Resultados del Cálculo')
    st.info(f"**Línea de operación:**")
    st.write(f"  - Cabeza de la torre (entrada de agua): (t = {tfin:.2f} {temp_unit}, H = {Hfin:.2f} {enthalpy_unit})")
    st.write(f"  - Base de la torre (salida de agua): (t = {tini:.2f} {temp_unit}, H = {Hini:.2f} {enthalpy_unit})")
    st.info(f"**Parámetros de Diseño:**")
    st.write(f"  - Humedad absoluta del aire a la salida: **Y = {Y_air[-1]:.5f}** (masa vapor de agua/masa de aire seco)")
    st.write(f"  - Agua evaporada (reposición): **Lrep = {Lrep:.2f} {flow_unit}**")
    st.write(f"  - Número de unidades de transferencia (NtoG): **{NtoG:.2f}**")
    st.write(f"  - Altura de unidad de transferencia (HtoG): **{HtoG:.2f} {length_unit}**")
    st.write(f"  - Altura total del relleno (Z): **{Z_total:.2f} {length_unit}**")

    # ==================== GRÁFICO FINAL ====================
    st.subheader('Diagrama de Entalpía-Temperatura')

    fig, ax = plt.subplots(figsize=(10, 7)) # Crear la figura y el eje

    T_plot = np.linspace(min(teq), max(teq) + 10, 200) # Extender un poco el rango para la curva
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
    ax.set_xlim(min(tini, tG1) - 10, max(tfin, max(t_air)) + 10) # Ajuste automático de límites
    ax.set_ylim(min(Hini, min(Heq_data)) - 10, max(Hfin, max(Heq_data)) + 30)
    ax.set_autoscale_on(True) # Asegurarse que el autoescalado es posible

    st.pyplot(fig) # Muestra el gráfico en Streamlit

except Exception as e:
    st.error(f"Ha ocurrido un error en los cálculos. Por favor, revise los datos de entrada. Detalle del error: {e}")
