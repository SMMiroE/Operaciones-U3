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
    flow_unit = "lb/(tiempo área)"
    length_unit = "ft"
    h_temp_ref = 32
    h_latent_ref = 1075.8
    h_cp_air_dry = 0.24
    h_cp_vapor = 0.45
else: # Sistema Internacional
    teq = np.array([0, 10, 20, 30, 40, 50, 60])  # °C
    Heq_data = np.array([9479, 29360, 57570, 100030, 166790, 275580, 461500])  # J/kg aire seco
    Cp_default = 4186       # calor específico del agua, J/(kg °C)
    temp_unit = "°C"
    enthalpy_unit = "J/kg aire seco"
    flow_unit = "kg/(tiempo área)"
    length_unit = "m"
    h_temp_ref = 0 # Referencia para °C
    h_latent_ref = 2501e3 # A 0°C, J/kg
    h_cp_air_dry = 1005 # J/kg°C
    h_cp_vapor = 1880 # J/kg°C (puede variar un poco)

# Función para calcular entalpía del aire húmedo (adaptada para ambos sistemas)
def calcular_entalpia_aire(t, Y, temp_ref, latent_ref, cp_air_dry, cp_vapor):
    return (cp_air_dry + cp_vapor * Y) * (t - temp_ref) + latent_ref * Y

# Función para calcular Y (humedad absoluta) (adaptada para ambos sistemas)
def calcular_Y(H, t, temp_ref, latent_ref, cp_air_dry, cp_vapor):
    return (H - cp_air_dry * (t - temp_ref)) / (cp_vapor * (t - temp_ref) + latent_ref)

# ==================== ENTRADA DE DATOS DEL PROBLEMA ====================
st.sidebar.header('Parámetros del Problema')

# Uso de st.number_input para permitir al usuario ingresar los valores
L = st.sidebar.number_input(f'Flujo de agua (L, {flow_unit})', value=2200.0, format="%.2f")
G = st.sidebar.number_input(f'Flujo de aire (G, {flow_unit})', value=2000.0, format="%.2f")
tfin = st.sidebar.number_input(f'Temperatura de entrada del agua (tfin, {temp_unit})', value=105.0, format="%.2f")
tini = st.sidebar.number_input(f'Temperatura de salida del agua (tini, {temp_unit})', value=85.0, format="%.2f")
tG1 = st.sidebar.number_input(f'Bulbo seco del aire a la entrada (tG1, {temp_unit})', value=90.0, format="%.2f")
tw1 = st.sidebar.number_input(f'Bulbo húmedo del aire a la entrada (tw1, {temp_unit})', value=76.0, format="%.2f")
Y1 = st.sidebar.number_input('Humedad absoluta del aire a la entrada (Y1, kg vapor/kg aire seco)', value=0.016, format="%.5f")
P = st.sidebar.number_input('Presión de operación (P, atm)', value=1.0, format="%.2f")
KYa = st.sidebar.number_input('Coef. volumétrico de transferencia de materia (KYa)', value=850.0, format="%.2f")
Cp = st.sidebar.number_input(f'Calor específico del agua (Cp, {Cp_default} por defecto)', value=Cp_default, format="%.2f")

# Botón para ejecutar el cálculo (opcional, Streamlit recalcula en cada cambio de input)
# He comentado esta sección ya que Streamlit recalcula automáticamente con los cambios de input
# if st.sidebar.button('Calcular'):
#     st.write("Calculando...")
# else:
#     st.write("Ajusta los parámetros y observa los resultados.")

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
        # No detendremos la ejecución aquí, solo mostraremos una advertencia.
        # if Hfin <= Hini: # Esta condición puede ser redundante si tini >= tfin y L, Cp, Gs son positivos
        #     st.warning("Advertencia: La entalpía final del aire no es mayor que la inicial. Posible problema en los datos de entrada o no hay enfriamiento.")
        #     st.stop()


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
        
    # Asegurarse de que el último punto de la línea de operación y la curva de evolución del aire coincida con Hfin
    # Este bloque ya debería estar cubierto por la lógica del bucle 'while True'
    # si la condición H_next >= Hfin se maneja correctamente dentro del bucle.
    # Lo dejo comentado para evitar duplicidad o lógica confusa.
    # if H_air[-1] < Hfin:
    #     H_air.append(Hfin)
    #     t_op_final = tfin
    #     if len(H_air) > 1 and len(t_air) > 1 and len(t_op) > 1 and len(H_star) > 1:
    #         H_prev_evo = H_air[-2]
    #         t_prev_evo = t_air[-2]
    #         t_op_prev_evo = t_op[-2]
    #         H_star_prev_evo = H_star[-2]
    #         DH_last = Hfin - H_prev_evo
    #         if abs(H_star_prev_evo - H_prev_evo) < 1e-6:
    #             t_air_final = t_prev_evo
    #         else:
    #             t_air_final = DH_last * ((t_op_prev_evo - t_prev_evo) / (H_star_prev_evo - H_prev_evo)) + t_prev_evo
    #     else:
    #         t_air_final = tG1
            
    #     t_air.append(t_air_final)
    #     Y_air.append(calcular_Y(Hfin, t_air_final, h_temp_ref, h_latent_ref, h_cp_air_dry, h_cp_vapor))
    #     t_op.append(t_op_final)
    #     H_op.append(Hfin)
    #     H_star.append(H_star_func(t_op_final))
            
    #     if len(t_air) >= 2:
    #         last_t_air = t_air[-1]
    #         last_H_air = H_air[-1]
    #         last_t_op = t_op[-1]
    #         last_H_star = H_star_func(last_t_op)
    #         segmentos.append(((last_t_air, last_H_air), (last_t_op, last_H_air)))
    #         segmentos.append(((last_t_op, last_H_air), (last_t_op, last_H_star)))
    #         segmentos.append(((last_t_op, last_H_star), (last_t_air, last_H_air)))

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
