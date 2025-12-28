# ==================== IMPORTACIÓN DE LIBRERÍAS ====================
import streamlit as st  # Importa la librería Streamlit
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splev, splrep  # Import splev and splrep
from scipy.optimize import fsolve  # Para resolver numéricamente el punto de pellizco

# ==================== CONFIGURACIÓN DE LA PÁGINA (OPCIONAL) ====================
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

Lrep = Gs * (Y_air[-1] - Y1)

# ==================== SECCIÓN DE RESULTADOS UNIFICADA Y COMPACTA ====================
    st.markdown("### 📊 Resultados de la Simulación")

    # --- PARTE 1: Puntos de Operación ---
    st.markdown("##### 🌡️ Condiciones en los extremos de la torre")

        st.write(f"🔥 **Entalpía del aire:** {H_air[-1]:.2f} {enthalpy_unit}")

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
        st.write(f"🌬️**Gs Mínimo:** {Gs_min:.1f} kg/h·m²")
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
