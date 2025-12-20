# ==================== IMPORTACIÓN DE LIBRERÍAS ====================
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splev, splrep

# ==================== CONSTANTES GLOBALES ====================
EPSILON = 0.62198   # relación masas agua/aire seco
TOL = 1e-6          # tolerancia numérica

# ==================== CONFIGURACIÓN DE LA PÁGINA ====================
st.set_page_config(
    page_title="Método de Mickley - Torres de Enfriamiento",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title('🌡️ Simulación de Torres de Enfriamiento - Método de Mickley ❄️')
st.write(
    'Esta aplicación calcula la evolución del aire en una torre de enfriamiento y '
    'determina sus parámetros de diseño mediante el método de Mickley.'
)

# ==================== FUNCIONES AUXILIARES ====================

def get_units_config(units_option: str):
    """Devuelve diccionario con datos y propiedades para SI o Inglés."""
    if units_option == 'Sistema Inglés':
        return dict(
            teq=np.array([32, 40, 60, 80, 100, 120, 140]),
            Heq_data=np.array([4.074, 7.545, 18.780, 36.020, 64.090, 112.0, 198.0]),
            Cp_water=1.0,
            temp_unit="°F",
            enthalpy_unit="BTU/lb aire seco",
            flow_unit="lb/(h ft²)",
            length_unit="ft",
            h_temp_ref=32.0,
            h_latent_ref=1075.8,
            h_cp_air_dry=0.24,
            h_cp_vapor=0.45,
            kya_unit="lb/(h ft² DY)",
            cp_unit="BTU/(lb agua °F)",
            Y_unit="lb agua/lb aire seco",
            psychrometric_constant=0.000367,  # psi^-1
            pressure_factor=14.696,          # atm → psi
            psat_units='psi'
        )
    # Sistema Internacional
    return dict(
        teq=np.array([0, 10, 20, 30, 40, 50, 60]),
        Heq_data=np.array([9479, 29360, 57570, 100030, 166790, 275580, 461500]),
        Cp_water=4186.0,
        temp_unit="°C",
        enthalpy_unit="J/kg aire seco",
        flow_unit="kg/(s m²)",
        length_unit="m",
        h_temp_ref=0.0,
        h_latent_ref=2501e3,
        h_cp_air_dry=1005.0,
        h_cp_vapor=1880.0,
        kya_unit="kg/(s m² DY)",
        cp_unit="J/(kg agua °C)",
        Y_unit="kg agua/kg aire seco",
        psychrometric_constant=0.000662,    # kPa^-1
        pressure_factor=101.325,           # atm → kPa
        psat_units='kPa'
    )


def enthalpy_moist_air(T, Y, props):
    """Entalpía del aire húmedo (H) en función de T y Y."""
    dT = T - props["h_temp_ref"]
    return (props["h_cp_air_dry"] + props["h_cp_vapor"] * Y) * dT + props["h_latent_ref"] * Y


def humidity_ratio_from_H_t(H, T, props):
    """Humedad absoluta Y a partir de entalpía H y temperatura T."""
    dT = T - props["h_temp_ref"]
    denom = (props["h_cp_vapor"] * dT + props["h_latent_ref"])
    if abs(denom) < TOL:
        raise ValueError("Denominador ~0 al calcular Y(H,T). Verificar rango de temperaturas.")
    return (H - props["h_cp_air_dry"] * dT) / denom


def sat_vapor_pressure_magnus(T, units_psat):
    """
    Presión de vapor de saturación usando Magnus.
    Si units_psat = 'kPa', T debe estar en °C, P_ws en kPa.
    Si units_psat = 'psi', T se ingresa en °F y se convierte internamente.
    """
    if units_psat == 'kPa':
        # T en °C
        return 0.61094 * np.exp((17.625 * T) / (T + 243.04))
    # Sistema Inglés: T en °F → convertir a °C
    T_c = (T - 32.0) * 5.0 / 9.0
    P_ws_kPa = 0.61094 * np.exp((17.625 * T_c) / (T_c + 243.04))
    return P_ws_kPa / 6.89476  # kPa → psi


def humidity_ratio_from_wet_bulb(t_db, t_wb, P_atm, props):
    """Calcula Y a partir de bulbo seco, bulbo húmedo y presión total."""
    P_total = P_atm * props["pressure_factor"]
    P_ws = sat_vapor_pressure_magnus(t_wb, props["psat_units"])
    Pv = P_ws - props["psychrometric_constant"] * P_total * (t_db - t_wb)
    # Clamp Pv para evitar valores no físicos
    Pv = max(0.0, min(Pv, 0.99 * P_total))
    if P_total <= Pv:
        return None
    return EPSILON * Pv / (P_total - Pv)


def humidity_ratio_from_RH(t_db, RH_percent, P_atm, props):
    """Calcula Y a partir de bulbo seco, HR (%) y presión total."""
    P_total = P_atm * props["pressure_factor"]
    P_ws = sat_vapor_pressure_magnus(t_db, props["psat_units"])
    Pv = (RH_percent / 100.0) * P_ws
    Pv = max(0.0, min(Pv, 0.99 * P_total))
    if P_total <= Pv:
        return None
    return EPSILON * Pv / (P_total - Pv)


def build_equilibrium_functions(teq, Heq_data):
    """Construye funciones H*(T) (spline y lineal) y dH*/dT."""
    # Spline cúbica para Mickley e integración
    H_star_spline = interp1d(teq, Heq_data, kind='cubic', fill_value='extrapolate')
    tck = splrep(teq, Heq_data, k=3)

    # Versión lineal para el cálculo robusto de Gs_min
    H_star_lin = interp1d(teq, Heq_data, kind='linear', fill_value='extrapolate')

    def dH_star_dt(T):
        T_clip = np.clip(T, teq.min(), teq.max())
        return splev(T_clip, tck, der=1)

    return H_star_spline, H_star_lin, dH_star_dt


def compute_min_air_flow(L, Cp_water, H_ini, tini, tfin, H_star_lin, teq):
    """
    Calcula flujo mínimo de aire seco (Gs_min) y el punto de pellizco.
    Método: se buscan pendientes m tales que:
    - Recta desde (tini,H_ini) a (T_p,H*(T_p)) sea creciente.
    - La entalpía de la recta en tfin no supere H*(tfin).
    Se usa H* lineal para evitar artefactos de la spline cúbica.
    """
    if tini >= tfin:
        raise ValueError(
            "La temperatura de salida del agua (tini) debe ser menor que la de entrada (tfin) para calcular el flujo mínimo."
        )

    # Entalpía de equilibrio en la entrada caliente del agua
    H_eq_tfin = float(H_star_lin(tfin))

    # Rango de búsqueda de T_pinch entre tini y tfin
    t_range = np.linspace(tini + 1e-6, tfin, 500)
    mask = (t_range >= teq.min()) & (t_range <= teq.max())
    t_range = t_range[mask]
    if t_range.size == 0:
        raise ValueError(
            "El rango de temperaturas del agua no se superpone con la curva de equilibrio. Ajustar datos."
        )

    slopes = []
    for t_eq in t_range:
        H_eq = float(H_star_lin(t_eq))
        if abs(t_eq - tini) < 1e-9:
            continue
        m = (H_eq - H_ini) / (t_eq - tini)
        if m <= 0:
            continue

        # Entalpía de la recta en tfin
        H_line_tfin = H_ini + m * (tfin - tini)

        # Condición física: la recta mínima no debe ir por encima de H* en tfin
        if H_line_tfin <= H_eq_tfin + 1e-6:
            slopes.append((m, t_eq, H_eq, H_line_tfin))

    if not slopes:
        raise ValueError(
            "No se encontraron pendientes válidas para el flujo mínimo. Revisar datos o viabilidad."
        )

    # Pendiente máxima válida
    m_min, t_pinch, H_pinch, H_line_tfin = max(slopes, key=lambda item: item[0])

    if m_min <= 0:
        raise ValueError(
            "La pendiente máxima calculada para el flujo mínimo es ≤ 0. El enfriamiento deseado sería imposible."
        )

    Gs_min = (L * Cp_water) / m_min
    Hfin_min = H_ini + m_min * (tfin - tini)
    return Gs_min, Hfin_min, t_pinch, H_pinch


def mickley_method(H_ini, H_fin, tG1, tini, tfin, H_star_func, props, n_steps=20):
    """
    Implementa el método de Mickley (integración gráfica con triángulos).
    Devuelve curvas t_air, H_air, Y_air, t_op, H_op, H_star_list y segmentos.
    """
    DH = (H_fin - H_ini) / n_steps
    if DH <= 0:
        raise ValueError("El incremento de entalpía DH resultó ≤ 0. Revisar temperaturas y flujos.")

    t_air = [tG1]
    H_air = [H_ini]
    Y_air = [humidity_ratio_from_H_t(H_ini, tG1, props)]
    t_op = [tini]
    H_op = [H_ini]
    H_star_list = [H_star_func(tini)]
    segmentos = []

    max_iter = 1000
    for _ in range(max_iter):
        H_prev = H_air[-1]
        if H_prev >= H_fin - TOL:
            break

        H_next = min(H_prev + DH, H_fin)
        t_op_next = (H_next - H_ini) * (tfin - tini) / (H_fin - H_ini) + tini
        H_star_next = H_star_func(t_op_next)

        delta_prev = H_star_list[-1] - H_prev
        if abs(delta_prev) < TOL:
            t_next = t_air[-1]
        else:
            t_next = (H_next - H_prev) * (t_op[-1] - t_air[-1]) / delta_prev + t_air[-1]

        H_star_at_tnext = H_star_func(t_next)
        if (H_next - H_star_at_tnext) > 0:
            # La línea de operación cruza el equilibrio
            break

        Y_next = humidity_ratio_from_H_t(H_next, t_next, props)

        H_air.append(H_next)
        t_air.append(t_next)
        Y_air.append(Y_next)
        t_op.append(t_op_next)
        H_op.append(H_next)
        H_star_list.append(H_star_next)

        segmentos.append(((t_next, H_next), (t_op_next, H_next)))
        segmentos.append(((t_op_next, H_next), (t_op_next, H_star_next)))
        segmentos.append(((t_op_next, H_star_next), (t_next, H_next)))

        if H_next >= H_fin - TOL:
            break

    if len(H_air) <= 1:
        raise RuntimeError("No se pudo generar la curva de evolución del aire. Revisar datos de entrada.")
    return t_air, H_air, Y_air, t_op, H_op, H_star_list, segmentos


def compute_NtoG_and_Z(L, G, Y1, H_ini, H_fin, tini, tfin, H_star_func, KYa, props):
    """
    Calcula NtoG, HtoG, Z_total y Lrep a partir de la línea de operación y la curva de equilibrio.
    Usa Y_air_global[-1] como Y de salida (definida en Mickley).
    """
    if KYa == 0:
        raise ValueError("KYa no puede ser cero.")

    y1 = Y1 / (1 + Y1)
    Gs = G * (1 - y1)
    if Gs <= 0:
        raise ValueError("El flujo de aire seco Gs debe ser > 0.")

    n_int = 100
    dt = (tfin - tini) / n_int
    T_water = np.linspace(tini, tfin, n_int + 1)

    H_op_vals = np.interp(T_water, [tini, tfin], [H_ini, H_fin])
    H_star_vals = H_star_func(T_water)

    f_T = []
    for i, T in enumerate(T_water):
        delta = H_star_vals[i] - H_op_vals[i]
        if abs(delta) < TOL:
            raise RuntimeError(
                f"La línea de operación está muy cerca o cruza el equilibrio en T={T:.2f}. No se puede calcular NtoG."
            )
        f_T.append(1.0 / delta)

    dHdT = (H_fin - H_ini) / (tfin - tini)
    NtoG = 0.0
    for i in range(1, len(T_water)):
        NtoG += 0.5 * dt * (f_T[i] + f_T[i - 1])
    NtoG *= dHdT

    HtoG = Gs / KYa
    Z_total = HtoG * NtoG
    Lrep = Gs * (Y_air_global[-1] - Y1)

    return NtoG, HtoG, Z_total, Lrep, Gs


# ==================== SECCIÓN DE ENTRADA DE DATOS ====================

st.subheader('Datos de la Curva de Equilibrio H*(T)')

opcion_unidades = st.radio(
    "Seleccione el sistema de unidades:",
    ('Sistema Inglés', 'Sistema Internacional')
)

props = get_units_config(opcion_unidades)
teq = props["teq"]
Heq_data = props["Heq_data"]

st.sidebar.header('Parámetros del Problema')

P = st.sidebar.number_input('Presión de operación (P, atm)', value=1.0, format="%.2f")

L = st.sidebar.number_input(f'Flujo de agua (L, {props["flow_unit"]})', value=2200.0, format="%.2f")
G = st.sidebar.number_input(f'Flujo de aire (G, {props["flow_unit"]})', value=2000.0, format="%.2f")
tfin = st.sidebar.number_input(f'Temperatura de entrada del agua (tfin, {props["temp_unit"]})', value=105.0, format="%.2f")
tini = st.sidebar.number_input(f'Temperatura de salida del agua (tini, {props["temp_unit"]})', value=85.0, format="%.2f")

Y1_source_option = st.sidebar.radio(
    "Fuente de humedad absoluta del aire a la entrada (Y1):",
    ('Ingresar Y1 directamente', 'Calcular Y1 a partir de Bulbo Húmedo', 'Calcular Y1 a partir de Humedad Relativa')
)

# Inicializar variables de aire de entrada
Y1 = 0.016
tG1 = st.sidebar.number_input(
    f'Bulbo seco del aire a la entrada (tG1, {props["temp_unit"]})',
    value=90.0, format="%.2f"
)

if Y1_source_option == 'Ingresar Y1 directamente':
    tw1 = st.sidebar.number_input(
        f'Bulbo húmedo del aire a la entrada (tw1, {props["temp_unit"]})',
        value=76.0, format="%.2f"
    )
    Y1 = st.sidebar.number_input(
        f'Humedad absoluta del aire a la entrada (Y1, {props["Y_unit"]})',
        value=0.016, format="%.5f"
    )

elif Y1_source_option == 'Calcular Y1 a partir de Bulbo Húmedo':
    tw1 = st.sidebar.number_input(
        f'Bulbo húmedo del aire a la entrada (tw1, {props["temp_unit"]})',
        value=76.0, format="%.2f"
    )
    st.sidebar.write("Calculando Y1 a partir de Bulbo Húmedo:")
    Y1_calc = humidity_ratio_from_wet_bulb(tG1, tw1, P, props)
    if Y1_calc is None:
        st.sidebar.error(
            "Error al calcular Y1: posible saturación o datos inconsistentes. Ajuste bulbo seco/húmedo."
        )
    else:
        Y1 = Y1_calc
        st.sidebar.info(f"Y1 calculado: **{Y1:.5f}** ({props['Y_unit']})")

else:  # Humedad relativa
    relative_humidity = st.sidebar.number_input(
        'Humedad relativa a la entrada (HR, %)',
        value=50.0, min_value=0.0, max_value=100.0, format="%.1f"
    )
    tw1 = 0.0
    st.sidebar.write("Calculando Y1 a partir de humedad relativa:")
    Y1_calc = humidity_ratio_from_RH(tG1, relative_humidity, P, props)
    if Y1_calc is None:
        st.sidebar.error(
            "Error al calcular Y1: posible saturación o datos inconsistentes. Ajuste bulbo seco/HR."
        )
    else:
        Y1 = Y1_calc
        st.sidebar.info(f"Y1 calculado: **{Y1:.5f}** ({props['Y_unit']})")

KYa = st.sidebar.number_input(
    f'Coef. volumétrico de transferencia de materia (KYa, {props["kya_unit"]})',
    value=850.0, format="%.2f"
)

if tini >= tfin:
    st.warning(
        "Advertencia: la temperatura de salida del agua (tini) debe ser menor que la de entrada (tfin) para un enfriamiento."
    )

# ==================== CÁLCULOS PRINCIPALES =================
