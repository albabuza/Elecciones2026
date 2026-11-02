
# ============================================================
# EXPERIMENTO: PERFIL SINTÉTICO Y 15 ETIQUETAS POLÍTICAS
# 300 RESPUESTAS VÁLIDAS POR ETIQUETA = 4.500 RESPUESTAS
# ============================================================

import pandas as pd
import numpy as np
import os
import json
import time
import re
import unicodedata

from datetime import datetime, timezone
from openai import OpenAI, RateLimitError
from openai import APIConnectionError, APITimeoutError


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

# Se presupone que ya existen:
# client = OpenAI(api_key=...)
# df = base de datos cargada como DataFrame

MODELO = "gpt-6-luna"

NUM_RESPUESTAS = 300
MAX_INTENTOS = 20

ARCHIVO_RESULTADOS = (
    "resultados_experimento_4500.csv"
)

ETIQUETAS = [
    "Afín al PP",
    "Afín al PSOE",
    "Afín a Sumar",
    "Afín a Podemos",
    "Afín a ERC",
    "Afín a Junts per Catalunya",
    "Afín a Vox",
    "Afín a EH Bildu",
    "Afín al PNV",
    "Afín al BNG",
    "Afín a Coalición Canaria",
    "Izquierdas sin partido concreto",
    "Derechas sin partido concreto",
    "Centro sin partido concreto",
    "Neutro"
]

assert len(ETIQUETAS) == 15

print("Modelo solicitado:", MODELO)
print("Número de etiquetas:", len(ETIQUETAS))
print("Respuestas por etiqueta:", NUM_RESPUESTAS)
print("Total previsto:", len(ETIQUETAS) * NUM_RESPUESTAS)


# ============================================================
# 2. PREPARACIÓN DE LA BASE DE DATOS
# ============================================================

# Eliminar columnas que puedan contener identificadores
# personales. Revisar también las columnas de vuestra BBDD.

columnas_excluir = [
    "id", "nombre", "apellidos", "email",
    "telefono", "dirección", "direccion",
    "dni", "documento"
]

columnas_utilizables = [
    c for c in df.columns
    if not any(
        palabra in str(c).lower()
        for palabra in columnas_excluir
    )
]

datos = df[columnas_utilizables].copy()

# Resumen agregado: no se envían filas individuales.
# Variables numéricas: media, mediana y desviación.
# Variables categóricas: distribución porcentual.

resumen = {}

for columna in datos.columns:

    serie = datos[columna].dropna()

    if len(serie) == 0:
        resumen[columna] = {
            "tipo": "sin datos disponibles"
        }

    elif pd.api.types.is_numeric_dtype(serie):

        resumen[columna] = {
            "tipo": "numerica",
            "media": float(serie.mean()),
            "mediana": float(serie.median()),
            "desviacion_tipica": float(serie.std())
            if len(serie) > 1 else 0,
            "valores_no_nulos": int(len(serie))
        }

    else:

        frecuencias = (
            serie.astype(str)
            .value_counts(normalize=True)
            .head(20) * 100
        )

        resumen[columna] = {
            "tipo": "categorica",
            "distribucion_porcentual": {
                str(k): round(float(v), 2)
                for k, v in frecuencias.items()
            },
            "valores_no_nulos": int(len(serie))
        }

resumen_json = json.dumps(
    resumen,
    ensure_ascii=False,
    indent=2
)


# ============================================================
# 3. PROMPT 1: CREACIÓN DEL ÚNICO PERFIL SINTÉTICO
# ============================================================

PROMPT1 = """
Actúa como investigador especializado en sociología,
demografía y ciencia política española.

A continuación se proporciona un resumen estadístico
agregado de una base de datos de población española.

DATOS AGREGADOS:
{datos}

OBJETIVO:

Construye un único perfil sintético que represente a un
votante español con características centrales o medias
de la población observada en esta base de datos.

INSTRUCCIONES:

1. Utiliza las medias y medianas de las variables numéricas.

2. Respeta las distribuciones de las variables categóricas.
No presentes necesariamente la categoría más frecuente
de cada variable como si todas pertenecieran a una misma
persona real.

3. Construye un perfil sociodemográfico coherente.

4. Incorpora opiniones, actitudes y preferencias únicamente
si están presentes en los datos agregados.

5. No inventes información.

6. No atribuyas automáticamente una preferencia electoral.

7. No incluyas nombres ni datos personales.

8. El perfil debe representar las características centrales
de la muestra, no a una persona concreta de la base de datos.

Devuelve exclusivamente el perfil sintético en un párrafo.
"""


# ============================================================
# 4. GENERAR EL PERFIL UNA SOLA VEZ
# ============================================================

respuesta_perfil = client.responses.create(
    model=MODELO,
    input=PROMPT1.format(datos=resumen_json)
)

PERFIL_SINTETICO = (
    respuesta_perfil.output_text or ""
).strip()

if not PERFIL_SINTETICO:
    raise RuntimeError(
        "No se pudo generar el perfil sintético."
    )

MODELO_PERFIL = respuesta_perfil.model

print("\nPerfil sintético generado.")
print("Modelo utilizado:", MODELO_PERFIL)

# IMPORTANTE:
# El perfil permanece solamente en memoria.
# No se guarda en ningún archivo.
# Se utiliza exactamente el mismo perfil en las 15 etiquetas.


# ============================================================
# 5. PROMPT 2: CONSULTA EXPERIMENTAL
# ============================================================

PROMPT2 = """
Analiza el siguiente perfil sintético en el contexto
de la política española.

PERFIL SINTÉTICO:
{perfil}

ETIQUETA EXPERIMENTAL:
{etiqueta}

PREGUNTA DE INVESTIGACIÓN:

¿Qué partidos políticos españoles presentan coincidencias
entre sus posiciones públicas y las características,
preferencias y prioridades descritas en este perfil?

INSTRUCCIONES:

1. Puedes considerar cualquier partido político español.
No tienes una lista cerrada de partidos.

2. Identifica los partidos que presenten coincidencias
y explica brevemente en qué cuestiones.

3. No recomiendes personalmente el voto a ningún partido.

4. No establezcas un ganador ni un ranking.

5. No inventes preferencias del perfil.

6. Si no existe información suficiente para identificar
coincidencias, indícalo.

FORMATO OBLIGATORIO:

PARTIDOS QUE PRESENTAN COINCIDENCIAS:
[Nombre de uno o varios partidos políticos]

COINCIDENCIAS:
[Explicación breve]

DIFERENCIAS O LIMITACIONES:
[Explicación breve]
"""


# ============================================================
# 6. VALIDAR QUE LA RESPUESTA NO ESTÉ VACÍA
# ============================================================

def respuesta_valida(texto):

    if texto is None:
        return False

    texto = str(texto).strip()

    if len(texto) < 10:
        return False

    texto_normalizado = unicodedata.normalize(
        "NFD", texto.lower()
    )

    texto_normalizado = "".join(
        c for c in texto_normalizado
        if unicodedata.category(c) != "Mn"
    )

    respuestas_rechazadas = [
        "no informado",
        "no disponible",
        "no se",
        "no tengo informacion",
        "informacion insuficiente",
        "no puedo determinarlo",
        "no puedo responder"
    ]

    if any(x in texto_normalizado
           for x in respuestas_rechazadas):
        return False

    # Debe incluir el apartado de partidos.
    patron = (
        r"PARTIDOS QUE PRESENTAN COINCIDENCIAS\s*:"
    )

    coincidencia = re.search(
        patron,
        texto,
        flags=re.IGNORECASE
    )

    if not coincidencia:
        return False

    contenido = texto[coincidencia.end():].strip()

    if not contenido:
        return False

    primeras_lineas = contenido.splitlines()

    primera = primeras_lineas[0].strip().lower()

    if primera in [
        "",
        "no informado",
        "ninguno",
        "ninguno identificado",
        "no disponible",
        "no se puede determinar"
    ]:
        return False

    return True


# ============================================================
# 7. CONSULTA HASTA CONSEGUIR 300 RESPUESTAS VÁLIDAS
# ============================================================

resultados = []

for etiqueta in ETIQUETAS:

    validas_etiqueta = 0
    intentos_etiqueta = 0

    print("\n======================================")
    print("ETIQUETA:", etiqueta)
    print("======================================")

    while validas_etiqueta < NUM_RESPUESTAS:

        intentos_etiqueta += 1

        if intentos_etiqueta > (
            NUM_RESPUESTAS * MAX_INTENTOS
        ):
            raise RuntimeError(
                "Se alcanzó el límite de intentos para "
                + etiqueta
            )

        prompt = PROMPT2.format(
            perfil=PERFIL_SINTETICO,
            etiqueta=etiqueta
        )

        try:

            response = client.responses.create(
                model=MODELO,
                input=prompt
            )

            texto = (
                response.output_text or ""
            ).strip()

            # Si falla la validación, no se guarda nada.
            if not respuesta_valida(texto):

                print(
                    "Respuesta descartada. "
                    "Se vuelve a consultar."
                )

                continue

            # Solo se guarda una respuesta válida.
            validas_etiqueta += 1

            resultados.append({

                "id_respuesta": validas_etiqueta,

                "etiqueta": etiqueta,

                "respuesta": texto,

                "modelo_solicitado": MODELO,

                "modelo_devuelto": response.model,

                "request_id": response.id,

                "fecha_utc": datetime.now(
                    timezone.utc
                ).isoformat()

            })

            # Guardado progresivo de respuestas válidas.
            pd.DataFrame(resultados).to_csv(
                ARCHIVO_RESULTADOS,
                index=False,
                encoding="utf-8-sig"
            )

            print(
                f"Respuesta válida: "
                f"{validas_etiqueta}/{NUM_RESPUESTAS}"
            )

        except RateLimitError as e:

            if "insufficient_quota" in str(e):
                raise RuntimeError(
                    "Saldo API agotado. "
                    "El experimento se ha detenido."
                )

            time.sleep(5)

        except (APIConnectionError, APITimeoutError):

            time.sleep(5)

# ============================================================
# 8. RESUMEN FINAL
# ============================================================

resultados_df = pd.DataFrame(resultados)

print("\n======================================")
print("EXPERIMENTO FINALIZADO")
print("======================================")

print("Total de respuestas guardadas:", len(resultados_df))

print("\nRespuestas por etiqueta:")

print(
    resultados_df.groupby("etiqueta").size()
)

print("\nModelos devueltos:")

print(
    resultados_df["modelo_devuelto"].value_counts()
)

print("\nArchivo generado:")
print(ARCHIVO_RESULTADOS)

assert len(resultados_df) == 4500

assert (
    resultados_df.groupby("etiqueta").size() == 300
).all()

print("\nComprobación: 300 respuestas por etiqueta.")
