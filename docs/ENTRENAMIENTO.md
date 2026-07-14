# Documentación técnica del entrenamiento — commit-summarizer-ms

Fase: ** (Machine Learning)** · Fecha: julio 2026 · Semilla global: `42`

Modelo generativo *sequence-to-sequence* que produce un mensaje de commit en
lenguaje natural a partir del `diff`. Entrenado **desde cero** (sin modelos
pre-entrenados, conforme a los límites del proyecto) en Google Colab (GPU T4).
Notebook reproducible en `training/notebooks/01_resumidor_commits.ipynb`.

## Dataset

Corpus limpio de **NNGen** (Liu et al., ASE 2018), derivado del corpus de
**Jiang, Armaly & McMillan (ASE 2017)** — paper citado en la bibliografía del
proyecto. Pares `diff → mensaje` de los 1,000 proyectos Java más populares de
GitHub, pre-tokenizados. Fuente: `github.com/Tbabm/nngen`.
Particiones según Liu et al.: ~22.1k train / ~2.5k valid / ~2.5k test.

## Preparación
- Corpus pre-tokenizado (tokenización por espacios).
- Truncado: diff a **100 tokens**, mensaje a **25 tokens** (igual que Jiang
  et al.; cubre a la gran mayoría según los percentiles del EDA).
- Vocabularios construidos solo con train (frecuencia mínima 2):
  ~50k tokens (diff) y ~20k (mensajes), con especiales `<pad> <sos> <eos> <unk>`.

## Arquitectura (desde cero, ~15M parámetros)
- **Encoder**: embedding 256 + GRU bidireccional (hidden 256).
- **Decoder**: GRU (hidden 256) con **atención de Luong** (puntaje general)
  sobre las salidas del encoder, con máscara de padding.
- Justificación: es la familia de arquitecturas de la línea de investigación
  citada (NMT aplicada a mensajes de commit), dimensionada para entrenar en
  una sesión de Colab y hacer inferencia en CPU (restricción de despliegue).

## Entrenamiento
- Teacher forcing, Adam (lr 1e-3), clip de gradiente 5.0, batch 64.
- Hasta 25 épocas con **parada temprana** (paciencia 3 sobre pérdida de
  validación). Detenido en la época 8; mejor checkpoint: **época 5**
  (val loss 3.815). La divergencia train/val posterior confirma que más
  épocas solo sobreajustarían.
- Inferencia: **beam search** (ancho 5, normalización por longitud) — mejoró
  BLEU de 4.10 (greedy) a 6.68.

## Resultados (test, 2.5k pares)

| Métrica | Valor | Protocolo |
|---|---|---|
| **BLEU-4** | **6.68** | sacrebleu `tokenize='none'` sobre el corpus pre-tokenizado (mismo protocolo que la literatura) |
| **ROUGE-L** | **0.2092** | rouge-score, F-measure promedio |

### Comparación honesta con la literatura (mismo dataset)

| Sistema | BLEU | Escala |
|---|---|---|
| NMT (Liu et al. 2018, tabla 1) | ~16.4 | seq2seq atencional grande, mucho mayor cómputo |
| NNGen (recuperación, sin red neuronal) | ~38.5 | no es un modelo entrenado |
| **Este trabajo** | **6.68** | ~15M parámetros, 1 sesión de Colab |

La brecha con el NMT de referencia se explica por la escala del modelo y del
cómputo (sin ensembles ni búsqueda extensiva de hiperparámetros). BLEU además
subestima la calidad percibida en esta tarea: compara n-gramas exactos contra
**una única referencia** en mensajes muy cortos.

### Evidencia cualitativa
Ejemplos del test (real vs generado) que ilustran coincidencia semántica que
BLEU no captura:

| Real | Generado |
|---|---|
| Bump parent pom reference . | prepare for next development iteration |
| Fix version in changes log | Updated CHANGES |

(Ambos pares describen el mismo evento con palabras distintas.)

## Artefactos exportados (`training/artefactos/`)

| Archivo | Contenido |
|---|---|
| `modelo_resumidor.pt` | state_dict de encoder+decoder + hiperparámetros |
| `vocab_diff.json`, `vocab_msg.json` | Vocabularios (lista índice→token) |
| `metricas_resumidor.json` | Métricas, versión de torch, contrato de inferencia |

## Restricciones de decodificación en el servicio de inferencia
El servicio (Fase 2) decodifica con beam search (ancho 5) y la restricción
estándar **no-repeat bigram**, que bloquea la repetición degenerativa típica
de los seq2seq compactos con entradas fuera de dominio. Es una restricción de
decodificación: no modifica el modelo ni las métricas reportadas arriba.

## Contrato de inferencia (interfaz para la Fase 2)
Entrada: `{"diff": str}` (salida de `git diff`, texto plano) → preproceso:
tokenización por espacios + truncado a 100 tokens → salida:
`{"resumen": str}` (inglés, ≤ 25 tokens). El servicio debe **filtrar tokens
`<unk>`** del texto final (post-proceso de presentación).

## Limitaciones
- Resúmenes en inglés y de **un commit individual** (límites del proyecto).
- Calidad modesta (BLEU 6.68): el sistema es una *asistencia editable*, no un
  generador autónomo — alineado con el alcance definido.
- Corpus de proyectos Java: posible degradación en otros lenguajes.
- Diffs muy largos se truncan a 100 tokens: cambios masivos pierden contexto.
