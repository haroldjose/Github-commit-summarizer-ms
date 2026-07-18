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

| Sistema | BLEU (corpus limpio) | Escala |
|---|---|---|
| NMT (Liu et al. 2018) | ~16.4 | seq2seq atencional grande, mucho mayor cómputo |
| NNGen (recuperación, sin red neuronal) | ~16.4 | no es un modelo entrenado |
| **Este trabajo — seq2seq base (Fase 1)** | **6.68** | ~15M parámetros, 1 sesión de Colab |
| **Este trabajo — pointer-generator (Fase 3)** | **11.62** | ~15M parámetros, mismo corpus y semilla |
| **Este trabajo — recuperación NNGen pura** | **16.72** | réplica del algoritmo sobre test completo |
| **Este trabajo — recuperación gated** (cos≥0.5 ∧ BLEU-diff≥0.5) | **23.80*** | 65 % de casos del test; el resto cae a ptrgen/heurística |

*Sobre el subconjunto donde el vecino supera ambos umbrales (1,647/2,521). El BLEU ~38
que suele citarse para NNGen corresponde al dataset **original ruidoso** de Jiang et al.;
sobre el corpus limpio el paper reporta ~16.4, que nuestra réplica reproduce (16.72).

La brecha del seq2seq con el NMT de referencia se explica por la escala del modelo y del
cómputo (sin ensembles ni búsqueda extensiva de hiperparámetros). BLEU además
subestima la calidad percibida en esta tarea: compara n-gramas exactos contra
**una única referencia** en mensajes muy cortos.

### Evidencia cualitativa (mismos 4 diffs reales, antes → después)

| Diff | Antes (seq2seq crudo) | Ahora (híbrido) |
|---|---|---|
| Python: lógica de precios en `calcular_total` | add migration to | Update calcular_total |
| Java: nuevo método `deleteUser` | Fix migration ; reviewed by… | Add deleteUser method |
| Java: bump `engine.io-client` 0.6.1→0.6.2 | gh - Add migration ( via… | bump engine.io - client |
| Java: null check en `close` | Fix migration timestamp… | Add null check |

## Artefactos exportados (`training/artefactos/`)

| Archivo | Contenido |
|---|---|
| `modelo_resumidor.pt` | state_dict de encoder+decoder + hiperparámetros |
| `vocab_diff.json`, `vocab_msg.json` | Vocabularios (lista índice→token) |
| `metricas_resumidor.json` | Métricas, versión de torch, contrato de inferencia |

## Restricciones de decodificación en el servicio de inferencia
El servicio (Fase 2) decodifica con beam search (ancho 5) y las restricciones
estándar **no-repeat bigram** (bloquea la repetición degenerativa típica de
los seq2seq compactos) y **longitud mínima** (3 tokens, evita salidas
triviales tipo "Fix"). Son restricciones de decodificación: no modifican el
modelo ni las métricas reportadas arriba.

## Estrategia híbrida: recuperación NNGen + heurística + generativo (Fase 2)
Además del modelo generativo, el servicio implementa la **recuperación de
NNGen** (Liu et al., ASE 2018 — el mismo paper de la comparación de arriba):
ante un diff nuevo, busca los 5 diffs más parecidos del corpus de
entrenamiento (bolsa de palabras + similitud coseno), re-rankea por BLEU-4
entre diffs y devuelve el mensaje real del vecino ganador. **No usa ninguna
red neuronal ni modelo pre-entrenado** — cumple los límites del proyecto; es
memoria del propio corpus de entrenamiento.

Política del servicio (en orden):

1. **Heurística SE** (línea ChangeScribe): plantillas compuestas para archivo
   nuevo/borrado, bump de versión, métodos nuevos (Java/Python/JS), null
   check e imports. Van primero porque extraen valores del **propio** diff
   (siempre verificables); un vecino recuperado casi idéntico en estructura
   puede traer detalles ajenos (otra versión, otro artefacto).
2. **Recuperación** si coseno ≥ 0.5 **y** BLEU-4 entre diffs ≥ 0.5.
   El coseno solo no basta: se infla con tokens estructurales (`mmm`/`ppp`/`<nl>`)
   en diffs fuera de distribución y devolvía mensajes irrelevantes.
3. **Generativo** (pointer-generator de la Fase 3, o el seq2seq base si no
   está su checkpoint) con beam search, no-repeat bigram, longitud mínima y
   detokenizador. Si la salida es degenerada (`Fix #`), se descarta.

El corpus de recuperación (`corpus_train.diff/msg`, ~8 MB) viaja junto a los
artefactos del modelo.

## Fase 3: pointer-generator (copy mechanism) — reentrenado desde cero

Reentrenamiento del generativo con la arquitectura **pointer-generator**
(See et al., ACL 2017 — citado en los antecedentes del proyecto), en la línea
de PtrGNCMsg (Liu et al., MSR 2019). En cada paso, el decoder combina:

    P(w) = p_gen · P_vocab(w) + (1 − p_gen) · Σ atención sobre posiciones del diff donde aparece w

Esto le permite **copiar identificadores del código** (nombres de métodos,
clases, versiones) directamente del diff al mensaje, aunque no existan en el
vocabulario — la debilidad principal del seq2seq base.

- Script reproducible: `training/train_ptrgen.py` (mismo corpus, misma
  semilla 42, mismos vocabularios y truncados que la Fase 1; ~15M parámetros).
- Entrenado en CPU local (24 núcleos, ~2.6 min/época); parada temprana en la
  época 6, mejor checkpoint: época 3 (val loss 3.593).
- Sin modelos pre-entrenados: cumple los límites del proyecto.

| Métrica (test, 2,521 pares) | Fase 1 (seq2seq) | Fase 3 (ptrgen) | Δ |
|---|---|---|---|
| BLEU-4 (protocolo `tokenize='none'`) | 6.68 | **11.62** | +74 % |
| ROUGE-L (F, rouge-score) | 0.2092 | **0.2927** | +40 % |

Ejemplo del efecto de la copia (diff nuevo, fuera del corpus): ante el método
`deleteUser`, el modelo base generaba mensajes sin relación; el
pointer-generator produce "Add deleteUser method to …" copiando el
identificador del diff.

El servicio carga `modelo_resumidor_ptrgen.pt` automáticamente si existe
junto a los artefactos (y cae al checkpoint de la Fase 1 si no).

## Poda del diff para entradas largas (Fase 2)

El corpus solo contiene diffs cortos (máximo 121 tokens; p99 = 98), pero los
diffs reales pueden medir miles. Con el truncado ingenuo ("primeros 100
tokens") el modelo solo veía el inicio del archivo (imports) y alucinaba
mensajes sin relación. El servicio aplica ahora una **poda estructural** al
estilo CODISUM cuando el diff excede el presupuesto: conserva cabeceras
(`mmm`/`ppp`), metadatos (`new file`, `Binary`, `index`…) y líneas `+`/`-`,
descartando el contexto sin cambiar. El presupuesto se reparte **round-robin
entre los bloques de cambio** (hunks/archivos): primero la primera línea de
cada bloque, luego la segunda, etc., de modo que un diff enorme
multi-archivo queda representado por el inicio de *todos* sus cambios y no
solo por el primero. Es análisis del
formato unified-diff, agnóstico al lenguaje. Percentiles medidos: la poda
lleva el p95 del corpus de 93 a 67 tokens, y un diff real de 1,185 tokens
queda en ~90 tokens de cambios.

## Fase 4: modelo multilenguaje (Java + Python + JS + C++ + C#)

Reentrenamiento del pointer-generator, desde cero, sobre un corpus mixto:
Java (NNGen, 22,112) + **30,000 pares por lenguaje** muestreados (semilla 42)
del dataset **MCMD** (Tao et al., ICSME 2021; splits procesados distribuidos
con RACE, Shi et al. EMNLP 2022 — zenodo.org/record/7196966, mismo formato
pre-tokenizado que NNGen). Filtros: mensaje de 3-25 tokens, sin duplicados,
sin mensajes de merge/revert/bot (criterio de limpieza de Liu et al. que
MCMD no aplica), y **la misma poda condicional del servicio** para diffs
>100 tokens — el modelo entrena con la distribución exacta de producción.

Vocabularios reconstruidos sobre la mezcla (50k/20k → 34.7M parámetros).
Entrenado en CPU local (~21 min/época); parada temprana en la época 5,
mejor checkpoint: época 2 (val loss 3.564). Script: `training/train_multilang.py`
(+ `training/prepara_multilang.py`). Artefactos: `training/artefactos_v3/`
(drop-in completo: modelo + vocabs + corpus de recuperación de 133k pares).

| Test (2.5k pares c/u) | BLEU-4 v3 | Antes (modelo solo-Java) |
|---|---|---|
| Python | **11.83** | ~0 (fuera de dominio) |
| JavaScript | **12.19** | ~0 |
| C++ | **11.82** | ~0 |
| C# | **12.72** | ~0 |
| Java | 9.52 | 11.62 |
| **Global** | **11.79** | — |

Trade-off medido y aceptado: −2.1 BLEU en Java a cambio de cobertura real
en 4 lenguajes nuevos al mismo nivel (~12). El filtro de salidas degeneradas
del servicio rechaza además las muletillas de merge que el modelo pueda
generar por el ruido residual del corpus.

### Experimento: reentrenar con poda estricta (resultado negativo, documentado)
Se reentrenó el pointer-generator aplicando la misma poda también al corpus
de entrenamiento (v2: misma semilla, mismos vocabularios; val loss 3.570,
mejor que v1). En test: BLEU 10.97 / ROUGE-L 0.2909 — **por debajo del v1**
(11.62 / 0.2927): en los diffs cortos del corpus el contexto sí aporta. En
diffs largos sintéticos, v1+poda igualó o superó a v2 (p. ej. "Upgrade
version to 1.5.0" con el cambio enterrado tras 40 líneas de contexto, donde
el truncado ingenuo producía "Remove unused variable"). Decisión: **v1 +
poda condicional** (solo para diffs que exceden el presupuesto); el v2 no se
promueve. Métricas del experimento en `metricas_ptrgen_v2.json`.

### Técnicas restantes (posible Fase 4)
- **CoDiSum / CoreGen** — encoder más rico o Transformer contextualizado entrenado en el corpus.
- **CoRec** — fusión neural de recuperación + generación en el decoder.

Ninguna usa modelos pre-entrenados externos.

## Contrato de inferencia (interfaz para la Fase 2)
Entrada: `{"diff": str}` (salida de `git diff`, texto plano) → preproceso:
**normalización al formato del corpus NNGen** (la que el corpus ya traía
aplicada: `---`/`+++` → `mmm`/`ppp`, cabeceras `@@ … @@` eliminadas
conservando el contexto, saltos de línea → token `<nl>`, puntuación separada
por espacios con `_` conservado en identificadores) + tokenización por
espacios + truncado a 100 tokens → salida: `{"resumen": str}` (inglés,
≤ 25 tokens). El servicio debe **filtrar tokens `<unk>`** del texto final
(post-proceso de presentación). Sin la normalización, ~50 % de los tokens de
un `git diff` crudo caen fuera del vocabulario y el modelo genera mensajes
sin relación con el cambio.

## Limitaciones
- Resúmenes en inglés y de **un commit individual** (límites del proyecto).
- Calidad modesta (BLEU 6.68): el sistema es una *asistencia editable*, no un
  generador autónomo — alineado con el alcance definido.
- Corpus de proyectos Java: posible degradación en otros lenguajes.
- Diffs muy largos se truncan a 100 tokens: cambios masivos pierden contexto.
