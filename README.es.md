# Agente CFDI

Automatización de cuentas por pagar para facturación mexicana. Lee la bandeja de
facturación, extrae y valida CFDIs, detecta anomalías y responde preguntas sobre
el gasto en lenguaje natural.

*[Read this in English](README.md)*

---

## La decisión de diseño de la que trata el proyecto

> **Un CFDI ya es dato estructurado con esquema publicado. Pasarlo por un LLM
> cuesta dinero por documento e introduce un modo de falla —error de
> transcripción— que de otro modo no existe.**

Así que cada documento se enruta a la capa más barata que lo resuelve
correctamente:

| Tier | Qué hace | Costo | Determinista |
|---|---|---|---|
| **0 — código** | Parseo de CFDI 4.0/3.3, aritmética, validación de RFC y catálogos | $0 | sí |
| **1 — local** | Embeddings, duplicado semántico, transcripción de escaneos | $0 marginal | no |
| **2 — API** | Loop del agente, explicación de anomalías, PDFs difíciles | medido por llamada | no |

**El LLM nunca escribe en la base de datos.** Todo lo que produce pasa por la
misma validación determinista que la ruta XML. Un total alucinado llega como
cadena bien formada, falla la aritmética y cae en la cola de revisión.

La orquestación es n8n; la lógica de dominio es Python probado detrás de HTTP.
Una regla fiscal expresada como nodos de canvas es una regla que nada puede
probar y nadie puede revisar.

---

## Resultados medidos

De `python -m evals.run_eval` — 300 facturas sintéticas, seed 1312, contra una
base dedicada. Se regenera con un comando; nada de esto es estimado.

**Exactitud campo a campo, tier 0** — 100% en `uuid`, `rfc_emisor`,
`rfc_receptor`, `subtotal`, `total`, `moneda`, `n_conceptos` (286/286 cada uno).

**Detección de anomalías**

| defecto | inyectados | recall | precisión | F1 |
|---|---:|---:|---:|---:|
| `bad_rfc` | 10 | 1.00 | 1.00 | 1.00 |
| `dup_uuid` | 14 | 1.00 | 1.00 | 1.00 |
| `folio_gap` | 12 | 1.00 | 0.60 | 0.75 |
| `line_math` | 4 | 1.00 | 1.00 | 1.00 |
| `price_spike` | 10 | 0.90 | 1.00 | 0.95 |
| `semantic_dup` | 15 | 0.00 | — | — |
| `total_mismatch` | 6 | 1.00 | 1.00 | 1.00 |

**Rendimiento** 21 documentos/s · p50 15 ms · p95 22 ms
**Costo** $0.00 por factura en la ruta XML — no interviene ningún modelo
**Esquema** 290/300 validan contra el XSD oficial del SAT; los 16 fallos son
exactamente los RFCs malformados a propósito

`semantic_dup` sale en recall 0.00 porque la etapa vectorial nunca corrió: esta
máquina no tiene backend de embeddings, así que ningún concepto se embebió. El
reporte lo dice en su propia sección en vez de dejar el cero como resultado. El
detector 2 se prueba contra un embebedor stub; su recall sobre texto reformulado
está sin medir.

`price_spike` sale en 0.90: uno de diez picos inyectados cayó sobre un producto
cuyo historial de precios estaba una factura por debajo del piso de cinco
muestras que el detector exige. Se negó a juzgar, que es el comportamiento
diseñado.

`folio_gap` es el débil, y el número se publica en vez de enterrarse. Cada falso
positivo traza a una factura con RFC de emisor malformado, que la archiva bajo
otro proveedor y deja hueco en la secuencia del real. Discutiblemente el
veredicto correcto, e inflado aquí por un corpus que corrompe RFCs al ~5%.

**De siete tipos de defecto inyectados, la validación XSD atrapa uno.**
Duplicados, precios inflados y totales que no cuadran son todos perfectamente
válidos contra el esquema. Pasar validación de esquema no dice nada sobre si una
factura debe pagarse — que es el argumento entero de la suite de detectores.

---

## Arquitectura

```
  ┌──────────── n8n — orquestación, editable en el canvas ─────────────┐
  │  Webhook / Gmail ─► POST /ingest ─► If status == "anomaly" ─► Slack │
  │  Cron 09:00 ──────► GET /anomalies/open ─► formato ────────► Slack  │
  └────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP
  ┌────────────────────────────▼─── FastAPI — dominio, Python ─────────┐
  │  dedupe por hash ─► router ─┬─ .xml ─► parser lxml       (tier 0)   │
  │                             └─ .pdf ─► visión          (tier 1/2)   │
  │                                 │                                   │
  │  validación ─┬─ rechazada ─► review_queue                           │
  │              └─ aceptada ──► Postgres + pgvector                    │
  │                                │                                    │
  │                        embeddings ─► 8 detectores                   │
  └────────────────────────────┬───────────────────────────────────────┘
                               ▼
                agente NL (solo lectura, solo vistas)
```

Cada documento escribe una fila en `extraction_runs` — modelo, latencia, tokens,
costo. Esa tabla **es** el harness de evaluación y el tablero de costos.

---

## Detectores

| # | Detector | Método | Severidad |
|---|---|---|---|
| 1 | `duplicate_uuid` | Constraint UNIQUE más chequeo explícito | crítico |
| 2 | `semantic_duplicate` | Mismo emisor, total ±1%, fecha ±7d, coseno > 0.93 sobre centroides de conceptos | crítico |
| 3 | `price_outlier` | Z robusta por MAD por (proveedor, producto), con piso y compuerta de materialidad | warn |
| 4 | `total_mismatch` / `subtotal_mismatch` / `line_math_mismatch` | Vuelve a sumar la factura | crítico |
| 5 | `invalid_rfc` | El patrón `t_RFC` del propio SAT | crítico |
| 6 | `new_supplier` | Primera factura de ese RFC | info |
| 7 | `folio_gap` | Salto de secuencia por (emisor, serie) | warn |
| 8 | `stale_stamp` | Timbrado más de 72h después de emitido | warn |
| 9 | `unknown_catalog_code` | Código SAT fuera del subconjunto incluido | info |

Cada uno emite `evidence` en JSONB — los valores exactos que dispararon la regla.
El LLM puede resumir esa evidencia; no puede inventar un hallazgo sin ella.

---

## Arranque

```bash
cp .env.example .env
docker compose up -d db                      # Postgres 17 + pgvector
python -m venv .venv && ./.venv/bin/pip install -e '.[api,llm,synth,dev]'

python -m scripts.fetch_xsd                  # versiona la cadena XSD del SAT
python -m synth.generate_cfdi --out data/synth
pytest                                       # sin API key, sin red

python -m cfdi_agent.ingest.watch_dir --once data/synth
python -m evals.run_eval                     # escribe evals/report.md
```

Levantar el API y los flujos:

```bash
uvicorn cfdi_agent.api.main:app --port 8000
docker compose up -d n8n
python -m flows.sync build                   # compila, no necesita n8n
python -m flows.sync push                    # necesita N8N_API_KEY
```

Preguntar (necesita `ANTHROPIC_API_KEY`):

```bash
python -m cfdi_agent.agent.loop -v \
  "¿Cuánto gasté con ACME en Q2 y hubo algo raro?"
```

Correr la extracción contra un modelo local — sin cambios de código:

```bash
LLM_PROVIDER=local LLM_BASE_URL=http://orin.local:8080/v1 \
  python -m evals.run_eval
```

**Hasta dónde llega la costura.** Esa variable controla únicamente la ruta de
extracción de documentos. Los embeddings siempre corren local, porque la API de
Anthropic no los sirve. **El agente de lenguaje natural siempre usa la API.**

La razón es concreta: el agente usa el tool runner del SDK de Anthropic, y un
servidor con API compatible OpenAI no lo tiene — expone un parámetro `tools` en
`/chat/completions`, pero el loop de llamadas hay que escribirlo. Mandar el
agente a un modelo local significa implementar ese loop en `openai_compat.py` y
hacer que `agent/loop.py` pase por `get_provider()`.

---

## n8n como código

Los workflows se definen en Python, se compilan a JSON de n8n, se empujan por la
API pública y se exportan de vuelta. Ambas direcciones son el mismo archivo.

```python
wf = Workflow("cfdi-invoice-intake")
trigger = wf.add(webhook("Factura recibida", path="cfdi"))
ingest  = wf.add(http_request("Ingestar CFDI", url=f"{api}/ingest", method="POST"))
branch  = wf.add(if_equals("¿Hay anomalía?", left="={{ $json.status }}", right="anomaly"))
wf.chain(trigger, ingest, branch)
wf.connect(branch, alert, port=0)   # rama true
```

```bash
python -m flows.sync push      # defines aquí, corre allá
# ...alguien reacomoda el canvas...
python -m flows.sync export    # y el diff se lee
```

El normalizador de exportación es lo que hace real ese último paso: un export
crudo de n8n trae ids del servidor, `updatedAt`, un `webhookId` aleatorio por
nodo y posiciones exactas al pixel. Commiteas eso y cada diff es ruido, así que
nadie los lee, así que el round-trip se abandona en una semana. Normalizar tira
el estado volátil, recalcula ids de nodo determinísticamente, cuadricula
posiciones y ordena. Un test simula una edición en el canvas y exige que la forma
normalizada sea byte-idéntica — con un test compañero que verifica que una
edición *real* sí aparece.

Los `typeVersion` de los nodos se leyeron del catálogo de tipos del n8n en
ejecución, no de memoria. Uno equivocado produce un workflow que importa sin
quejarse y luego se comporta distinto.

---

## Seguridad

Las descripciones de conceptos las escribe el proveedor y terminan en el contexto
del agente. "Ignora las instrucciones previas" es algo que un vendedor te puede
facturar, así que las propiedades de seguridad del agente son estructurales, no
basadas en el prompt:

- `SET TRANSACTION READ ONLY`, impuesto por Postgres
- Allowlist de vistas — rutas de archivo, hashes y cola de revisión inalcanzables
- Una sola sentencia, `LIMIT` forzado, timeout de sentencia
- Filtrado de keywords como defensa en profundidad, explícitamente *no* el
  respaldo: filtrar SQL con regex es juego perdido

El agente puede leer anomalías pero no crearlas. La detección ya ocurrió, de
forma determinista, antes de que viera nada.

El XML de facturas es entrada no confiable: el parser no resuelve entidades
externas.

---

## Lo que esto NO hace

Dicho sin rodeos, porque un demo que sobrepromete es peor que uno con una lista
corta y honesta.

- **El corpus es sintético.** Generado desde la estructura CFDI 4.0 y validado
  contra el XSD oficial, pero no incluye facturas reales — las reales cargan RFCs
  de terceros y UUIDs rastreables que no van en un repo público.
- **Los sellos digitales no se verifican.** `Sello`, `Certificado` y `SelloSAT`
  están presentes pero nunca se revisan; hacerlo bien requiere la cadena de
  certificados del PAC.
- **No hay verificación de estatus ante el SAT.** La columna
  `invoices.sat_status` existe y siempre queda en NULL. Preguntarle al SAT si un
  UUID sigue vigente requiere el web service `ConsultaCFDIService`, y no está
  implementado.
- **El detector 2 necesita un backend de embeddings.** Sin `EMBED_BASE_URL` la
  etapa vectorial se salta y el motivo queda en `extraction_runs`. Su recall
  sobre texto reformulado está sin medir, porque medirlo exige un modelo
  corriendo.
- **Los catálogos SAT son un subconjunto.** Solo `c_ClaveProdServ` son ~52,000
  filas. Un código desconocido produce una nota `info`, nunca un rechazo.
- **El dígito verificador del RFC no se calcula.** Adoptar el regex del propio
  SAT recupera parte —el último carácter solo puede ser `[0-9A]`— pero el
  algoritmo completo no está implementado, porque equivocarlo rechazaría
  contribuyentes válidos.
- **La costura de proveedores no cubre al agente.** `LLM_PROVIDER` mueve la
  extracción de documentos a un modelo local. El agente de lenguaje natural
  siempre usa la API. La razón está en la sección de arranque.
- **El tier 2 está sin medir.** La ruta de visión y la comparación API vs local
  necesitan credenciales; `evals/report.md` lo dice en vez de dejar un blanco.

---

## Ruta de falla

Qué pasa cuando el modelo se equivoca:

1. La salida de visión se valida contra un schema Pydantic. Malformada → cola de
   revisión.
2. Luego se vuelve a sumar: conceptos, subtotal, impuestos, total. Cualquier
   descuadre → anomalía crítica.
3. Nada llega a `invoices` sin estar completamente validado — donde "validado"
   significa *sabemos exactamente qué tiene mal*, no que no tenga nada mal. Una
   factura que falla la aritmética **se persiste y se marca**, porque
   registrarla es todo el punto.
4. `review_queue` guarda documentos que no pudimos extraer con confianza, o que
   van dirigidos a otra empresa.

---

## Estructura

```
src/cfdi_agent/
  schemas.py          modelo canónico Decimal + schema LLM sin constraints
  extract/            xml_parser (tier 0), pdf_vision, providers/
  validate/           rules (puras), catalogs, xsd_validator
  enrich/             embeddings, duplicados semánticos
  db/                 schema.sql, repo, conexión
  ingest/             pipeline, watch_dir, dedupe
  agent/              tools (frontera de seguridad), loop
  api/                FastAPI — el contrato que consume n8n
flows/                n8n como código: builder, definiciones, JSON exportado
synth/                generador de corpus con ground truth gratis
evals/                harness y reporte
xsd/                  cadena XSD del SAT versionada + manifiesto de procedencia
```

---

## Stack

Python 3.12 · Postgres 17 + pgvector · FastAPI · n8n · API de Anthropic
(`claude-opus-5`) · llama.cpp para la ruta local · lxml · Pydantic v2 · Docker
Compose
