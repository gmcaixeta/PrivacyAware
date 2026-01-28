# Projeto PrivacyAware


Objetivo do projeto:

Classificar pedidos de acesso à informação em:

* **1 = contém dados pessoais**
* **0 = não contém**

Maximizando **F1-score**, com **ênfase em recall** (minimizar falsos negativos).


---
## 🎯 Objetivo do pipeline

* ✔️ Maximizar recall 
* ✔️ Simples de explicar 
* ✔️ Fácil de rodar e reproduzir
* ✔️ Não depender de LLM, GPU ou APIs externas
* ✔️ Robustez > sofisticação

---

## 🧠 Estratégia geral (o pulo do gato)

**Pipeline híbrido**:

1. **Regras determinísticas (regex)** → capturam casos óbvios
2. **Modelo estatístico simples (ML clássico)** → pega padrões não explícitos
3. **OR lógico final** → se *qualquer* um detectar → classifica como positivo

```mermaid
flowchart TD
    A[Texto de entrada] --> B[Pré-processamento]
    B --> C{Detector Regex<br/>dados pessoais explícitos}
    C -->|Encontrou match| D[Aplicar regras]
    C -->|Sem match| E[Vetorização TF-IDF]
    E --> F[Classificador Linear]
    F --> G[Predição ML]
    D --> H[Combinação<br/>Regex OR ML]
    G --> H
    H --> I{Predição final}
    I -->|Dados pessoais| J[🚫 Identificado]
    I -->|Sem dados pessoais| K[✅ Limpo]
    
    style A fill:#e1f5fe
    style I fill:#ffebee
    style J fill:#ffcdd2
    style K fill:#c8e6c9

```

