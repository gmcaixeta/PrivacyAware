# Projeto PrivacyAware

---

## 🎯 Objetivo do projeto:

Classificar pedidos de acesso à informação em:

* **1 = contém dados pessoais**
* **0 = não contém**

Maximizando **F1-score**, com **ênfase em recall** (minimizar falsos negativos).

---

### ▶️ Video Explicativo (clicar na imagem abaixo)

[![Video Thumbnail Alt Text](https://img.youtube.com/vi/5qS9KVnrAiI/0.jpg)](https://youtu.be/5qS9KVnrAiI)



---
## 🎯 Objetivo do pipeline

* ✔️ Maximizar recall 
* ✔️ Simples de explicar 
* ✔️ Fácil de rodar e reproduzir
* ✔️ Não depender de LLM, GPU ou APIs externas
* ✔️ Robustez > sofisticação

---


## 🧠 Estratégia geral

**Pipeline híbrido**:

1. **Regras determinísticas (regex)** → capturam casos óbvios
2. **Modelo NER (Named Entity Recognition)** → pega padrões não explícitos
3. **OR lógico final** → se *qualquer* um detectar → classifica como positivo

```mermaid
%%{init: {
  "theme": "default",
  "themeVariables": {
    "fontSize": "10px",
    "nodePadding": "4",
    "nodeBorder": "0.5px"
  },
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 30
  }
}}%%
flowchart LR
    A[Texto] --> B[Pré-processo]
    B --> C{Regex}
    C -->|+| F[Dados Pessoais]
    C -->|N| D[NER]
    D --> E[Classificador]
    E -->|+| F
    E -->|-| G[Dados públicos]
    
    style A fill:#e1f5fe
    style F fill:#ffcdd2
    style G fill:#c8e6c9

```


## 🧰 Tutorial de Uso


---

## 🧾 1. Preparar o arquivo CSV

<img src="project/imgs/formato_csv.png" width="300">



* O arquivo deve ter **apenas uma coluna**
* **Cada linha** deve conter **um texto a ser validado**
* A **primeira linha** deve ser o **nome da coluna** (exemplo: `mensagens`)


Exemplo conceitual:

```
mensagens
Olá, tudo bem?
Meu CPF é 123.456.789-00
Confirmando reunião amanhã
```

---

## 🌐 2. Acessar o dashboard

* Abra o navegador
* Acesse:
  **[http://localhost:8081](http://localhost:8081)**

---

## 📂 3. Carregar o arquivo CSV


![btn_csv](project/imgs/selecionar_btn.png)

* Clique no botão **“Abrir arquivos”**
* Selecione o arquivo CSV preparado anteriormente

![arquivo_csv](project/imgs/selecionar_arquivo.png)


---

## ✅ 4. Verificar a coluna de textos

* Confirme se a **coluna que contém os textos** está selecionada corretamente
* Caso exista mais de uma coluna (cenários futuros), selecione a correta

---

## ⚙️ 5. Processar o CSV


* Clique no botão **“Processar CSV”**
* Aguarde o processamento dos textos

![inf](project/imgs/realizar_inferencia.png)

---

## ⬇️ 6. Baixar o resultado

* Após o processamento, clique no **botão de download**
* Um **novo arquivo CSV** será gerado

![result](project/imgs/resultado_csv.png)


---

## 📊 7. Entendendo o resultado

O arquivo gerado terá uma nova coluna chamada:

**`tem_pii`**

Valores possíveis:

* **1** → o texto contém **dados pessoais (PII)**
* **0** → o texto **não contém dados pessoais**

Isso permite validar automaticamente grandes volumes de mensagens de forma simples e reproduzível ✅
