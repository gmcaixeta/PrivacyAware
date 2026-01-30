"""
Detector de PII - VERSÃO CORRIGIDA
==================================

MUDANÇA FUNDAMENTAL:
- Separação clara entre EXTRAÇÃO (NER) e CLASSIFICAÇÃO (decisão de PII)
- PII só existe quando há PAPEL INDIVIDUALIZANTE
- Nome ≠ PII por default

Baseado na análise do edital:
"Solicitações do cidadão que não permitam a identificação de uma pessoa natural 
podem ser consideradas públicas."
"""

from typing import List, Tuple, Dict, Optional, Set
import re
import spacy
from spacy.training import Example
import random
from dataclasses import dataclass, field
from pathlib import Path
import json

# Importa dicionários de nomes
from first_names_static import FIRST_NAMES
from sir_name_static import SIR_NAMES

# =============================================================================
# CAMADA 1: PALAVRAS-GATILHO PARA CONTEXTO PÚBLICO (Exclusão Explícita)
# =============================================================================

CONTEXTO_NAO_PII = {
    # Denominações institucionais
    'hospital', 'maternidade', 'upa', 'ubs', 'posto de saúde', 'clínica', 
    'policlínica', 'ambulatório', 'pronto-socorro', 'pronto socorro',
    
    'escola', 'colégio', 'universidade', 'faculdade', 'instituto', 'fundação',
    'creche', 'centro educacional', 'campus',
    
    'biblioteca', 'museu', 'arquivo', 'teatro', 'centro cultural', 'galeria',
    'auditório', 'casa de cultura', 'memorial',
    
    # Vias públicas e topônimos
    'rua', 'r.', 'r', 'avenida', 'av.', 'av', 'alameda', 'travessa', 'praça',
    'largo', 'rodovia', 'estrada', 'via', 'viaduto', 'ponte', 'túnel', 
    'rotatória', 'passarela', 'viela', 'beco', 'bairro', 'distrito',
    
    # Edificações públicas
    'edifício', 'prédio', 'palácio', 'fórum', 'tribunal', 'cartório', 
    'delegacia', 'batalhão', 'quartel', 'prefeitura', 'câmara', 'assembleia',
    
    # Atos normativos e homenagens
    'lei', 'decreto', 'portaria', 'resolução', 'instrução normativa',
    'programa', 'projeto', 'plano', 'prêmio', 'medalha', 'comenda',
    'relatório', 'parecer', 'nota técnica',
    
    # Empresas (sufixos)
    's.a.', 'sa', 's.a', 'ltda', 'ltda.', 'eireli', 'me', 'mei', 'companhia',
    'empresa', 'grupo', 'holding', 'associação', 'cooperativa',
}

# Normaliza (lowercase, sem pontuação)
CONTEXTO_NAO_PII_NORMALIZED = set()
for palavra in CONTEXTO_NAO_PII:
    CONTEXTO_NAO_PII_NORMALIZED.add(palavra.lower())
    CONTEXTO_NAO_PII_NORMALIZED.add(palavra.lower().replace('.', ''))

# =============================================================================
# CAMADA 2: PAPÉIS INDIVIDUALIZANTES (Decisão de PII)
# =============================================================================

# Estes padrões indicam que o nome identifica uma PESSOA NATURAL
PAPEIS_INDIVIDUALIZANTES = {
    # Ações individuais (verbos)
    'verbos': {
        'solicitou', 'requereu', 'requisitou', 'pediu', 'demandou',
        'protocolou', 'apresentou', 'encaminhou', 'enviou',
        'compareceu', 'assinou', 'autorizou', 'declarou',
        'reclamou', 'denunciou', 'reportou',
    },
    
    # Papéis nominais
    'papeis': {
        'solicitante', 'requerente', 'requisitante', 'demandante',
        'cidadão', 'cidadã', 'munícipe', 'contribuinte',
        'titular', 'responsável', 'representante', 'interessado',
        'reclamante', 'denunciante', 'autor', 'peticionário',
        'morador', 'moradora', 'residente', 'paciente',
    },
    
    # Contextos de identificação
    'contextos_id': {
        'nome:', 'nome completo:', 'identificação:', 'titular:',
        'dados do solicitante', 'dados do requerente',
        'qualidade de', 'na qualidade de',  # "na qualidade de representante"
    }
}

# =============================================================================
# FUNÇÕES DE CLASSIFICAÇÃO SEMÂNTICA
# =============================================================================

def extrair_janela_contexto(texto: str, pos_inicio: int, pos_fim: int, 
                           janela_antes: int = 50, janela_depois: int = 50) -> str:
    """Extrai janela de contexto ao redor de uma entidade."""
    inicio = max(0, pos_inicio - janela_antes)
    fim = min(len(texto), pos_fim + janela_depois)
    return texto[inicio:fim].lower()

def tem_papel_individualizante(texto: str, nome: str, pos_inicio: int, pos_fim: int) -> Dict:
    """
    Verifica se o nome está associado a um papel individualizante.
    
    Returns:
        {
            'tem_papel': bool,
            'tipo': str (verbo/papel/contexto_id),
            'evidencia': str
        }
    """
    # Extrai contexto expandido
    contexto = extrair_janela_contexto(texto, pos_inicio, pos_fim, 100, 100)
    
    # 1. Verifica verbos de ação individual
    for verbo in PAPEIS_INDIVIDUALIZANTES['verbos']:
        if verbo in contexto:
            return {
                'tem_papel': True,
                'tipo': 'verbo_individual',
                'evidencia': verbo
            }
    
    # 2. Verifica papéis nominais
    for papel in PAPEIS_INDIVIDUALIZANTES['papeis']:
        if papel in contexto:
            return {
                'tem_papel': True,
                'tipo': 'papel_nominal',
                'evidencia': papel
            }
    
    # 3. Verifica contextos de identificação
    for contexto_id in PAPEIS_INDIVIDUALIZANTES['contextos_id']:
        if contexto_id in contexto:
            return {
                'tem_papel': True,
                'tipo': 'contexto_identificacao',
                'evidencia': contexto_id
            }
    
    return {
        'tem_papel': False,
        'tipo': None,
        'evidencia': None
    }

def tem_dado_associado(texto: str, nome: str, pos_inicio: int, pos_fim: int) -> bool:
    """
    Verifica se há dados pessoais (CPF, RG, email, telefone) próximos ao nome.
    Isso é forte indicador de pessoa natural.
    """
    contexto = extrair_janela_contexto(texto, pos_inicio, pos_fim, 150, 150)
    
    # Padrões simplificados
    padroes = [
        r'\bcpf\b',
        r'\brg\b',
        r'\bemail\b',
        r'\btelefone\b',
        r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}',  # CPF
        r'\d{2}\.?\d{3}\.?\d{3}',  # RG
        r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}',  # Email
    ]
    
    for padrao in padroes:
        if re.search(padrao, contexto):
            return True
    
    return False

def tem_contexto_exclusao(texto: str, nome: str, pos_inicio: int, pos_fim: int) -> Dict:
    """
    Verifica se o nome está em contexto de EXCLUSÃO (não é PII).
    
    Returns:
        {
            'excluir': bool,
            'motivo': str,
            'palavra_gatilho': str
        }
    """
    # Janela mais curta para contextos de exclusão (precisam estar MUITO próximos)
    contexto = extrair_janela_contexto(texto, pos_inicio, pos_fim, 30, 30)
    
    # 1. Busca palavras-gatilho de denominação institucional
    for palavra in CONTEXTO_NAO_PII_NORMALIZED:
        if palavra in contexto:
            return {
                'excluir': True,
                'motivo': 'denominacao_institucional',
                'palavra_gatilho': palavra
            }
    
    # 2. Padrões específicos de exclusão
    padroes_exclusao = [
        # Lei + Nome
        (r'\blei\s+[a-záàâãéèêíïóôõöúçñ]+\s+[a-záàâãéèêíïóôõöúçñ]+', 'lei_homenagem'),
        
        # Prêmio/Projeto + Nome
        (r'\b(prêmio|projeto|programa)\s+[a-záàâãéèêíïóôõöúçñ]+', 'homenagem'),
        
        # Relatório + Nome
        (r'\brelatório\s+[a-záàâãéèêíïóôõöúçñ]+', 'relatorio_nomeado'),
    ]
    
    for padrao, motivo in padroes_exclusao:
        if re.search(padrao, contexto, re.IGNORECASE):
            return {
                'excluir': True,
                'motivo': motivo,
                'palavra_gatilho': padrao
            }
    
    return {
        'excluir': False,
        'motivo': None,
        'palavra_gatilho': None
    }

def nome_identifica_pessoa_natural(texto: str, nome: str, 
                                   pos_inicio: int, pos_fim: int) -> Dict:
    """
    DECISÃO CENTRAL: Este nome identifica uma pessoa natural?
    
    Lógica:
    1. Se tem contexto de exclusão → NÃO é PII
    2. Se tem papel individualizante → É PII
    3. Se tem dado associado → É PII
    4. Caso contrário → NÃO é PII
    
    Returns:
        {
            'e_pii': bool,
            'razao': str,
            'detalhes': dict
        }
    """
    # 1. EXCLUSÃO tem prioridade
    exclusao = tem_contexto_exclusao(texto, nome, pos_inicio, pos_fim)
    if exclusao['excluir']:
        return {
            'e_pii': False,
            'razao': 'contexto_exclusao',
            'detalhes': exclusao
        }
    
    # 2. Verifica papel individualizante
    papel = tem_papel_individualizante(texto, nome, pos_inicio, pos_fim)
    if papel['tem_papel']:
        return {
            'e_pii': True,
            'razao': 'papel_individualizante',
            'detalhes': papel
        }
    
    # 3. Verifica dados associados
    if tem_dado_associado(texto, nome, pos_inicio, pos_fim):
        return {
            'e_pii': True,
            'razao': 'dados_associados',
            'detalhes': {'tipo': 'documento_ou_contato'}
        }
    
    # 4. Default: nome sem contexto individualizante NÃO é PII
    return {
        'e_pii': False,
        'razao': 'sem_papel_individualizante',
        'detalhes': {}
    }

# =============================================================================
# GERADOR DE DADOS - VERSÃO CORRIGIDA
# =============================================================================

class TrainingDataGeneratorV2:
    """
    Gerador de dados de treinamento com foco em classificação semântica.
    
    Diferenças da versão anterior:
    1. Exemplos negativos explícitos (nomes que NÃO são PII)
    2. Mais variação em papéis individualizantes
    3. Menos dependência de palavras-gatilho isoladas
    """
    
    def __init__(self):
        self.first_names = list(FIRST_NAMES)
        self.last_names = list(SIR_NAMES)
    
    def _gerar_nome(self, incluir_titulo: bool = False) -> str:
        """Gera nome completo."""
        primeiro = random.choice(self.first_names)
        sobrenome = random.choice(self.last_names)
        
        if incluir_titulo and random.random() < 0.3:
            titulos = ['Dr.', 'Dra.', 'Prof.', 'Profª', 'Eng.']
            return f"{random.choice(titulos)} {primeiro} {sobrenome}"
        
        if random.random() < 0.3:
            meio = random.choice(self.first_names)
            return f"{primeiro} {meio} {sobrenome}"
        
        return f"{primeiro} {sobrenome}"
    
    def _gerar_cpf(self) -> str:
        """Gera CPF."""
        formatos = ["{}{}{}.{}{}{}.{}{}{}-{}{}", "{}{}{}{}{}{}{}{}{}{}{}"]
        digitos = [str(random.randint(0, 9)) for _ in range(11)]
        return random.choice(formatos).format(*digitos)
    
    def _gerar_email(self) -> str:
        """Gera email."""
        primeiro = random.choice(self.first_names).lower()
        dominios = ["gmail.com", "outlook.com", "yahoo.com.br", "hotmail.com"]
        return f"{primeiro}{random.randint(1, 999)}@{random.choice(dominios)}"
    
    def _gerar_empresa(self) -> str:
        """Gera nome de empresa realista."""
        nomes = [
            "BIOCASA COMERCIO DE MATERIAL FISIOTERAPICO LTDA",
            "CONSTRUTORA SILVA E SANTOS S.A.",
            "TRANSPORTADORA RÁPIDA LTDA",
            "COMERCIAL ATACADISTA DO NORDESTE",
            "SERVIÇOS DE ENGENHARIA XYZ EIRELI",
        ]
        return random.choice(nomes)
    
    def gerar_exemplos_pii(self, n: int = 500) -> List:
        """
        Gera exemplos com PII (papel individualizante presente).
        """
        exemplos = []
        
        templates = [
            # Papéis explícitos
            ("Requerente: {nome}", "papel_nominal"),
            ("Solicitante: {nome}", "papel_nominal"),
            ("Cidadão {nome} solicitou", "papel_nominal"),
            ("Titular dos dados: {nome}", "papel_nominal"),
            ("{nome}, CPF {cpf}", "dados_associados"),
            ("{nome}, email: {email}", "dados_associados"),
            
            # Verbos de ação individual
            ("{nome} solicitou acesso à informação", "verbo_individual"),
            ("{nome} requereu documentos", "verbo_individual"),
            ("{nome} protocolou pedido", "verbo_individual"),
            ("{nome} compareceu para atendimento", "verbo_individual"),
            
            # Contextos de identificação
            ("Nome: {nome}", "contexto_id"),
            ("Identificação: {nome}", "contexto_id"),
            ("Na qualidade de representante da {empresa}, solicito...", "representante_empresa"),
            
            # Casos mais complexos (reais)
            ("Prezados, na qualidade de representante da {empresa}, {nome} solicita informações.", "caso_complexo"),
        ]
        
        for _ in range(n):
            template, tipo = random.choice(templates)
            
            texto = template.format(
                nome=self._gerar_nome(),
                cpf=self._gerar_cpf(),
                email=self._gerar_email(),
                empresa=self._gerar_empresa()
            )
            
            exemplos.append({
                'text': texto,
                'intent': 'tem_pii',
                'tipo_pii': tipo
            })
        
        return exemplos
    
    def gerar_exemplos_publicos(self, n: int = 500) -> List:
        """
        Gera exemplos PÚBLICOS (sem papel individualizante).
        
        IMPORTANTE: Inclui nomes completos que NÃO são PII.
        """
        exemplos = []
        
        templates_institucionais = [
            # Denominações institucionais
            ("Hospital {nome}", "instituicao"),
            ("Escola Municipal {nome}", "instituicao"),
            ("Biblioteca {nome}", "instituicao"),
            ("Teatro {nome}", "instituicao"),
            ("Rua {nome}", "toponimo"),
            ("Avenida {nome}", "toponimo"),
            ("Praça {nome}", "toponimo"),
            
            # Homenagens e atos normativos
            ("Lei {nome}", "lei_homenagem"),
            ("Decreto {nome}", "lei_homenagem"),
            ("Prêmio {nome} de Direitos Humanos", "premio"),
            ("Programa {nome}", "programa"),
            ("Projeto {nome}", "projeto"),
            ("Relatório {nome}", "relatorio_nomeado"),
            
            # Empresas (sem representante identificado)
            ("{empresa} solicitou informações", "empresa_juridica"),
            ("A empresa {empresa} protocolou", "empresa_juridica"),
            
            # Processos e documentos genéricos
            ("Processo {numero}", "processo"),
            ("Protocolo {numero}", "processo"),
            ("Solicitação de dados sobre licitação", "pedido_generico"),
            ("Informações sobre contrato público", "pedido_generico"),
        ]
        
        for _ in range(n):
            template, tipo = random.choice(templates_institucionais)
            
            texto = template.format(
                nome=self._gerar_nome(incluir_titulo=True),
                empresa=self._gerar_empresa(),
                numero=f"{random.randint(1000, 9999)}-{random.randint(100, 999)}/{random.randint(2020, 2025)}"
            )
            
            exemplos.append({
                'text': texto,
                'intent': 'publico',
                'tipo': tipo
            })
        
        return exemplos
    
    def gerar_dataset_completo(self, n_pii: int = 500, n_publico: int = 500) -> List:
        """Gera dataset balanceado."""
        print(f"Gerando {n_pii} exemplos com PII...")
        pii = self.gerar_exemplos_pii(n_pii)
        
        print(f"Gerando {n_publico} exemplos públicos...")
        publico = self.gerar_exemplos_publicos(n_publico)
        
        todos = pii + publico
        random.shuffle(todos)
        
        print(f"\n✓ Total: {len(todos)} exemplos")
        print(f"  - Com PII: {len(pii)}")
        print(f"  - Públicos: {len(publico)}")
        
        return todos
    
    def salvar_json(self, exemplos: List[Dict], output_file: str):
        """Salva dataset em JSON."""
        training_data = {
            "version": "2.0",
            "language": "pt",
            "metadata": {
                "modelo": "classificacao_semantica",
                "criterio": "papel_individualizante"
            },
            "data": {"common_examples": exemplos}
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Dados salvos: {output_file}")

# =============================================================================
# DETECTOR V2 - COM CLASSIFICAÇÃO SEMÂNTICA
# =============================================================================

class PIIDetectorV2:
    """
    Detector de PII com classificação semântica em 3 camadas.
    """
    
    def __init__(self, model_name: str = "pt_core_news_sm"):
        try:
            self.nlp = spacy.load(model_name)
            print(f"✓ Modelo {model_name} carregado")
        except:
            print(f"⚠ Modelo não encontrado. Criando vazio...")
            self.nlp = spacy.blank("pt")
        
        # Regex para camada 1
        self.regex_patterns = {
            "CPF": r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',
            "RG": r'\b\d{2}\.?\d{3}\.?\d{3}-?\d{1}\b|\b\d{9}\b',
            "EMAIL": r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            "TELEFONE": r'\b\(?\d{2}\)?\s?\d{4,5}[-\s]?\d{4}\b',
        }
        
        # NER
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner")
        else:
            self.ner = self.nlp.get_pipe("ner")
        
        self.ner.add_label("PESSOA")
        self._is_trained = False
    
    def train(self, training_data: List[Dict], n_iter: int = 30):
        """Treina apenas o NER (não a classificação semântica)."""
        print(f"\n🧠 Treinando NER ({n_iter} iterações)...")
        
        # Prepara exemplos apenas com entidades PESSOA
        spacy_examples = []
        
        for ex in training_data:
            if ex['intent'] == 'tem_pii':
                doc = self.nlp.make_doc(ex['text'])
                
                # Encontra nomes no texto (heurística simples: 2+ palavras capitalizadas)
                palavras = ex['text'].split()
                entities = []
                
                i = 0
                while i < len(palavras):
                    if palavras[i][0].isupper():
                        # Tenta formar nome de 2-3 palavras
                        nome_candidato = []
                        j = i
                        while j < len(palavras) and palavras[j][0].isupper() and len(nome_candidato) < 3:
                            nome_candidato.append(palavras[j])
                            j += 1
                        
                        if len(nome_candidato) >= 2:
                            nome = ' '.join(nome_candidato)
                            pos = ex['text'].find(nome)
                            if pos != -1:
                                entities.append((pos, pos + len(nome), "PESSOA"))
                            i = j
                        else:
                            i += 1
                    else:
                        i += 1
                
                if entities:
                    spacy_examples.append(Example.from_dict(doc, {"entities": entities}))
        
        print(f"  Exemplos com entidades: {len(spacy_examples)}")
        
        # Treina
        other_pipes = [p for p in self.nlp.pipe_names if p != "ner"]
        
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            
            for iteration in range(n_iter):
                random.shuffle(spacy_examples)
                losses = {}
                
                for i in range(0, len(spacy_examples), 8):
                    batch = spacy_examples[i:i+8]
                    self.nlp.update(batch, drop=0.5, losses=losses, sgd=optimizer)
                
                if iteration % 5 == 0:
                    print(f"  Iteração {iteration}/{n_iter} | Loss: {losses.get('ner', 0):.2f}")
        
        self._is_trained = True
        print("✓ Treinamento de NER concluído!")
    
    def predict(self, text: str, verbose: bool = False) -> Dict:
        """
        Predição em 3 camadas:
        1. Extração (Regex + NER)
        2. Classificação semântica (decide se é PII)
        3. Regras de exclusão
        """
        entities_pii = []
        entities_nao_pii = []
        
        # CAMADA 1: EXTRAÇÃO - Regex (sempre PII)
        for entity_type, pattern in self.regex_patterns.items():
            for match in re.finditer(pattern, text):
                entities_pii.append({
                    "start": match.start(),
                    "end": match.end(),
                    "value": match.group(0),
                    "entity": entity_type,
                    "extractor": "RegexEntityExtractor",
                    "razao": "documento_ou_contato"
                })
        
        # CAMADA 2 + 3: NER + CLASSIFICAÇÃO SEMÂNTICA
        if self._is_trained:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PESSOA" and len(ent.text.split()) >= 2:
                    # DECISÃO: Este nome identifica pessoa natural?
                    decisao = nome_identifica_pessoa_natural(
                        text, ent.text, ent.start_char, ent.end_char
                    )
                    
                    if decisao['e_pii']:
                        entities_pii.append({
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "value": ent.text,
                            "entity": "PESSOA",
                            "extractor": "SemanticClassifier",
                            "razao": decisao['razao'],
                            "detalhes": decisao['detalhes']
                        })
                    else:
                        if verbose:
                            entities_nao_pii.append({
                                "value": ent.text,
                                "razao": decisao['razao'],
                                "detalhes": decisao['detalhes']
                            })
        
        has_pii = len(entities_pii) > 0
        intent = "tem_pii" if has_pii else "publico"
        
        result = {
            "intent": intent,
            "confidence": 0.9 if has_pii else 0.85,
            "entities": entities_pii,
            "text": text
        }
        
        if verbose:
            result["entities_excluidas"] = entities_nao_pii
        
        return result
    
    def save(self, model_path: str):
        """Salva modelo."""
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(model_path)
        print(f"✓ Modelo salvo: {model_path}")
    
    def load(self, model_path: str):
        """Carrega modelo."""
        self.nlp = spacy.load(model_path)
        self.ner = self.nlp.get_pipe("ner")
        self._is_trained = True
        print(f"✓ Modelo carregado: {model_path}")

# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("DETECTOR DE PII V2 - COM CLASSIFICAÇÃO SEMÂNTICA")
    print("="*80)
    
    # 1. Gera dados
    print("\n1️⃣ Gerando dados de treinamento...")
    generator = TrainingDataGeneratorV2()
    training_data = generator.gerar_dataset_completo(n_pii=500, n_publico=500)
    
    print("\n2️⃣ Salvando...")
    generator.salvar_json(training_data, "training_data_v2.json")
    
    # 2. Treina
    print("\n3️⃣ Treinando...")
    detector = PIIDetectorV2()
    detector.train(training_data, n_iter=20)
    
    # 3. Testa
    print("\n4️⃣ Testando casos críticos...")
    test_cases = [
        # DEVEM SER PÚBLICOS (sem papel individualizante)
        "Hospital Dr. João Silva",
        "Rua Maria Santos",
        "Lei Carlos Alberto",
        "Prêmio João da Silva de Direitos Humanos",
        "Relatório Pedro Álvares",
        "BIOCASA COMERCIO DE MATERIAL FISIOTERAPICO LTDA solicita informações",
        
        # DEVEM SER PII (papel individualizante presente)
        "João Silva solicitou acesso",
        "Requerente: Maria Santos",
        "Na qualidade de representante da BIOCASA, João Silva solicita",
        "CPF: 123.456.789-00",
        "Cidadão Pedro Oliveira requereu documentos",
    ]
    
    print("\n" + "="*80)
    print("RESULTADOS")
    print("="*80)
    
    for text in test_cases:
        result = detector.predict(text, verbose=True)
        
        if result['intent'] == 'publico':
            status = "✅ PÚBLICO"
            cor = "\033[92m"  # Verde
        else:
            status = "⚠️  PII"
            cor = "\033[91m"  # Vermelho
        
        print(f"\n{cor}{status}\033[0m | {text}")
        
        if result['entities']:
            for ent in result['entities']:
                print(f"  🔴 {ent['entity']}: {ent['value']}")
                print(f"     Razão: {ent['razao']}")
        
        if 'entities_excluidas' in result and result['entities_excluidas']:
            for ent in result['entities_excluidas']:
                print(f"  🟢 EXCLUÍDO: {ent['value']}")
                print(f"     Razão: {ent['razao']}")
    
    print("\n" + "="*80)
    print("✅ TESTE CONCLUÍDO!")
    print("="*80)
    
    # 4. Testa caso real problemático
    print("\n5️⃣ Testando caso real do edital...")
    caso_real = """Prezados, boa noite. Na qualidade de representante da BIOCASA COMERCIO DE MATERIAL FISIOTERÁPICO LTDA - ME, solicito, gentilmente, o envio dos Processos Administrativos, extratos, bem como quaisquer outras informações relativas às Certidões de Dívida Ativa nº 1000258954 e 0002574863. Agradeço a disponibilidade e aguardo o retorno. Atenciosamente,"""
    
    result = detector.predict(caso_real, verbose=True)
    
    print(f"\nTexto: {caso_real[:100]}...")
    print(f"\nClassificação: {result['intent'].upper()}")
    print(f"Confiança: {result['confidence']:.2f}")
    
    if result['entities']:
        print("\nPII detectado:")
        for ent in result['entities']:
            print(f"  - {ent['entity']}: {ent['value']}")
            print(f"    Razão: {ent['razao']}")
    
    if 'entities_excluidas' in result and result['entities_excluidas']:
        print("\nEntidades excluídas (não são PII):")
        for ent in result['entities_excluidas']:
            print(f"  - {ent['value']}")
            print(f"    Razão: {ent['razao']}")