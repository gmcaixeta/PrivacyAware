"""
Detector de PII com sistema de palavras-gatilho para contexto público.

NOVIDADE: Usa palavras-gatilho para distinguir nomes em contexto público
(ex: "Hospital João Silva" = público) de nomes pessoais (ex: "João Silva solicitou" = PII)
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
# PALAVRAS-GATILHO PARA CONTEXTO PÚBLICO
# =============================================================================

PALAVRAS_GATILHO_PUBLICO = {
    # Saúde
    'hospital', 'maternidade', 'upa', 'ubs', 'posto de saúde', 'clínica', 
    'policlínica', 'ambulatório', 'pronto-socorro', 'pronto socorro',
    
    # Educação
    'escola', 'colégio', 'universidade', 'faculdade', 'instituto', 'fundação',
    'creche', 'centro educacional', 'campus',
    
    # Cultura
    'biblioteca', 'museu', 'arquivo', 'teatro', 'centro cultural', 'galeria',
    'auditório', 'casa de cultura', 'memorial',
    
    # Vias públicas
    'rua', 'r.', 'r', 'avenida', 'av.', 'av', 'alameda', 'travessa', 'praça',
    'largo', 'rodovia', 'estrada', 'via', 'viaduto', 'ponte', 'túnel', 
    'rotatória', 'passarela', 'viela', 'beco',
    
    # Edificações públicas
    'edifício', 'prédio', 'palácio', 'fórum', 'tribunal', 'cartório', 
    'delegacia', 'batalhão', 'quartel', 'prefeitura', 'câmara', 'assembleia',
    'secretaria', 'ministério', 'sede', 'anexo',
    
    # Atos normativos
    'lei', 'decreto', 'portaria', 'resolução', 'instrução normativa', 
    'ato normativo', 'estatuto', 'regimento', 'medida provisória', 'emenda',
    'norma', 'regulamento', 'diretriz',
    
    # Órgãos públicos
    'ministério', 'secretaria', 'autarquia', 'fundação pública', 'empresa pública',
    'poder executivo', 'poder legislativo', 'poder judiciário', 'tribunal',
    'stf', 'stj', 'tcu', 'cgu', 'mp', 'conselho', 'comissão',
    
    # Documentos administrativos
    'processo', 'procedimento', 'protocolo', 'ofício', 'memorando', 'despacho',
    'relatório', 'parecer', 'nota técnica', 'edital', 'licitação', 
    'contrato administrativo', 'termo', 'ata',
    
    # Empresas (sufixos)
    's.a.', 'sa', 's.a', 'ltda', 'ltda.', 'eireli', 'me', 'mei', 'companhia',
    'empresa', 'grupo', 'holding', 'associação', 'cooperativa', 'sindicato',
    'federação', 'confederação', 'ong', 'oscip',
    
    # Cargos públicos
    'presidente', 'diretor', 'secretário', 'ministro', 'prefeito', 'governador',
    'senador', 'deputado', 'vereador', 'relator', 'servidor', 'funcionário',
    'gestor', 'coordenador', 'chefe', 'superintendente', 'procurador',
    
    # Programas e projetos
    'programa', 'projeto', 'plano', 'ação', 'iniciativa', 'campanha', 'operação',
    'prêmio', 'medalha', 'comenda', 'ordem', 'bolsa', 'auxílio',
    
    # Dados e estatísticas
    'dados', 'estatísticas', 'indicadores', 'relatório anual', 'balanço',
    'demonstrativo', 'série histórica', 'painel', 'dashboard', 'censo',
    
    # Religiosos
    'igreja', 'catedral', 'basílica', 'capela', 'mosteiro', 'convento', 
    'templo', 'santuário', 'paróquia', 'diocese',
    
    # Localização
    'bairro', 'distrito', 'região', 'zona', 'setor', 'quadra', 'lote',
    'km', 'cep', 'endereço', 'logradouro',
}

# Converte para lowercase e cria versões com/sem pontuação
PALAVRAS_GATILHO_PUBLICO_NORMALIZED = set()
for palavra in PALAVRAS_GATILHO_PUBLICO:
    PALAVRAS_GATILHO_PUBLICO_NORMALIZED.add(palavra.lower())
    # Remove pontos
    PALAVRAS_GATILHO_PUBLICO_NORMALIZED.add(palavra.lower().replace('.', ''))

# =============================================================================
# FUNÇÕES DE DETECÇÃO DE CONTEXTO
# =============================================================================

def tem_contexto_publico(texto: str, pos_inicio: int, pos_fim: int, janela: int = 50) -> bool:
    """
    Verifica se um nome está em contexto público.
    
    Args:
        texto: Texto completo
        pos_inicio: Posição inicial do nome
        pos_fim: Posição final do nome
        janela: Tamanho da janela de busca (caracteres antes/depois)
    
    Returns:
        True se o nome está em contexto público (não é PII)
    """
    # Extrai contexto ao redor do nome
    inicio_contexto = max(0, pos_inicio - janela)
    fim_contexto = min(len(texto), pos_fim + janela)
    contexto = texto[inicio_contexto:fim_contexto].lower()
    
    # Busca por palavras-gatilho
    for palavra_gatilho in PALAVRAS_GATILHO_PUBLICO_NORMALIZED:
        if palavra_gatilho in contexto:
            return True
    
    return False

def extrair_contexto_nome(texto: str, nome: str) -> Dict:
    """
    Extrai informações de contexto de um nome no texto.
    
    Returns:
        {
            'tem_contexto_publico': bool,
            'palavras_gatilho_encontradas': List[str],
            'contexto': str
        }
    """
    import re
    
    # Encontra todas as ocorrências do nome
    pattern = re.escape(nome)
    matches = list(re.finditer(pattern, texto, re.IGNORECASE))
    
    if not matches:
        return {
            'tem_contexto_publico': False,
            'palavras_gatilho_encontradas': [],
            'contexto': ''
        }
    
    # Analisa primeira ocorrência
    match = matches[0]
    pos_inicio = match.start()
    pos_fim = match.end()
    
    # Contexto expandido
    janela = 100
    inicio_contexto = max(0, pos_inicio - janela)
    fim_contexto = min(len(texto), pos_fim + janela)
    contexto = texto[inicio_contexto:fim_contexto]
    contexto_lower = contexto.lower()
    
    # Busca palavras-gatilho
    palavras_encontradas = []
    for palavra in PALAVRAS_GATILHO_PUBLICO_NORMALIZED:
        if palavra in contexto_lower:
            palavras_encontradas.append(palavra)
    
    return {
        'tem_contexto_publico': len(palavras_encontradas) > 0,
        'palavras_gatilho_encontradas': palavras_encontradas,
        'contexto': contexto.strip()
    }

# =============================================================================
# MAPEAMENTO DE NÚMEROS POR EXTENSO
# =============================================================================

NUMEROS_EXTENSO = {
    'zero': '0', 'um': '1', 'dois': '2', 'três': '3', 'tres': '3',
    'quatro': '4', 'cinco': '5', 'seis': '6', 'sete': '7',
    'oito': '8', 'nove': '9', 'dez': '10'
}

def normalizar_numero_extenso(texto: str) -> str:
    """Converte números por extenso para dígitos."""
    palavras = texto.lower().split()
    digitos = []
    
    for palavra in palavras:
        palavra_limpa = palavra.strip('.,;:!?')
        if palavra_limpa in NUMEROS_EXTENSO:
            digitos.append(NUMEROS_EXTENSO[palavra_limpa])
    
    return ''.join(digitos) if digitos else texto

def detectar_numeros_extenso(texto: str) -> List[Dict]:
    """Detecta sequências de números escritos por extenso."""
    matches = []
    palavras_numero = '|'.join(NUMEROS_EXTENSO.keys())
    
    pattern = rf'\b(?:{palavras_numero})(?:\s+(?:{palavras_numero})){{2,}}\b'
    
    for match in re.finditer(pattern, texto, re.IGNORECASE):
        texto_extenso = match.group(0)
        valor_numerico = normalizar_numero_extenso(texto_extenso)
        
        if len(valor_numerico) >= 6:
            matches.append({
                'start': match.start(),
                'end': match.end(),
                'texto_original': texto_extenso,
                'valor_numerico': valor_numerico
            })
    
    return matches

# =============================================================================
# ESTRUTURA DE DADOS
# =============================================================================

@dataclass
class TrainingExample:
    """Exemplo de treino."""
    text: str
    intent: str
    entities: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "intent": self.intent,
            "entities": self.entities
        }

# =============================================================================
# GERADOR DINÂMICO DE DADOS PÚBLICOS
# =============================================================================

class PublicDataGenerator:
    """Gera variações dinâmicas de dados públicos."""
    
    ORGAOS = [
        "Secretaria Municipal de Saúde", "Secretaria de Educação",
        "Ministério da Justiça", "Ministério da Economia",
        "Governo Federal", "Prefeitura Municipal",
        "Câmara dos Deputados", "Senado Federal",
        "STF", "STJ", "Tribunal de Contas",
        "Controladoria Geral", "Advocacia Geral da União",
        "Defensoria Pública", "Ministério Público",
    ]
    
    ACOES = [
        "informou", "publicou", "divulgou", "anunciou", "comunicou",
        "notificou", "respondeu", "deferiu", "indeferiu", "arquivou",
        "protocolou", "recebeu", "aprovou", "rejeitou",
    ]
    
    TIPOS_PROCESSO = [
        "Processo SEI", "Processo Administrativo", "Processo Judicial",
        "Procedimento", "Protocolo", "Requerimento",
        "Ofício", "Memorando", "Despacho", "Parecer",
    ]
    
    TIPOS_NORMA = [
        "Lei", "Decreto", "Portaria", "Resolução",
        "Instrução Normativa", "Medida Provisória", "Emenda Constitucional",
    ]
    
    TEMAS_GENERICOS = [
        "informações sobre licitação", "dados sobre contratos",
        "relatórios financeiros", "processos administrativos",
        "obras públicas", "arrecadação municipal",
        "execução orçamentária", "estatísticas de atendimento",
        "dados de transparência", "prestação de contas",
        "folha de pagamento", "compras governamentais",
    ]
    
    EMPRESAS = [
        "Petrobras S.A.", "Banco do Brasil", "Caixa Econômica Federal",
        "BNDES", "Eletrobras", "Correios",
        "Sabesp", "Cemig", "Copel",
    ]
    
    # Usa palavras-gatilho para gerar exemplos com contexto
    PALAVRAS_GATILHO_LISTA = list(PALAVRAS_GATILHO_PUBLICO)
    NOMES_HOMENAGEM = list(FIRST_NAMES)[:100]
    SOBRENOMES_HOMENAGEM = list(SIR_NAMES)[:100]
    
    def __init__(self):
        pass
    
    def _gerar_numero_processo(self) -> str:
        """Gera número de processo realista."""
        formatos = [
            "{:05d}-{:08d}/{:04d}-{:02d}",
            "{:05d}.{:06d}/{:04d}-{:02d}",
            "{:04d}{:010d}",
        ]
        
        formato = random.choice(formatos)
        ano = random.randint(2015, 2025)
        
        if "{:04d}" in formato and formato.count("{") == 4:
            return formato.format(
                random.randint(0, 99999),
                random.randint(0, 99999999),
                ano,
                random.randint(0, 99)
            )
        elif "{:05d}.{:06d}" in formato:
            return formato.format(
                random.randint(0, 99999),
                random.randint(0, 999999),
                ano,
                random.randint(0, 99)
            )
        else:
            return formato.format(
                random.randint(0, 9999),
                random.randint(0, 9999999999)
            )
    
    def _gerar_numero_lei(self) -> str:
        """Gera número de lei/decreto."""
        ano = random.randint(1990, 2025)
        numero = random.randint(1, 99999)
        
        formatos = [
            f"nº {numero}/{ano}",
            f"n° {numero:05d}/{ano}",
            f"{numero}/{ano}",
        ]
        
        return random.choice(formatos)
    
    def _gerar_nome_homenagem(self) -> str:
        """Gera nome completo para homenagem."""
        primeiro = random.choice(self.NOMES_HOMENAGEM)
        sobrenome = random.choice(self.SOBRENOMES_HOMENAGEM)
        
        if random.random() < 0.4:
            titulos = ["Dr.", "Profª", "Prof.", "Eng.", "Cel.", "Gen.", "Min."]
            return f"{random.choice(titulos)} {primeiro} {sobrenome}"
        
        return f"{primeiro} {sobrenome}"
    
    def gerar_exemplo_publico(self) -> str:
        """Gera um exemplo de dado público aleatório."""
        
        categorias = [
            self._gerar_orgao_acao,
            self._gerar_processo,
            self._gerar_norma,
            self._gerar_pedido_generico,
            self._gerar_empresa,
            self._gerar_local_com_gatilho,  # NOVO: usa palavras-gatilho
            self._gerar_lei_homenagem,
            self._gerar_cargo_generico,
        ]
        
        gerador = random.choice(categorias)
        return gerador()
    
    def _gerar_orgao_acao(self) -> str:
        orgao = random.choice(self.ORGAOS)
        acao = random.choice(self.ACOES)
        return f"{orgao} {acao}"
    
    def _gerar_processo(self) -> str:
        tipo = random.choice(self.TIPOS_PROCESSO)
        numero = self._gerar_numero_processo()
        return f"{tipo} {numero}"
    
    def _gerar_norma(self) -> str:
        tipo = random.choice(self.TIPOS_NORMA)
        numero = self._gerar_numero_lei()
        return f"{tipo} {numero}"
    
    def _gerar_pedido_generico(self) -> str:
        verbos = ["Solicitação de", "Pedido de", "Requisição de", "Consulta sobre"]
        tema = random.choice(self.TEMAS_GENERICOS)
        return f"{random.choice(verbos)} {tema}"
    
    def _gerar_empresa(self) -> str:
        empresa = random.choice(self.EMPRESAS)
        acao = random.choice(self.ACOES)
        return f"{empresa} {acao}"
    
    def _gerar_local_com_gatilho(self) -> str:
        """
        NOVO: Gera exemplos usando palavras-gatilho + nomes.
        Ex: "Hospital Dr. João Silva", "Rua Maria Santos"
        """
        # Escolhe palavra-gatilho de locais
        gatilhos_locais = [
            'hospital', 'upa', 'ubs', 'escola', 'colégio', 'universidade',
            'biblioteca', 'museu', 'teatro', 'rua', 'avenida', 'praça',
            'edifício', 'fórum', 'delegacia', 'igreja', 'bairro'
        ]
        
        gatilho = random.choice(gatilhos_locais).title()
        nome = self._gerar_nome_homenagem()
        
        return f"{gatilho} {nome}"
    
    def _gerar_lei_homenagem(self) -> str:
        nome = self._gerar_nome_homenagem()
        temas = [
            "proteção à infância", "direitos humanos", "acesso à informação",
            "proteção ao consumidor", "meio ambiente", "saúde pública", "educação",
        ]
        return f"Lei {nome}, {random.choice(temas)}"
    
    def _gerar_cargo_generico(self) -> str:
        cargos = ["O Diretor", "O Secretário", "A Ministra", "O Presidente", 
                  "O Coordenador", "A Chefe", "O Relator"]
        acoes = ["informou em reunião", "declarou em coletiva", "assinou despacho",
                 "votou favoravelmente", "apresentou relatório"]
        return f"{random.choice(cargos)} {random.choice(acoes)}"

# =============================================================================
# GERADOR DE DADOS DE TREINAMENTO
# =============================================================================

class TrainingDataGenerator:
    """Gera dados de treinamento."""
    
    PII_TEMPLATES_WITH_ENTITIES = [
        {"template": "{pessoa} solicitou acesso à informação",
         "entities": [{"entity": "PESSOA", "role": "solicitante"}]},
        {"template": "Requerente: {pessoa}",
         "entities": [{"entity": "PESSOA", "role": "requerente"}]},
        {"template": "Cidadão {pessoa} requisitou documentos",
         "entities": [{"entity": "PESSOA", "role": "cidadao"}]},
        {"template": "Titular dos dados: {pessoa}",
         "entities": [{"entity": "PESSOA", "role": "titular"}]},
        
        # Documentos
        {"template": "CPF: {cpf}",
         "entities": [{"entity": "CPF", "role": "documento"}]},
        {"template": "RG: {rg}",
         "entities": [{"entity": "RG", "role": "documento"}]},
        {"template": "CIN: {cin}",
         "entities": [{"entity": "CIN", "role": "documento"}]},
        {"template": "Passaporte: {passaporte}",
         "entities": [{"entity": "PASSAPORTE", "role": "documento"}]},
        {"template": "Email: {email}",
         "entities": [{"entity": "EMAIL", "role": "contato"}]},
        {"template": "Telefone: {telefone}",
         "entities": [{"entity": "TELEFONE", "role": "contato"}]},
        
        # Combinações
        {"template": "{pessoa}, CPF {cpf}",
         "entities": [{"entity": "PESSOA", "role": "solicitante"},
                      {"entity": "CPF", "role": "documento"}]},
        {"template": "{pessoa} - RG {rg}",
         "entities": [{"entity": "PESSOA", "role": "solicitante"},
                      {"entity": "RG", "role": "documento"}]},
    ]
    
    def __init__(self, first_names: List[str] = None, last_names: List[str] = None):
        self.first_names = first_names if first_names is not None else list(FIRST_NAMES)
        self.last_names = last_names if last_names is not None else list(SIR_NAMES)
        self.public_generator = PublicDataGenerator()
        
        print(f"✓ Gerador inicializado com {len(self.first_names):,} primeiros nomes")
        print(f"✓ Sistema de palavras-gatilho ativado ({len(PALAVRAS_GATILHO_PUBLICO_NORMALIZED)} palavras)")
    
    def _generate_pessoa(self) -> str:
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        if random.random() < 0.3:
            middle = random.choice(self.first_names)
            return f"{first} {middle} {last}"
        return f"{first} {last}"
    
    def _generate_cpf(self) -> str:
        formats = ["{}{}{}.{}{}{}.{}{}{}-{}{}", "{}{}{}{}{}{}{}{}{}{}{}"]
        digits = [str(random.randint(0, 9)) for _ in range(11)]
        return random.choice(formats).format(*digits)
    
    def _generate_rg(self) -> str:
        formats = ["{}{}.{}{}{}.{}{}{}-{}", "{}{}{}{}{}{}{}{}{}", "{}{}.{}{}{}.{}{}{}"]
        digits = [str(random.randint(0, 9)) for _ in range(9)]
        return random.choice(formats).format(*digits)
    
    def _generate_cin(self) -> str:
        formats = ["{}{}{}{} {}{}{}{} {}{}{}{}", "{}{}{}{}{}{}{}{}{}{}{}{}"]
        digits = [str(random.randint(0, 9)) for _ in range(12)]
        return random.choice(formats).format(*digits)
    
    def _generate_passaporte(self) -> str:
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
        digits = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        return f"{letters}{digits}"
    
    def _generate_email(self) -> str:
        first = random.choice(self.first_names).lower()
        domains = ["gmail.com", "outlook.com", "yahoo.com.br", "hotmail.com"]
        return f"{first}{random.randint(1, 999)}@{random.choice(domains)}"
    
    def _generate_telefone(self) -> str:
        formats = ["({}{}) {}{}{}{}-{}{}{}{}", "({}{}) {}{}{}{}{}{}{}{}", "{}{}{}{}{}{}{}{}{}{}"]
        digits = [str(random.randint(0, 9)) for _ in range(10)]
        return random.choice(formats).format(*digits)
    
    def _numero_para_extenso(self, numero: str) -> str:
        mapa = {'0': 'zero', '1': 'um', '2': 'dois', '3': 'três', '4': 'quatro',
                '5': 'cinco', '6': 'seis', '7': 'sete', '8': 'oito', '9': 'nove'}
        return ' '.join(mapa.get(d, d) for d in numero if d.isdigit())
    
    def generate_training_data(self, n_public: int = 500, n_pii: int = 500, 
                              incluir_extenso: bool = True) -> List[TrainingExample]:
        """Gera dados de treinamento."""
        examples = []
        
        # Públicos
        print(f"Gerando {n_public} exemplos públicos...")
        for i in range(n_public):
            text = self.public_generator.gerar_exemplo_publico()
            examples.append(TrainingExample(text=text, intent="publico", entities=[]))
            if (i + 1) % 100 == 0:
                print(f"  ✓ {i + 1}/{n_public}")
        
        # PII
        print(f"Gerando {n_pii} exemplos com PII...")
        for i in range(n_pii):
            template_data = random.choice(self.PII_TEMPLATES_WITH_ENTITIES)
            template = template_data["template"]
            entity_specs = template_data["entities"]
            
            text = template
            entities = []
            
            for entity_spec in entity_specs:
                entity_type = entity_spec["entity"]
                role = entity_spec.get("role", "")
                
                if entity_type == "PESSOA":
                    value = self._generate_pessoa()
                    placeholder = "{pessoa}"
                elif entity_type == "CPF":
                    value = self._generate_cpf()
                    placeholder = "{cpf}"
                elif entity_type == "RG":
                    value = self._generate_rg()
                    placeholder = "{rg}"
                elif entity_type == "CIN":
                    value = self._generate_cin()
                    placeholder = "{cin}"
                elif entity_type == "PASSAPORTE":
                    value = self._generate_passaporte()
                    placeholder = "{passaporte}"
                elif entity_type == "EMAIL":
                    value = self._generate_email()
                    placeholder = "{email}"
                elif entity_type == "TELEFONE":
                    value = self._generate_telefone()
                    placeholder = "{telefone}"
                else:
                    continue
                
                start = text.find(placeholder)
                if start != -1:
                    text = text.replace(placeholder, value, 1)
                    end = start + len(value)
                    entities.append({
                        "start": start, "end": end, "value": value,
                        "entity": entity_type, "role": role
                    })
            
            examples.append(TrainingExample(text=text, intent="tem_pii", entities=entities))
            if (i + 1) % 100 == 0:
                print(f"  ✓ {i + 1}/{n_pii}")
        
        # Extenso
        if incluir_extenso:
            n_extenso = min(100, n_pii // 5)
            print(f"Gerando {n_extenso} exemplos com números por extenso...")
            
            for _ in range(n_extenso):
                num_digitos = random.choice([9, 10, 11])
                numero = ''.join([str(random.randint(0, 9)) for _ in range(num_digitos)])
                valor_extenso = self._numero_para_extenso(numero)
                
                templates = ["CPF: {v}", "RG {v}", "Documento {v}"]
                text = random.choice(templates).replace("{v}", valor_extenso)
                
                entity_type = "CPF" if num_digitos == 11 else ("RG" if num_digitos == 9 else "CIN")
                
                examples.append(TrainingExample(
                    text=text, intent="tem_pii",
                    entities=[{"start": text.find(valor_extenso), 
                              "end": text.find(valor_extenso) + len(valor_extenso),
                              "value": valor_extenso, "entity": entity_type, 
                              "role": "documento_extenso"}]
                ))
        
        random.shuffle(examples)
        
        print(f"\n✓ Total: {len(examples)} exemplos")
        print(f"  - Públicos: {sum(1 for e in examples if e.intent == 'publico')}")
        print(f"  - PII: {sum(1 for e in examples if e.intent == 'tem_pii')}")
        
        return examples
    
    def save_to_json(self, examples: List[TrainingExample], output_file: str):
        training_data = {
            "version": "1.0",
            "language": "pt",
            "data": {"common_examples": [ex.to_dict() for ex in examples]}
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Dados salvos: {output_file}")

# =============================================================================
# DETECTOR COM SISTEMA DE CONTEXTO
# =============================================================================

class PIIDetector:
    """Detector de PII com análise de contexto."""
    
    def __init__(self, model_name: str = "pt_core_news_sm"):
        try:
            self.nlp = spacy.load(model_name)
            print(f"✓ Modelo {model_name} carregado")
        except:
            print(f"⚠ Modelo não encontrado. Criando vazio...")
            self.nlp = spacy.blank("pt")
        
        self.regex_patterns = {
            "CPF": r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',
            "RG": r'\b\d{2}\.?\d{3}\.?\d{3}-?\d{1}\b|\b\d{9}\b|\b\d{2}\.?\d{3}\.?\d{3}\b',
            "CIN": r'\b\d{4}\s?\d{4}\s?\d{4}\b|\b\d{12}\b',
            "PASSAPORTE": r'\b[A-Z]{2}\d{6}\b',
            "EMAIL": r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            "TELEFONE": r'\b\(?\d{2}\)?\s?\d{4,5}[-\s]?\d{4}\b',
        }
        
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner")
        else:
            self.ner = self.nlp.get_pipe("ner")
        
        self.ner.add_label("PESSOA")
        self._is_trained = False
    
    def train(self, training_examples: List[TrainingExample], n_iter: int = 30, drop: float = 0.5):
        print(f"\n🧠 Treinando ({n_iter} iterações)...")
        
        spacy_examples = []
        for ex in training_examples:
            doc = self.nlp.make_doc(ex.text)
            entities = [(ent["start"], ent["end"], "PESSOA") 
                       for ent in ex.entities if ent["entity"] == "PESSOA"]
            spacy_examples.append(Example.from_dict(doc, {"entities": entities}))
        
        other_pipes = [p for p in self.nlp.pipe_names if p != "ner"]
        
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            
            for iteration in range(n_iter):
                random.shuffle(spacy_examples)
                losses = {}
                
                for i in range(0, len(spacy_examples), 8):
                    batch = spacy_examples[i:i+8]
                    self.nlp.update(batch, drop=drop, losses=losses, sgd=optimizer)
                
                if iteration % 5 == 0:
                    print(f"  Iteração {iteration}/{n_iter} | Loss: {losses.get('ner', 0):.2f}")
        
        self._is_trained = True
        print("✓ Treinamento concluído!")
    
    def predict(self, text: str, usar_contexto: bool = True) -> Dict:
        """
        Predição com análise de contexto.
        
        Args:
            text: Texto para análise
            usar_contexto: Se True, filtra nomes em contexto público
        """
        entities = []
        
        # 1. Regex
        for entity_type, pattern in self.regex_patterns.items():
            for match in re.finditer(pattern, text):
                entities.append({
                    "start": match.start(), "end": match.end(),
                    "value": match.group(0), "entity": entity_type,
                    "extractor": "RegexEntityExtractor"
                })
        
        # 2. Números extenso
        numeros_extenso = detectar_numeros_extenso(text)
        for num_ext in numeros_extenso:
            tamanho = len(num_ext['valor_numerico'])
            entity_type = "CPF" if tamanho == 11 else ("CIN" if tamanho == 12 else "RG")
            
            entities.append({
                "start": num_ext['start'], "end": num_ext['end'],
                "value": num_ext['texto_original'],
                "valor_numerico": num_ext['valor_numerico'],
                "entity": entity_type,
                "extractor": "NumeroExtensoExtractor"
            })
        
        # 3. NER com filtro de contexto
        if self._is_trained:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PESSOA" and len(ent.text.split()) >= 2:
                    # NOVO: Verifica contexto
                    if usar_contexto:
                        contexto_info = extrair_contexto_nome(text, ent.text)
                        
                        if contexto_info['tem_contexto_publico']:
                            # Nome em contexto público - NÃO é PII
                            continue
                    
                    entities.append({
                        "start": ent.start_char, "end": ent.end_char,
                        "value": ent.text, "entity": "PESSOA",
                        "extractor": "IntentEntityClassifier"
                    })
        
        has_pii = len(entities) > 0
        intent = "tem_pii" if has_pii else "publico"
        confidence = 0.95 if has_pii else 0.85
        
        return {
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "text": text
        }
    
    def save(self, model_path: str):
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(model_path)
        print(f"✓ Modelo salvo: {model_path}")
    
    def load(self, model_path: str):
        self.nlp = spacy.load(model_path)
        self.ner = self.nlp.get_pipe("ner")
        self._is_trained = True
        print(f"✓ Modelo carregado: {model_path}")

# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("DETECTOR DE PII COM SISTEMA DE PALAVRAS-GATILHO")
    print("="*80)
    
    # Demonstração
    print("\n📊 DEMONSTRAÇÃO: Exemplos com palavras-gatilho")
    print("-" * 80)
    
    demo_gen = PublicDataGenerator()
    for i in range(15):
        exemplo = demo_gen.gerar_exemplo_publico()
        print(f"{i+1:2d}. {exemplo}")
    
    print("\n" + "="*80)
    
    # Treina
    print("\n1️⃣ Gerando dados de treinamento...")
    generator = TrainingDataGenerator()
    training_data = generator.generate_training_data(n_public=500, n_pii=500)
    
    print("\n2️⃣ Salvando...")
    generator.save_to_json(training_data, "training_data_final.json")
    
    print("\n3️⃣ Treinando...")
    detector = PIIDetector()
    detector.train(training_data, n_iter=20)
    
    print("\n4️⃣ Testando com palavras-gatilho...")
    test_cases = [
        # Contexto público (NÃO deve detectar como PII)
        "Hospital Dr. João Silva",
        "Rua Maria Santos",
        "Lei Carlos Alberto, proteção ao consumidor",
        "Escola Municipal Pedro Álvares",
        "Universidade Federal de São Paulo",
        
        # PII (DEVE detectar)
        "João Silva solicitou acesso",
        "Requerente: Maria Santos",
        "CPF: 123.456.789-00",
        "RG um dois três quatro cinco seis sete oito nove",
    ]
    
    print("\n" + "="*80)
    print("RESULTADOS")
    print("="*80)
    
    for text in test_cases:
        result = detector.predict(text, usar_contexto=True)
        contexto = "🟢 PÚBLICO" if result['intent'] == 'publico' else "🔴 PII"
        
        print(f"\n{contexto} | {text}")
        if result['entities']:
            for ent in result['entities']:
                print(f"         └─ {ent['entity']}: {ent['value']}")
    
    print("\n" + "="*80)
    print("✅ CONCLUÍDO!")
    print("="*80)