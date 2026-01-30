import io
import os
import zipfile
import panel as pn
import pandas as pd

from pathlib import Path
from typing import Optional

# Importa o detector NLU
from detector_nlu import PIIDetectorV2

pn.extension('tabulator', notifications=True)

# Caminho do modelo treinado
MODEL_PATH = Path("./models/pii_v2_model/latest")

# =============================================================================
# APP PANEL HOLOVIZ - DETECTOR DE PII COM NLU
# =============================================================================

class PIIDetectorApp:
    """App Panel para detecção de PII em CSV usando NLU."""
    
    def __init__(self):
        # Estado
        self.df_original: Optional[pd.DataFrame] = None
        self.df_result: Optional[pd.DataFrame] = None
        self.detector: Optional[PIIDetectorV2] = None
        self.is_detector_ready = False
        
        # Widgets
        self._create_widgets()
        
        # Layout
        self.template = self._create_template()
        
        # Auto-inicializa detector
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Inicializa o detector carregando o modelo NLU pré-treinado."""
        try:
            print("🚀 Iniciando carregamento do detector NLU...")
            pn.state.notifications.info('Carregando detector NLU...', duration=2000)
            
            self.detector_status.object = "⏳ **Status:** Carregando modelo NLU pré-treinado..."
            
            # Verifica se modelo existe
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Modelo não encontrado em {MODEL_PATH}!\n"
                    f"Execute 'python train_nlu_model.py' primeiro para treinar o modelo."
                )
            
            print(f"🧠 Carregando modelo NLU de: {MODEL_PATH}")
            
            # Carrega o detector NLU com modelo treinado
            self.detector = PIIDetectorV2()
            self.detector.load(str(MODEL_PATH))
            
            self.is_detector_ready = True
            
            # Atualiza UI
            self.detector_status.object = f"""
            ✅ **Status:** Detector NLU pronto!
            
            - **Pipeline:** SpaCy + Regex + NER
            - **Modelo:** Pré-treinado com dados reais
            - **Intenções:** publico / tem_pii
            - **Entidades:** PESSOA, CPF, EMAIL, TELEFONE
            - **Tempo de inicialização:** ~3-5 segundos
            """
            
            self.csv_upload.disabled = False
            
            print("✅ Detector NLU inicializado com sucesso!")
            pn.state.notifications.success(
                '✅ Detector NLU pronto! Modelo carregado.',
                duration=5000
            )
            
        except FileNotFoundError as e:
            error_msg = str(e)
            print(f"❌ ERRO: {error_msg}")
            pn.state.notifications.error(
                'Modelo não encontrado! Execute train_nlu_model.py primeiro.',
                duration=10000
            )
            self.detector_status.object = f"""
            ❌ **Status:** Modelo não encontrado
            
            **Como resolver:**
            1. Execute: `python train_nlu_model.py`
            2. Aguarde o treino (~2-5 minutos)
            3. Reinicie este app
            
            **Detalhes:** {error_msg}
            """
            
        except Exception as e:
            print(f"❌ Erro ao inicializar: {e}")
            import traceback
            traceback.print_exc()
            
            pn.state.notifications.error(f'Erro ao inicializar: {str(e)}', duration=5000)
            self.detector_status.object = f"❌ **Status:** Erro - {str(e)}"
    
    def _create_widgets(self):
        """Cria todos os widgets da interface."""
        
        # ===== SEÇÃO 1: STATUS =====
        self.title = pn.pane.Markdown(
            "# 🔍 Detector de PII - Acesso à Informação\n"
            "Sistema de classificação automática usando NLU (Natural Language Understanding)",
            sizing_mode='stretch_width'
        )
        
        self.detector_status = pn.pane.Markdown(
            "⏳ **Status:** Inicializando...",
            sizing_mode='stretch_width'
        )
        
        # ===== SEÇÃO 2: UPLOAD CSV =====
        self.csv_upload = pn.widgets.FileInput(
            name='📊 Upload CSV com Pedidos',
            accept='.csv',
            multiple=False,
            height=50,
            disabled=True
        )
        
        self.column_selector = pn.widgets.Select(
            name='Coluna com Texto',
            options=[],
            width=300,
            disabled=True
        )
        
        self.process_button = pn.widgets.Button(
            name='⚡ Processar CSV',
            button_type='success',
            width=200,
            disabled=True
        )
        self.process_button.on_click(self._process_csv)
        
        # ===== SEÇÃO 3: RESULTADOS =====
        self.progress = pn.indicators.Progress(
            name='Processamento',
            value=0,
            max=100,
            visible=False,
            sizing_mode='stretch_width'
        )
        
        self.stats_pane = pn.pane.Markdown(
            "",
            sizing_mode='stretch_width'
        )
        
        self.table_original = pn.widgets.Tabulator(
            pd.DataFrame(),
            name='📄 Dados Originais',
            disabled=True,
            page_size=10,
            sizing_mode='stretch_both',
            height=300,
        )
        
        self.table_result = pn.widgets.Tabulator(
            pd.DataFrame(),
            name='✅ Resultado Classificado',
            disabled=True,
            page_size=10,
            sizing_mode='stretch_both',
            height=300,
        )
        
        self.download_button = pn.widgets.FileDownload(
            callback=self._download_csv,
            filename='resultado_pii_classificado.csv',
            button_type='success',
            label='⬇️ Download Resultado (CSV)',
            width=250,
            visible=False
        )
        
        # ===== CALLBACKS =====
        self.csv_upload.param.watch(self._on_csv_upload, 'value')
        self.column_selector.param.watch(self._on_column_select, 'value')
    
    def _on_csv_upload(self, event):
        """Callback quando CSV é carregado."""
        try:
            if self.csv_upload.value is None:
                return
            
            # Carrega CSV
            csv_content = io.BytesIO(self.csv_upload.value)
            self.df_original = pd.read_csv(csv_content)
            
            # Atualiza seletor de colunas
            self.column_selector.options = list(self.df_original.columns)
            self.column_selector.disabled = False
            
            # Mostra preview
            self.table_original.value = self.df_original.head(100)
            
            pn.state.notifications.success(
                f'CSV carregado: {len(self.df_original):,} linhas, {len(self.df_original.columns)} colunas',
                duration=3000
            )
            
        except Exception as e:
            pn.state.notifications.error(f'Erro ao carregar CSV: {str(e)}', duration=5000)
    
    def _on_column_select(self, event):
        """Callback quando coluna é selecionada."""
        if self.column_selector.value:
            self.process_button.disabled = False
    
    def _process_csv(self, event):
        """Processa o CSV classificando com NLU."""
        try:
            if not self.is_detector_ready:
                pn.state.notifications.error('Detector não está pronto!', duration=3000)
                return
            
            if self.df_original is None:
                pn.state.notifications.error('Carregue um CSV primeiro!', duration=3000)
                return
            
            column = self.column_selector.value
            if not column:
                pn.state.notifications.error('Selecione uma coluna!', duration=3000)
                return
            
            # Prepara processamento
            self.progress.visible = True
            self.progress.value = 0
            self.process_button.disabled = True
            pn.state.notifications.info('Processando com NLU...', duration=2000)
            
            # Cria cópia do dataframe
            self.df_result = self.df_original.copy()
            
            # Processa linha por linha
            total = len(self.df_result)
            intents = []
            confidences = []
            entity_counts = []
            extractors_used = {'RegexEntityExtractor': 0, 'IntentEntityClassifier': 0}
            
            for idx, row in self.df_result.iterrows():
                text = str(row[column])
                
                # Classifica com NLU
                result = self.detector.predict(text)
                
                intent = result['intent']
                confidence = result['confidence']
                entities = result['entities']
                
                intents.append(intent)
                confidences.append(confidence)
                entity_counts.append(len(entities))
                
                # Rastreia extractors usados
                for entity in entities:
                    extractor = entity.get('extractor', 'unknown')
                    if extractor in extractors_used:
                        extractors_used[extractor] += 1
                
                # Atualiza progresso a cada 10 linhas
                if (idx + 1) % 10 == 0:
                    self.progress.value = int((idx + 1) / total * 100)
            
            # Adiciona colunas de resultado
            self.df_result['intent'] = intents
            self.df_result['confidence'] = confidences
            self.df_result['num_entities'] = entity_counts
            
            # Converte intent para binário (compatibilidade)
            self.df_result['tem_pii'] = [1 if i == 'tem_pii' else 0 for i in intents]
            
            # Calcula estatísticas
            n_pii = sum(1 for i in intents if i == 'tem_pii')
            n_publico = total - n_pii
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            self.stats_pane.object = f"""
            ## 📊 Estatísticas de Classificação
            
            ### Resultados Gerais
            - **Total de registros:** {total:,}
            - **Com PII (dados identificáveis):** {n_pii:,} ({100*n_pii/total:.1f}%)
            - **Públicos (sem PII):** {n_publico:,} ({100*n_publico/total:.1f}%)
            - **Confiança média:** {avg_confidence:.1%}
            
            ### Métodos de Extração de Entidades
            - **Regex (padrões):** {extractors_used['RegexEntityExtractor']:,} entidades
            - **NER (ML):** {extractors_used['IntentEntityClassifier']:,} entidades
            
            ### Distribuição de Entidades
            - **Registros com 0 entidades:** {sum(1 for c in entity_counts if c == 0):,}
            - **Registros com 1+ entidades:** {sum(1 for c in entity_counts if c > 0):,}
            - **Máximo de entidades em um registro:** {max(entity_counts) if entity_counts else 0}
            """
            
            # Atualiza tabela
            self.table_result.value = self.df_result
            
            # Mostra botão de download
            self.download_button.visible = True
            
            # Finaliza
            self.progress.visible = False
            self.process_button.disabled = False
            
            pn.state.notifications.success(
                f'✅ Classificação concluída! {n_pii:,} com PII, {n_publico:,} públicos',
                duration=5000
            )
            
        except Exception as e:
            self.progress.visible = False
            self.process_button.disabled = False
            pn.state.notifications.error(f'Erro ao processar: {str(e)}', duration=5000)
            print(f"Erro detalhado: {e}")
            import traceback
            traceback.print_exc()
    
    def _download_csv(self):
        """Gera CSV para download."""
        if self.df_result is None:
            return io.BytesIO()
        
        buffer = io.BytesIO()
        self.df_result.to_csv(buffer, index=False, encoding='utf-8-sig')
        buffer.seek(0)
        return buffer
    
    def _create_template(self):
        """Cria o template do app."""
        
        # Seção de status
        status_section = pn.Card(
            pn.Column(
                pn.pane.Markdown("### 1️⃣ Status do Sistema"),
                self.detector_status,
            ),
            title='Status',
            collapsible=False,
        )
        
        # Seção de upload
        upload_section = pn.Card(
            pn.Column(
                pn.pane.Markdown("### 2️⃣ Upload e Processamento"),
                self.csv_upload,
                pn.Row(
                    self.column_selector,
                    self.process_button,
                ),
                self.progress,
            ),
            title='Upload CSV',
            collapsible=False,
        )
        
        # Seção de resultados
        results_section = pn.Card(
            pn.Column(
                pn.pane.Markdown("### 3️⃣ Resultados da Classificação"),
                self.stats_pane,
                pn.Tabs(
                    ('Original', self.table_original),
                    ('Classificado', self.table_result),
                ),
                self.download_button,
            ),
            title='Resultados',
            collapsible=False,
        )
        
        # Layout principal
        template = pn.template.FastListTemplate(
            title='🔍 Detector de PII - Sistema NLU',
            sidebar=[
                pn.pane.Markdown("""
                ## 📖 Como usar
                
                ### ⚡ Sistema NLU
                - Pipeline de processamento modular
                - Classificação de intenção (público/privado)
                - Extração de entidades (PII)
                - Modelo pré-treinado
                
                ### 📋 Passos
                
                1. **Aguarde** detector ficar pronto
                2. **Upload** do CSV com pedidos
                3. **Selecione** coluna com texto
                4. **Clique** em "Processar CSV"
                5. **Download** resultado classificado
                
                ## 🎯 Classificação
                
                ### Intent: "tem_pii" (Privado)
                Quando detecta:
                - ✅ Nome completo (primeiro + sobrenome)
                - ✅ CPF (válido ou não)
                - ✅ RG, CNH, outros documentos
                - ✅ Email pessoal
                - ✅ Telefone
                - ✅ Endereço completo
                
                ### Intent: "publico" (Público)
                Quando encontra:
                - ✅ Apenas primeiro nome
                - ✅ Nome de lei/decreto/órgão
                - ✅ Pedido genérico sem identificação
                - ✅ Contexto institucional
                
                ## ⚙️ Pipeline NLU
                
                ### 1. SpacyNLP
                - Carrega modelo pt_core_news_sm
                - Tokenização e features
                
                ### 2. RegexEntityExtractor
                - Extrai CPF, Email, Telefone
                - Padrões regex otimizados
                - Alta precisão
                
                ### 3. IntentEntityClassifier
                - NER para detecção de PESSOA
                - Treinado com dados reais
                - Machine Learning (spaCy)
                
                ## 📊 Colunas do Resultado
                
                - **intent**: publico ou tem_pii
                - **confidence**: 0.0 a 1.0
                - **num_entities**: quantidade de PII detectado
                - **tem_pii**: 0 (público) ou 1 (privado)
                
                ## 🚀 Performance
                
                - Inicialização: ~3-5 segundos
                - Processamento: ~100-200 registros/seg
                - Modelo pré-treinado (sem treino no startup)
                """),
            ],
            main=[
                self.title,
                status_section,
                upload_section,
                results_section,
            ],
            accent_base_color='#2e7d32',
            header_background='#1b5e20',
        )
        
        return template
    
    def show(self):
        """Exibe o app."""
        return self.template.servable()


# =============================================================================
# INICIALIZAÇÃO PARA DEPLOY
# =============================================================================

def create_pii_app():
    """Factory function para criar instância do app."""


    # Descompactar models.zip se a pasta models/ não existir
    if not os.path.exists('models') and os.path.exists('models.zip'):
        with zipfile.ZipFile('models.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        print("✅ Modelo descompactado com sucesso!")
        
    app = PIIDetectorApp()
    return app.show()


# Configuração do servidor Panel
if __name__ == "__main__":
    pn.serve(
        {
            "/pii-detector": create_pii_app,
        },
        port=8081,
        allow_websocket_origin=["*"],
        address="0.0.0.0",
        show=False,
        unused_session_lifetime=120000,  # 2 minutos (modelo demora ~30s para carregar)
        check_unused_sessions=30000,      # Checa a cada 30s
        reuse_sessions=False,
        admin=False,
        titles={
            "": "Home",
            "pii-detector": "🔍 Detector de PII - NLU",
        }
    )
