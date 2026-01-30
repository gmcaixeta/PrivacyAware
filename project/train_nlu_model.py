"""
Script de Treinamento do Detector de PII

Este script treina o modelo de detecção de PII e salva o modelo treinado.
Inclui geração de dados, treinamento, avaliação e salvamento.

Uso:
    python train_model.py
"""

from pathlib import Path
from datetime import datetime
import json
import random
from typing import Dict, List
import sys

# Importa o detector
from detector_nlu import TrainingDataGenerator, PIIDetector

# =============================================================================
# CONFIGURAÇÕES DE TREINAMENTO
# =============================================================================

# Diretórios
MODEL_OUTPUT_DIR = Path("./models/pii_nlu_model")
DATA_OUTPUT_DIR = Path("./data")
LOGS_OUTPUT_DIR = Path("./logs")

# Hiperparâmetros
CONFIG = {
    "n_public_examples": 1000,      # Número de exemplos públicos
    "n_pii_examples": 1000,         # Número de exemplos com PII
    "incluir_extenso": True,        # Incluir números por extenso
    "n_iterations": 30,             # Iterações de treino
    "dropout_rate": 0.5,            # Taxa de dropout
    "batch_size": 8,                # Tamanho do batch (fixo no modelo)
    "test_split": 0.2,              # % de dados para teste
    "random_seed": 42,              # Seed para reprodutibilidade
}

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def setup_directories():
    """Cria diretórios necessários."""
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("✓ Diretórios criados/verificados")

def split_train_test(examples: List, test_split: float = 0.2, seed: int = 42):
    """
    Divide dados em treino e teste, balanceando as classes.
    
    Args:
        examples: Lista de TrainingExample
        test_split: Proporção para teste (0.0 a 1.0)
        seed: Seed para reprodutibilidade
    
    Returns:
        (train_examples, test_examples)
    """
    random.seed(seed)
    
    # Separa por classe
    public_examples = [ex for ex in examples if ex.intent == "publico"]
    pii_examples = [ex for ex in examples if ex.intent == "tem_pii"]
    
    # Embaralha cada classe
    random.shuffle(public_examples)
    random.shuffle(pii_examples)
    
    # Calcula tamanhos
    n_test_public = int(len(public_examples) * test_split)
    n_test_pii = int(len(pii_examples) * test_split)
    
    # Divide
    test_public = public_examples[:n_test_public]
    train_public = public_examples[n_test_public:]
    
    test_pii = pii_examples[:n_test_pii]
    train_pii = pii_examples[n_test_pii:]
    
    # Combina e embaralha
    train_data = train_public + train_pii
    test_data = test_public + test_pii
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    print(f"\n📊 Divisão de dados:")
    print(f"   Treino: {len(train_data)} exemplos ({len(train_public)} públicos + {len(train_pii)} PII)")
    print(f"   Teste:  {len(test_data)} exemplos ({len(test_public)} públicos + {len(test_pii)} PII)")
    
    return train_data, test_data

def evaluate_model(detector: PIIDetector, test_examples: List) -> Dict:
    """
    Avalia o modelo no conjunto de teste.
    
    Returns:
        Dicionário com métricas de avaliação
    """
    print("\n🔍 Avaliando modelo no conjunto de teste...")
    
    results = {
        "total": len(test_examples),
        "correct": 0,
        "incorrect": 0,
        "by_intent": {
            "publico": {"total": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0},
            "tem_pii": {"total": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0}
        },
        "confusion_matrix": {
            "publico_as_publico": 0,
            "publico_as_pii": 0,
            "pii_as_publico": 0,
            "pii_as_pii": 0
        },
        "errors": []
    }
    
    for ex in test_examples:
        pred = detector.predict(ex.text, usar_contexto=True)
        predicted_intent = pred["intent"]
        true_intent = ex.intent
        
        # Contadores gerais
        results["by_intent"][true_intent]["total"] += 1
        
        if predicted_intent == true_intent:
            results["correct"] += 1
            results["by_intent"][true_intent]["correct"] += 1
            results["by_intent"][true_intent]["tp"] += 1
        else:
            results["incorrect"] += 1
            results["by_intent"][true_intent]["fn"] += 1
            results["by_intent"][predicted_intent]["fp"] += 1
            
            # Guarda erros para análise
            if len(results["errors"]) < 10:  # Limita a 10 exemplos
                results["errors"].append({
                    "text": ex.text,
                    "true_intent": true_intent,
                    "predicted_intent": predicted_intent,
                    "entities_detected": len(pred["entities"])
                })
        
        # Matriz de confusão
        if true_intent == "publico" and predicted_intent == "publico":
            results["confusion_matrix"]["publico_as_publico"] += 1
        elif true_intent == "publico" and predicted_intent == "tem_pii":
            results["confusion_matrix"]["publico_as_pii"] += 1
        elif true_intent == "tem_pii" and predicted_intent == "publico":
            results["confusion_matrix"]["pii_as_publico"] += 1
        elif true_intent == "tem_pii" and predicted_intent == "tem_pii":
            results["confusion_matrix"]["pii_as_pii"] += 1
    
    # Calcula métricas
    accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["accuracy"] = accuracy
    
    # Calcula precision, recall, F1 para cada classe
    for intent in ["publico", "tem_pii"]:
        data = results["by_intent"][intent]
        tp = data["tp"]
        fp = data["fp"]
        fn = data["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        data["precision"] = precision
        data["recall"] = recall
        data["f1_score"] = f1
    
    return results

def print_evaluation_report(results: Dict):
    """Imprime relatório de avaliação formatado."""
    print("\n" + "="*80)
    print("RELATÓRIO DE AVALIAÇÃO")
    print("="*80)
    
    print(f"\n📊 Acurácia Geral: {results['accuracy']:.2%}")
    print(f"   Corretas: {results['correct']}/{results['total']}")
    print(f"   Incorretas: {results['incorrect']}/{results['total']}")
    
    print("\n📈 Métricas por Classe:")
    print("-" * 80)
    
    for intent in ["publico", "tem_pii"]:
        data = results["by_intent"][intent]
        label = "Público" if intent == "publico" else "PII"
        
        print(f"\n{label}:")
        print(f"   Total de exemplos: {data['total']}")
        print(f"   Acurácia: {data['correct']/data['total']:.2%} ({data['correct']}/{data['total']})")
        print(f"   Precision: {data['precision']:.2%}")
        print(f"   Recall: {data['recall']:.2%}")
        print(f"   F1-Score: {data['f1_score']:.2%}")
    
    print("\n🎯 Matriz de Confusão:")
    print("-" * 80)
    cm = results["confusion_matrix"]
    print(f"                    Predito")
    print(f"                Público    PII")
    print(f"Real  Público     {cm['publico_as_publico']:4d}    {cm['publico_as_pii']:4d}")
    print(f"      PII         {cm['pii_as_publico']:4d}    {cm['pii_as_pii']:4d}")
    
    if results["errors"]:
        print("\n❌ Exemplos de Erros (primeiros 10):")
        print("-" * 80)
        for i, error in enumerate(results["errors"], 1):
            print(f"\n{i}. Texto: {error['text'][:60]}...")
            print(f"   Real: {error['true_intent']} | Predito: {error['predicted_intent']}")
            print(f"   Entidades detectadas: {error['entities_detected']}")

def save_training_log(config: Dict, results: Dict, timestamp: str):
    """Salva log do treinamento."""
    log_data = {
        "timestamp": timestamp,
        "config": config,
        "results": {
            "accuracy": results["accuracy"],
            "total_examples": results["total"],
            "correct": results["correct"],
            "incorrect": results["incorrect"],
            "by_intent": results["by_intent"],
            "confusion_matrix": results["confusion_matrix"]
        }
    }
    
    log_file = LOGS_OUTPUT_DIR / f"training_log_{timestamp}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Log salvo: {log_file}")

# =============================================================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# =============================================================================

def main():
    """Função principal de treinamento."""
    
    print("="*80)
    print("TREINAMENTO DO DETECTOR DE PII")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Setup
    print("\n1️⃣ Configurando ambiente...")
    setup_directories()
    random.seed(CONFIG["random_seed"])
    
    print("\nConfigurações:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # 2. Gera dados
    print("\n2️⃣ Gerando dados de treinamento...")
    generator = TrainingDataGenerator()
    
    all_examples = generator.generate_training_data(
        n_public=CONFIG["n_public_examples"],
        n_pii=CONFIG["n_pii_examples"],
        incluir_extenso=CONFIG["incluir_extenso"]
    )
    
    # 3. Divide treino/teste
    print("\n3️⃣ Dividindo dados em treino e teste...")
    train_data, test_data = split_train_test(
        all_examples,
        test_split=CONFIG["test_split"],
        seed=CONFIG["random_seed"]
    )
    
    # 4. Salva dados
    print("\n4️⃣ Salvando dados de treinamento...")
    
    train_file = DATA_OUTPUT_DIR / f"train_data_{timestamp}.json"
    test_file = DATA_OUTPUT_DIR / f"test_data_{timestamp}.json"
    
    generator.save_to_json(train_data, str(train_file))
    generator.save_to_json(test_data, str(test_file))
    
    print(f"✓ Treino salvo: {train_file}")
    print(f"✓ Teste salvo: {test_file}")
    
    # 5. Treina modelo
    print("\n5️⃣ Treinando modelo...")
    detector = PIIDetector()
    
    detector.train(
        training_examples=train_data,
        n_iter=CONFIG["n_iterations"],
        drop=CONFIG["dropout_rate"]
    )
    
    # 6. Avalia modelo
    print("\n6️⃣ Avaliando modelo...")
    results = evaluate_model(detector, test_data)
    print_evaluation_report(results)
    
    # 7. Salva modelo
    print("\n7️⃣ Salvando modelo treinado...")
    model_dir = MODEL_OUTPUT_DIR / timestamp
    detector.save(str(model_dir))
    
    # Salva também como "latest" para fácil acesso
    latest_dir = MODEL_OUTPUT_DIR / "latest"
    if latest_dir.exists():
        import shutil
        shutil.rmtree(latest_dir)
    detector.save(str(latest_dir))
    print(f"✓ Modelo salvo: {model_dir}")
    print(f"✓ Modelo salvo: {latest_dir} (versão mais recente)")
    
    # 8. Salva log
    print("\n8️⃣ Salvando log de treinamento...")
    save_training_log(CONFIG, results, timestamp)
    
    # 9. Salva configuração
    config_file = model_dir / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)
    print(f"✓ Configuração salva: {config_file}")
    
    # 10. Testes rápidos
    print("\n9️⃣ Testes rápidos...")
    print("-" * 80)
    
    test_cases = [
        ("Hospital Dr. João Silva", "publico"),
        ("Rua Maria Santos", "publico"),
        ("João Silva solicitou acesso", "tem_pii"),
        ("CPF: 123.456.789-00", "tem_pii"),
        ("Lei nº 12.527/2011", "publico"),
    ]
    
    print("\nExemplo | Esperado → Predito | Status")
    print("-" * 80)
    
    for text, expected in test_cases:
        pred = detector.predict(text, usar_contexto=True)
        status = "✓" if pred["intent"] == expected else "✗"
        print(f"{status} {text[:40]:40s} | {expected:8s} → {pred['intent']:8s}")
    
    # Resumo final
    print("\n" + "="*80)
    print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print(f"\n📁 Arquivos gerados:")
    print(f"   - Modelo: {model_dir}")
    print(f"   - Modelo (latest): {latest_dir}")
    print(f"   - Dados de treino: {train_file}")
    print(f"   - Dados de teste: {test_file}")
    print(f"   - Log: {LOGS_OUTPUT_DIR}/training_log_{timestamp}.json")
    print(f"\n📊 Desempenho:")
    print(f"   - Acurácia: {results['accuracy']:.2%}")
    print(f"   - F1-Score (Público): {results['by_intent']['publico']['f1_score']:.2%}")
    print(f"   - F1-Score (PII): {results['by_intent']['tem_pii']['f1_score']:.2%}")
    
    print("\n💡 Para usar o modelo treinado:")
    print("   from detector_nlu import PIIDetector")
    print(f"   detector = PIIDetector()")
    print(f"   detector.load('./models/pii_nlu_model/latest')")
    print(f"   result = detector.predict('seu texto aqui')")
    
    return detector, results

# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    try:
        detector, results = main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Treinamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)