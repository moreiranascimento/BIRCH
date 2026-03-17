# BIRCH — Comparativo de Clusterização

Este projeto compara o algoritmo **BIRCH** (scikit-learn) com uma implementação simples de **d-means** (simulação do GrouPlanner) para agrupar perfis de personalidade no espaço **Big Five (OCEAN)**.

## Objetivo

Avaliar, de forma prática, o comportamento dos dois métodos em:

- número de clusters gerados;
- tempo de execução;
- qualidade dos agrupamentos por métricas clássicas.

## O que o script faz

O arquivo principal [Birch.py](Birch.py):

1. Gera um dataset sintético com 10.000 usuários e 5 traços (OCEAN);
2. Executa o **BIRCH**;
3. Executa o **d-means**;
4. Calcula as métricas:
	 - `Silhouette Score`;
	 - `Davies-Bouldin`;
5. Mostra um resumo comparativo no terminal.

## Requisitos

- Python 3.10+
- Bibliotecas:
	- `numpy`
	- `scikit-learn`

## Como executar

Na raiz do projeto:

1. (Opcional) Criar e ativar ambiente virtual
2. Instalar dependências
3. Rodar o script

Exemplo:

- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install numpy scikit-learn`
- `python Birch.py`

## Saída esperada

Você verá no terminal uma tabela com:

- `Clusters Gerados`
- `Tempo de Execução`
- `Silhouette Score`
- `Davies-Bouldin`

## Estrutura

- [Birch.py](Birch.py): script principal com geração de dados, clusterização e métricas.
- [README.md](README.md): documentação do projeto.

## Autor

Repositório: https://github.com/moreiranascimento/BIRCH
