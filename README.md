# machine-learning-notebooks

AI研修のハンズオン用リポジトリです。  

## 注意事項
このリポジトリは`hands-on`ディレクトリの各チャプター内にある`*.ipynb`ファイルに対して、GCPのJupyterLab環境内で動作させることを想定しています。  
Vertex AIやGCS等を使用している部分は、研修以外の環境でやる場合、適宜変更する必要があるので注意してください。 

`03_document_retrieval_by_LLM_and_RAG`のハンズオンで使用する画像データは研修用GCSに配置しています
研修外で利用する場合は別途用意するか、`use_image = False`で実行してください

## ハンズオン目次
- 00_intro_jupyter_notebook
- 01_image_classification
  - 01ex_pruning
- 02_transfer_learning
  - 02ex_parameter_tuning
- 03_document_retrieval_by_LLM_and_RAG
- 04_deploy_and_serving
