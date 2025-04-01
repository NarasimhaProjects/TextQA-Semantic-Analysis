{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ec8b452c-b02b-4204-a3b7-dfc78b3cb6e8",
   "metadata": {},
   "source": [
    "### TextQA/Semantic Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "76a4b028-cf65-4ca6-98b5-2418a22e7f0f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting sentence-transformers\n",
      "  Downloading sentence_transformers-4.0.1-py3-none-any.whl.metadata (13 kB)\n",
      "Requirement already satisfied: transformers<5.0.0,>=4.41.0 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from sentence-transformers) (4.45.2)\n",
      "Requirement already satisfied: tqdm in c:\\users\\admin\\anaconda3\\lib\\site-packages (from sentence-transformers) (4.66.4)\n",
      "Requirement already satisfied: torch>=1.11.0 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from sentence-transformers) (2.3.1)\n",
      "Requirement already satisfied: scikit-learn in c:\\users\\admin\\anaconda3\\lib\\site-packages (from sentence-transformers) (1.4.2)\n",
      "Requirement already satisfied: scipy in c:\\users\\admin\\anaconda3\\lib\\site-packages (from sentence-transformers) (1.13.1)\n",
      "Requirement already satisfied: huggingface-hub>=0.20.0 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from sentence-transformers) (0.24.6)\n",
      "Requirement already satisfied: Pillow in c:\\users\\admin\\anaconda3\\lib\\site-packages (from sentence-transformers) (10.3.0)\n",
      "Requirement already satisfied: typing_extensions>=4.5.0 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from sentence-transformers) (4.12.2)\n",
      "Requirement already satisfied: filelock in c:\\users\\admin\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (3.13.1)\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2024.3.1)\n",
      "Requirement already satisfied: packaging>=20.9 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (23.2)\n",
      "Requirement already satisfied: pyyaml>=5.1 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (6.0.1)\n",
      "Requirement already satisfied: requests in c:\\users\\admin\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2.32.2)\n",
      "Requirement already satisfied: sympy in c:\\users\\admin\\anaconda3\\lib\\site-packages (from torch>=1.11.0->sentence-transformers) (1.12)\n",
      "Requirement already satisfied: networkx in c:\\users\\admin\\anaconda3\\lib\\site-packages (from torch>=1.11.0->sentence-transformers) (3.2.1)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from torch>=1.11.0->sentence-transformers) (3.1.4)\n",
      "Requirement already satisfied: colorama in c:\\users\\admin\\anaconda3\\lib\\site-packages (from tqdm->sentence-transformers) (0.4.6)\n",
      "Requirement already satisfied: numpy>=1.17 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (1.26.4)\n",
      "Requirement already satisfied: regex!=2019.12.17 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (2023.10.3)\n",
      "Requirement already satisfied: tokenizers<0.21,>=0.20 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.20.1)\n",
      "Requirement already satisfied: safetensors>=0.4.1 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.4.5)\n",
      "Requirement already satisfied: joblib>=1.2.0 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from scikit-learn->sentence-transformers) (1.4.2)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from scikit-learn->sentence-transformers) (2.2.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from jinja2->torch>=1.11.0->sentence-transformers) (2.1.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (2.2.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (2024.12.14)\n",
      "Requirement already satisfied: mpmath>=0.19 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from sympy->torch>=1.11.0->sentence-transformers) (1.3.0)\n",
      "Downloading sentence_transformers-4.0.1-py3-none-any.whl (340 kB)\n",
      "   ---------------------------------------- 0.0/340.6 kB ? eta -:--:--\n",
      "   --------- ------------------------------ 81.9/340.6 kB 1.5 MB/s eta 0:00:01\n",
      "   -------------------------- ------------- 225.3/340.6 kB 2.3 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 340.6/340.6 kB 2.6 MB/s eta 0:00:00\n",
      "Installing collected packages: sentence-transformers\n",
      "Successfully installed sentence-transformers-4.0.1\n"
     ]
    },
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'bert_score'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[12], line 6\u001b[0m\n\u001b[0;32m      4\u001b[0m get_ipython()\u001b[38;5;241m.\u001b[39msystem(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpip install -U sentence-transformers\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m      5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msentence_transformers\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m SentenceTransformer, util\n\u001b[1;32m----> 6\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mbert_score\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m score\n\u001b[0;32m      7\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtransformers\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m AutoTokenizer, AutoModel\n\u001b[0;32m      8\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmetrics\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpairwise\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m cosine_similarity\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'bert_score'"
     ]
    }
   ],
   "source": [
    "import yaml\n",
    "import pandas as pd\n",
    "import torch\n",
    "!pip install -U sentence-transformers\n",
    "from sentence_transformers import SentenceTransformer, util\n",
    "from bert_score import score\n",
    "from transformers import AutoTokenizer, AutoModel\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "# Load configuration\n",
    "with open(\"config.yaml\", \"r\") as file:\n",
    "    config = yaml.safe_load(file)\n",
    "\n",
    "# Load models\n",
    "st_model = SentenceTransformer(config[\"sentence_transformer_model\"])\n",
    "bert_tokenizer = AutoTokenizer.from_pretrained(config[\"bert_model\"])\n",
    "bert_model = AutoModel.from_pretrained(config[\"bert_model\"])\n",
    "\n",
    "def compute_similarity(human_solution, llm_solution):\n",
    "    \"\"\"Computes similarity scores using multiple methods\"\"\"\n",
    "    \n",
    "    # Sentence Transformers cosine similarity\n",
    "    emb1 = st_model.encode(human_solution, convert_to_tensor=True)\n",
    "    emb2 = st_model.encode(llm_solution, convert_to_tensor=True)\n",
    "    st_similarity = util.pytorch_cos_sim(emb1, emb2).item()\n",
    "\n",
    "    # BERT-Score\n",
    "    P, R, F1 = score([llm_solution], [human_solution], lang=\"en\", model_type=config[\"bert_model\"])\n",
    "    \n",
    "    return {\n",
    "        \"st_similarity\": st_similarity,\n",
    "        \"bert_score\": F1.tolist()[0]\n",
    "    }\n",
    "\n",
    "# Example usage\n",
    "human_text = \"The issue is with database connectivity.\"\n",
    "llm_text = \"The error is caused by a database connection failure.\"\n",
    "print(compute_similarity(human_text, llm_text))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6447ba1e-fc04-4325-beda-3f181aa83ac3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
