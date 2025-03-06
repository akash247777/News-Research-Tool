# News-Research-Tool
A powerful news research tool that allows you to process and analyze news articles from various URLs. It uses advanced NLP models and embeddings to provide accurate  answers to your questions, along with relevant sources.

# 🤖📈 AKashBot: AI-Powered News Research Tool 📰🔍

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-brightgreen) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)

## 🌟 Overview
AKashBot is an AI-powered research assistant that:
- 📰 Processes news articles from multiple URLs
- 🧠 Creates semantic embeddings with HuggingFace models
- 🔍 Enables natural language queries with source attribution
- ⚡ Provides instant answers using FLAN-T5 large language model

## 🚀 Key Features
✅ **Multi-URL Processing** - Analyze up to 3 news articles simultaneously  
✅ **Smart Chunking** - Intelligent text splitting (1,000 token chunks)  
✅ **Vector Database** - FAISS-powered semantic search  
✅ **Source Tracking** - Shows exact sources for every answer  
✅ **User-Friendly** - Beautiful Streamlit interface  

## 🛠️ Tech Stack
| Component | Technology | Use Case |
|-----------|------------|----------|
| **NLP** | 🤗 HuggingFace Transformers | Text generation & embeddings |
| **Vector DB** | FAISS | Semantic similarity search |
| **UI** | Streamlit 🚀 | Web interface |
| **Text Processing** | LangChain | Document splitting & QA |

## 📦 Setup Instructions
1. **Clone the repository**  

   ```bash
   git clone https://github.com/your-username/akashbot.git
   cd akashbot
   ```
3. **Install dependencies 📚**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set HuggingFace API Token 🔑**
   
   * Get your token from ![HuggingFace]([https://img.shields.io/badge/Python-3.8%2B-blue](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending))
   * Replace the placeholder in `app.py`

5. **Run the app 🏃♂️**
   ```bash
   streamlit run app.py
   ```

# 📝 Usage Guide

1. **Input URLs**
      * https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html
      * https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html

2. **Process Content**

    Click "Process URLs" to:
    
     * 📥 Load web content
     * ✂️ Split into semantic chunks
     * 🧠 Generate embeddings
      
3. **Ask Questions**
 
   Type your question in natural language and get:
   
    * 📌 Concise answers
    * 🔗 Source citations

# ⚙️ Customization

|PARAMETER|	LOCATION|	DESCRIPTION|
|---------|---------|------------|
|**Model**	|`app.py`|	Change FLAN-T5 to other HuggingFace models|
|**Chunk Size**|	`RecursiveCharacterTextSplitter`|	Adjust chunk size (default 1000)|
|**Temperature**|	`HuggingFacePipeline`|	Control response creativity (0.0-1.0)|


# 🤝 Contributing
Contributions welcome! 🎉

Fork the repository
1. Create your feature branch (`git checkout -b feature/fooBar`)
2. Commit changes (`git commit -am 'Add some fooBar'`)
3. Push to the branch (`git push origin feature/fooBar`)
4. Create a new Pull Request

# ⚠️ Important Notes

 * Requires internet connection for HuggingFace API
 * Processing time depends on article length
 * FAISS index is stored in vectorindex_huggingface.pkl
