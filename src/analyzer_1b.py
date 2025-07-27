# src/analyzer_1b.py

import os
import json
import fitz
import numpy as np
from datetime import datetime

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from src.parser_1a import get_document_structure

class PersonaBasedPDFAnalyzer:
    def __init__(self, r1a_model_path):
        self.r1a_model_path = r1a_model_path
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            print("Downloading NLTK 'punkt' model...")
            nltk.download('punkt', quiet=True)
        print("✅ 1B Analyzer initialized.")

    def analyze_documents(self, input_json_path: str):
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)

        pdf_dir = os.path.dirname(input_json_path)
        persona = input_data['persona']['role']
        job_to_be_done = input_data['job_to_be_done']['task']
        print(f"\n🎯 Processing for Persona: {persona} | Task: {job_to_be_done}")

        print("\n[Step 1/3] Parsing document structures and extracting text chunks...")
        all_chunks = []
        for doc_info in input_data['documents']:
            pdf_path = os.path.join(pdf_dir, doc_info['filename'])
            if not os.path.exists(pdf_path):
                print(f"  ⚠️ WARNING: File not found, skipping: {pdf_path}")
                continue

            structure_data = get_document_structure(pdf_path, self.r1a_model_path)
            chunks = self._create_text_chunks(pdf_path, structure_data)
            all_chunks.extend(chunks)
            print(f"  -> Extracted {len(chunks)} chunks from {doc_info['filename']}.")

        if not all_chunks:
            return {"error": "No text chunks could be extracted from the documents."}

        print("\n[Step 2/3] Ranking chunks by relevance...")
        ranked_chunks = self._rank_chunks_by_relevance(all_chunks, persona, job_to_be_done)
        print(f"  -> Ranked {len(ranked_chunks)} total chunks.")

        print("\n[Step 3/3] Formatting final output...")
        final_output = self._format_output(ranked_chunks, input_data)
        print("  -> Output formatted.")
        return final_output

    def _create_text_chunks(self, pdf_path, structure_data):
        doc = fitz.open(pdf_path)
        chunks = []
        outline = structure_data.get('outline', [])

        for i, section in enumerate(outline):
            try:
                page = doc[section['page']]
                start_y = section['bbox'][3]
                end_y = page.rect.height
                if i + 1 < len(outline) and outline[i+1]['page'] == section['page']:
                    end_y = outline[i+1]['bbox'][1]

                clip_rect = fitz.Rect(0, start_y, page.rect.width, end_y)
                text = page.get_text("text", clip=clip_rect).strip()

                if text:
                    chunks.append({
                        "source_doc": os.path.basename(pdf_path),
                        "page": section['page'],
                        "section_title": section['text'],
                        "content": text
                    })
            except Exception as e:
                print(f"    - Warning: Could not process chunk '{section.get('text', 'N/A')}': {e}")
                continue
        doc.close()
        return chunks

    def _rank_chunks_by_relevance(self, chunks, persona, task):
        if not chunks: return []
        query = f"As a {persona}, I need to {task}"
        corpus = [f"{c['section_title']}. {c['content']}" for c in chunks]
        all_texts = [query] + corpus
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        for i, chunk in enumerate(chunks):
            chunk['relevance_score'] = float(similarities[i])
        return sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)

    def _format_output(self, ranked_chunks, input_data):
        top_chunks = ranked_chunks[:15]
        extracted_sections = []
        for i, chunk in enumerate(top_chunks):
            extracted_sections.append({
                "document": chunk['source_doc'],
                "section_title": chunk['section_title'],
                "importance_rank": i + 1,
                "page_number": chunk['page']
            })

        query = f"{input_data['persona']['role']} {input_data['job_to_be_done']['task']}"
        subsection_analysis = self._create_subsection_analysis(top_chunks, query)

        return {
            "metadata": {
                "input_documents": [d['filename'] for d in input_data['documents']],
                "persona": input_data['persona']['role'],
                "job_to_be_done": input_data['job_to_be_done']['task'],
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

    def _create_subsection_analysis(self, chunks, query):
        subsections = []
        for chunk in chunks:
            sentences = sent_tokenize(chunk['content'])
            if not sentences:
                refined_text = chunk['content'][:300] + "..."
            else:
                try:
                    all_texts = [query] + sentences
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
                    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                    best_sentence_index = np.argmax(similarities)
                    refined_text = sentences[best_sentence_index]
                except ValueError:
                    refined_text = sentences[0] if sentences else ""

            subsections.append({
                "document": chunk['source_doc'],
                "refined_text": refined_text,
                "page_number": chunk['page']
            })
        return subsections