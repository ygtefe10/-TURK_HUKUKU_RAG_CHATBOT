# ==============================================================================
#      *** TÜRK HUKUKU RAG CHATBOT - PROJE KODU (v8) - TEK HÜCRE ***
# ==============================================================================
# Proje: Türk Hukuk metinleri üzerinde RAG (Retrieval-Augmented Generation)
#        yöntemiyle çalışan bir chatbot.
# Sürüm v8: Paket kurulumları (pip) en başa alındı,
#            vektör veritabanı olarak yerel (local) ChromaDB kullanılıyor.
# Ortam: Google Colab
# ==============================================================================

import time  # Kodun çalışma süresini ve zaman damgalarını yönetmek için
import os    # İşletim sistemiyle ilgili işlemler (API anahtarı, dosya yolları)
import shutil # Dosya/klasör silme işlemleri (örn: eski DB'yi temizleme)
import subprocess # Sistem komutlarını çalıştırmak için
import threading  # (Şu an kullanılmıyor ancak Streamlit gibi arayüzler için gerekebilir)
import pandas as pd # Veri işleme için (şu an doğrudan kullanılmıyor ancak gelecekte veri analizi için tutuluyor)

print(f"--- Proje Başlangıç Zamanı: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

# ==============================================================================
# Adım 1: Gerekli Bağımlılıkların Yüklenmesi
# ==============================================================================
# Colab ortamı her başladığında paketlerin yeniden kurulması gerekir.
# Bu adımı en başa alarak, kodun geri kalanının ihtiyaç duyduğu
# kütüphanelerin (LangChain, ChromaDB, Gemini vb.) yüklü olduğundan emin oluyoruz.
print("\n--- Adım 1: Gerekli Paketler Yükleniyor ---")
print("Gerekli tüm kütüphaneler yükleniyor (Bu işlem biraz sürebilir)...")


print("✅ Gerekli ana paketler yüklendi/güncellendi.")

# Paketler yüklendikten sonra, onları kod içinde kullanabilmek için import ediyoruz.
# Bu importları bir try-except bloğuna alıyoruz ki,
# eğer bir kütüphane eksikse veya yüklenememişse programı en başta durdurabilelim.
try:
    from pyngrok import ngrok, conf # (Streamlit'i dışarıya açmak için gerekli)
    from datasets import load_dataset # Hugging Face'den veri setini çekmek için
    import google.generativeai as genai # Gemini LLM'i kullanmak için
    import chromadb # Yerel vektör veritabanı için
    from langchain_text_splitters import RecursiveCharacterTextSplitter # Metinleri parçalara (chunk) ayırmak için
    from langchain_core.documents import Document # LangChain'in standart belge formatı
    from langchain_core.prompts import PromptTemplate # LLM'e göndereceğimiz şablon
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda # LangChain Expression Language (LCEL) zincirini kurmak için

    # Colab'e özel 'userdata' secret yöneticisini kontrol et.
    # Eğer Colab'de değilsek (örn: local VSCode), bu import hata verecektir.
    try:
        from google.colab import userdata; USE_SECRETS = True
        print("Google Colab 'userdata' (Secrets) modülü bulundu.")
    except ImportError:
        USE_SECRETS = False # Colab'de değiliz, API anahtarını manuel isteyeceğiz.
        print("Google Colab 'userdata' bulunamadı. API anahtarı manuel istenecek.")

    print("✅ Gerekli kütüphaneler başarıyla import edildi.")
except ImportError as e:
    print(f"❌ Kütüphane import hatası: {e}. Lütfen paket kurulumunu kontrol edin.")
    # Kritik bir kütüphane eksikse, programın devam etmesi anlamsız.
    raise SystemExit("Kritik kütüphaneler yüklenemedi. Notebook durduruluyor.")
print("-" * 50)


# ==============================================================================
# Adım 2: Gemini API Anahtarının Ayarlanması
# ==============================================================================
# Güvenlik nedeniyle API anahtarını koda doğrudan yazmıyoruz.
# Her çalıştırmada kullanıcıdan manuel olarak girmesini istiyoruz (input).
# Bu, kodun paylaşımını kolaylaştırır ve anahtarın sızmasını engeller.
print("\n--- Adım 2: Gemini API Anahtarı Ayarlanıyor ---")
API_KEY_LOADED = False
try:
    # Kullanıcıdan API anahtarını güvenli bir şekilde al
    api_key_input = input("Lütfen Gemini API Anahtarınızı Girin: ")
    if not api_key_input:
        raise ValueError("API Anahtarı girilmedi.")

    # Alınan anahtarı işletim sistemi ortam değişkeni (environment variable) olarak ayarlıyoruz.
    # Bu, 'genai' kütüphanesinin anahtarı otomatik olarak bulması için standart bir yoldur.
    os.environ["GEMINI_API_KEY"] = api_key_input
    api_key_env = os.getenv('GEMINI_API_KEY')

    if not api_key_env:
        print("❌ Hata: API anahtarı ortam değişkeni olarak ayarlanamadı.")
    else:
        # genai kütüphanesini bu anahtarla yapılandır
        genai.configure(api_key=api_key_env)
        print("✅ Gemini API başarıyla yapılandırıldı.")
        API_KEY_LOADED = True # Sonraki adımlar için bayrağı ayarla
except Exception as e:
    print(f"❌ Gemini API yapılandırma hatası: {e}")
print("-" * 50)

# ==============================================================================
# Ek Adım: Google Cloud Kimlik Doğrulaması (ADC)
# ==============================================================================
# Gemini API anahtarı (Adım 2) LLM (gemini-2.0-flash) için kullanılır.
# Ancak, yeni nesil embedding modelleri (örn: text-embedding-004)
# genellikle ek olarak Google Cloud projesi üzerinden kimlik doğrulaması
# (Application Default Credentials - ADC) gerektirir. Bu adım bu ek doğrulamayı yapar.
GCLOUD_AUTH_DONE = False
if API_KEY_LOADED: # Sadece API anahtarı başarıyla yüklendiyse devam et
    print("\n--- Ek Adım: Google Cloud Kimlik Doğrulaması (ADC) ---")
    try:
        # ADC kimlik bilgilerinin Colab'de saklandığı varsayılan yol
        adc_path = "/content/.config/application_default_credentials.json"

        # Eğer bu dosya zaten varsa, daha önce doğrulama yapılmış demektir.
        # Tekrar sormamak için bu adımı atlıyoruz.
        if os.path.exists(adc_path):
            print("Mevcut ADC dosyası bulundu, kimlik doğrulaması atlanıyor.")
            GCLOUD_AUTH_DONE = True
        else:
            # Dosya yoksa, kullanıcıdan kimlik doğrulaması istiyoruz.
            # 'gcloud auth' komutu bir link açar, kullanıcı izin verir ve bir kod yapıştırır.
             print("Lütfen çıkan linke tıklayıp Google hesabınızla izin verin ve doğrulama kodunu buraya yapıştırın.")
             get_ipython().system('gcloud auth application-default login --quiet --no-launch-browser')

             if os.path.exists(adc_path):
                 print("\n✅ ADC kimlik doğrulaması başarıyla tamamlandı.")
                 GCLOUD_AUTH_DONE = True
             else:
                 print("\n⚠️ ADC dosyası oluşturulamadı. Embedding adımında sorun yaşanabilir.")
    except Exception as gcloud_e:
        print(f"⚠️ gcloud kimlik doğrulama hatası: {gcloud_e}")
    print("-" * 50)
else:
    print("API Anahtarı yüklenemediği için Google Cloud ADC adımı atlandı.")
    print("-" * 50)

# ==============================================================================
# Adım 3: Veri Setinin Yüklenmesi
# ==============================================================================
# Bu projede, Hugging Face Hub üzerinde bulunan özel bir Türk hukuku
# Soru-Cevap veri setini kullanıyoruz.
print("\n--- Adım 3: Veri Seti Yükleniyor ---")
dataset_name = "Renicames/turkish-law-chatbot" # Veri setinin Hugging Face ID'si
data = None
DATA_LOADED = False
print(f"Hugging Face Hub'dan '{dataset_name}' veri seti yükleniyor...")
try:
    # 'load_dataset' fonksiyonu ile veriyi çek
    dataset = load_dataset(dataset_name)
    print("✅ Veri seti yüklendi.")

    # Genellikle veri 'train' bölünmesinde (split) bulunur
    if 'train' in dataset:
        data = dataset['train']
        print(f"Veri setinin 'train' bölümü alındı. Toplam kayıt: {len(data)}")

        # Verinin RAG için uygun olup olmadığını kontrol et.
        # Bizim 'Cevap' sütunundaki metinlere (asıl hukuk metni) ihtiyacımız var.
        if 'Soru' in data.column_names and 'Cevap' in data.column_names:
            print("✅ Gerekli 'Soru' ve 'Cevap' sütunları bulundu."); DATA_LOADED = True
        else:
            print("❌ Gerekli 'Soru'/'Cevap' sütunları bulunamadı."); data = None
    else:
        print("❌ Veri setinde 'train' bölümü bulunamadı."); data = None
except Exception as e:
    print(f"❌ Veri seti yüklenirken hata oluştu: {e}"); data = None

if not DATA_LOADED:
    print("❌ Veri seti yüklenemedi. Sonraki adımlar çalışmayabilir.")
print("-" * 50)

# ==============================================================================
# Adım 4: Verinin Hazırlanması ve Parçalara Ayrılması (Chunking)
# ==============================================================================
# RAG modelinin çalışması için, uzun hukuk metinlerini (Cevaplar)
# daha küçük, yönetilebilir parçalara (chunks) bölmemiz gerekiyor.
# Bu, embedding modelinin (Adım 5) metinleri daha iyi anlamlandırmasını
# ve vektör veritabanının daha verimli çalışmasını sağlar.
print("\n--- Adım 4: Veri Hazırlanıyor ve Parçalanıyor (Tüm Veri) ---")
SORU_COLUMN = 'Soru'; CEVAP_COLUMN = 'Cevap'
documents_for_rag = []; doc_chunks = []
CHUNKS_CREATED = False

if DATA_LOADED and data is not None:
    # 1. Veriyi LangChain 'Document' formatına dönüştürme
    # 'page_content' = Asıl metin (vektör veritabanına girecek olan)
    # 'metadata' = Metinle ilişkili ek bilgi (kaynak soru, ID, vb.)
    print(f"'{CEVAP_COLUMN}' sütunundaki metinler 'Document' formatına getiriliyor...")
    for i, item in enumerate(data):
        page_content = item[CEVAP_COLUMN] # Asıl hukuk metni
        # Orijinal soruyu metadata'ya ekliyoruz. Bu, kaynak takibi için yararlı.
        metadata = {"source_id": i, SORU_COLUMN: item[SORU_COLUMN]}
        documents_for_rag.append(Document(page_content=page_content, metadata=metadata))
    print(f"✅ {len(documents_for_rag)} adet LangChain 'Document' nesnesi oluşturuldu.")

    # 2. Metin Parçalayıcıyı (Text Splitter) Tanımlama
    # RecursiveCharacterTextSplitter: Metinleri "\n\n", sonra "\n", sonra " "
    # gibi ayraçlara göre bölmeye çalışır. Anlamsal bütünlüğü korumaya çalışır.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Her bir parçanın maksimum karakter sayısı
        chunk_overlap=150  # Parçalar arası ortak karakter sayısı (anlam kaybını önlemek için)
    )

    # 3. Parçalama İşlemi
    try:
        doc_chunks = text_splitter.split_documents(documents_for_rag)
        print(f"✅ {len(documents_for_rag)} belge, {len(doc_chunks)} adet metin parçasına (chunk) ayrıldı.")
        if doc_chunks:
            # İlk parçanın metaverisini basarak işlemin doğruluğunu kontrol et
            print("Test: İlk Parçanın Metaverisi:", doc_chunks[0].metadata); CHUNKS_CREATED = True
        else:
            print("❌ Parçalama işlemi boş bir liste döndürdü.")
    except Exception as e:
        print(f"❌ Metin bölme işlemi sırasında hata: {e}"); doc_chunks = []
else:
    print("❌ Veri seti yüklenemediği için (Adım 3) parçalama yapılamıyor.")

if not CHUNKS_CREATED:
    print("❌ Metin parçaları (chunks) oluşturulamadı.")
print("-" * 50)

# ==============================================================================
# Adım 5: Embedding Oluşturma ve Yerel Vektör Veritabanı Kurulumu
# ==============================================================================
# Bu adımda, Adım 4'te oluşturduğumuz her bir metin parçasını (chunk)
# Gemini'nin 'text-embedding-004' modelini kullanarak sayısal bir vektöre
# (embedding) dönüştüreceğiz.
# Ardından bu vektörleri, yerel bir ChromaDB veritabanına kaydedeceğiz.
print("\n--- Adım 5: Embedding ve YEREL Vektör Veritabanı Kurulumu ---")

# --- Yardımcı Fonksiyon: Embedding (Hata Yönetimli) ---
# Google API'leri sık kullanımda "429 - Resource Exhausted" (Rate Limit) hatası verebilir.
# Bu fonksiyon, bu hatayı yakalayıp, bir süre bekleyip (exponential backoff)
# işlemi tekrar denemek (retry) için yazılmıştır. Projenin sağlamlığı için kritiktir.
def embed_content_with_retry(content, model="models/text-embedding-004", task_type="RETRIEVAL_DOCUMENT", max_retries=5, initial_delay=2):
    delay = initial_delay; last_exception = None; is_batch = isinstance(content, list)
    for attempt in range(max_retries):
        try:
            # Sorgu (QUERY) ve Belge (DOCUMENT) embedding'leri farklı 'task_type' gerektirir.
            # Eğer tek bir metin gelirse ve tipi 'QUERY' ise onu kullan,
            # aksi halde (liste veya tek metin fark etmeksizin) 'DOCUMENT' olarak etiketle.
            current_task_type = task_type if isinstance(content, str) and task_type == 'RETRIEVAL_QUERY' else 'RETRIEVAL_DOCUMENT'

            result = genai.embed_content(model=model, content=content, task_type=current_task_type)

            # Dönen sonucun formatı tekil (embedding) veya çoğul (embeddings) olabilir
            embedding_key = 'embedding' if 'embedding' in result else ('embeddings' if 'embeddings' in result else None)
            if embedding_key:
                return result[embedding_key] # Başarılı embedding'i döndür
            else:
                raise ValueError("Embedding sonucu beklenen 'embedding' veya 'embeddings' anahtarını içermiyor.")

        except Exception as e:
            error_str = str(e); print(f"Embedding hatası (Deneme {attempt + 1}/{max_retries}): {error_str[:200]}...")
            # Rate limit hatası (429) veya kaynak tükenmesi hatasını yakala
            if "Resource has been exhausted" in error_str or "429" in error_str:
                wait_time = delay * 5 # Rate limit için daha uzun bekle
                print(f"Rate limit tespit edildi, {wait_time} saniye bekleniyor...");
                time.sleep(wait_time); delay *= 2 # Bekleme süresini üssel olarak artır
            elif attempt < max_retries - 1: # Diğer geçici hatalar için
                print(f"Geçici hata, {delay} saniye bekleniyor...");
                time.sleep(delay); delay *= 2 # Bekleme süresini üssel olarak artır
            else:
                print("Maksimum deneme sayısına ulaşıldı."); last_exception = e; break # Döngüden çık

    if last_exception: # Tüm denemelere rağmen başarısız olduysa
        if is_batch: return [None] * len(content) # Batch ise 'None' listesi döndür
        else: raise last_exception # Tekil ise hatayı fırlat
    return None
# --- Yardımcı Fonksiyon Bitişi ---

chroma_client = None
chroma_collection = None
# Veritabanını Colab ortamında yerel bir klasörde sakla
db_path = "./chroma_db_law_local_full"
collection_name = "hukuk_tr_collection_full_local" # DB içindeki koleksiyon adı

# Mevcut sürümde, her çalıştırmada veritabanını sıfırdan oluşturuyoruz.
# Bu, veride veya chunking ayarlarında değişiklik yapıldığında tutarlılık sağlar.
DB_CREATED_OR_LOADED = False

if CHUNKS_CREATED and doc_chunks and (API_KEY_LOADED or GCLOUD_AUTH_DONE):
    print(f"Yerel Chroma veritabanı yolu: {db_path}")
    try:
        # 1. Eski Veritabanını Temizle
        # Eğer bu klasör varsa, içindekileri sil (shutil.rmtree)
        if os.path.exists(db_path):
            print(f"Eski veritabanı '{db_path}' siliniyor...");
            shutil.rmtree(db_path)
        os.makedirs(db_path, exist_ok=True) # Klasörü (yeniden) oluştur

        print("\n--- Embedding İşlemi Başlatılıyor (TÜM VERİ - UZUN SÜREBİLİR!) ---")
        # Chroma'ya göndermek için metinleri, metaları ve ID'leri ayır
        chunk_texts = [doc.page_content for doc in doc_chunks]
        chunk_metadatas = [doc.metadata for doc in doc_chunks]
        chunk_ids = [f"doc_{i}" for i in range(len(doc_chunks))] # Her chunk için benzersiz ID
        print(f"{len(chunk_texts)} adet metin parçası (chunk) bulundu.")

        # 2. Embedding'leri Batch (Grup) Halinde Oluştur
        print("\nMetin parçaları için vektörler (embeddings) oluşturuluyor...")
        start_time = time.time()
        all_embeddings = []
        batch_size_embed = 100 # API'ye tek seferde 100 metin gönder (Rate limit'i aşmamak için)
        num_batches_embed = (len(chunk_texts) + batch_size_embed - 1) // batch_size_embed

        for i in range(0, len(chunk_texts), batch_size_embed):
            batch_index_embed = i // batch_size_embed + 1
            batch_texts = chunk_texts[i:i+batch_size_embed]
            try:
                # Hata yönetimli fonksiyonumuzu kullanarak embedding'leri al
                batch_embeddings = embed_content_with_retry(batch_texts, task_type="RETRIEVAL_DOCUMENT")
                all_embeddings.extend(batch_embeddings)
                print(f"Embedding Batch {batch_index_embed}/{num_batches_embed} işlendi (Toplam {len(all_embeddings)}/{len(chunk_texts)}).")
            except Exception as e:
                print(f"❌ Batch {batch_index_embed} işlenirken kritik hata: {e}.")
                all_embeddings.extend([None] * len(batch_texts)) # Başarısız olanları 'None' ile doldur

            time.sleep(1) # API'ye saygı için her batch arası 1 saniye bekle

        end_time = time.time()
        print(f"Embedding işlemi toplam {end_time - start_time:.2f} saniye sürdü.")

        # 3. Başarısız Embedding'leri Filtrele
        # 'None' olarak işaretlenen (başarısız olan) embedding'leri ayıkla
        valid_indices = [i for i, emb in enumerate(all_embeddings) if emb is not None]
        valid_texts = [chunk_texts[i] for i in valid_indices]
        valid_metadatas = [chunk_metadatas[i] for i in valid_indices]
        valid_embeddings = [all_embeddings[i] for i in valid_indices]
        valid_ids = [chunk_ids[i] for i in valid_indices]
        if len(valid_indices) != len(chunk_texts):
            print(f"⚠️ Uyarı: {len(chunk_texts) - len(valid_indices)} adet metin için embedding hesaplanamadı.")

        # 4. ChromaDB'ye Verileri Ekle
        if valid_ids:
             print(f"{len(valid_ids)} adet geçerli embedding bulundu.")
             print(f"\nYerel Chroma veritabanı '{db_path}' içine oluşturuluyor...")
             try:
                 # 'PersistentClient', verileri diske (belirtilen yola) kaydeder.
                 chroma_client = chromadb.PersistentClient(path=db_path)
                 print(f"'{collection_name}' koleksiyonu oluşturuluyor/yükleniyor...")
                 # 'get_or_create_collection' ile koleksiyonu güvenle al (varsa yükler, yoksa oluşturur)
                 chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
                 # chroma_collection = chroma_client.create_collection(name=collection_name) # Bu hata verir eğer varsa

                 # Veritabanına ekleme işlemini de batch halinde yapıyoruz.
                 # Binlerce veriyi tek seferde eklemek Chroma'yı yorabilir.
                 batch_size_chroma = 4000
                 num_chroma_batches = (len(valid_ids) + batch_size_chroma - 1) // batch_size_chroma
                 print(f"{len(valid_ids)} öğe Chroma'ya {num_chroma_batches} batch halinde eklenecek...")
                 total_added = 0

                 for i in range(0, len(valid_ids), batch_size_chroma):
                      batch_index_chroma = i // batch_size_chroma + 1
                      # İlgili batch için ID, embedding, metin ve metaveriyi seç
                      ids_batch=valid_ids[i:i+batch_size_chroma]
                      embeddings_batch=valid_embeddings[i:i+batch_size_chroma]
                      documents_batch=valid_texts[i:i+batch_size_chroma]
                      metadatas_batch=valid_metadatas[i:i+batch_size_chroma]

                      print(f"Chroma Batch {batch_index_chroma}/{num_chroma_batches} ({len(ids_batch)} öğe) ekleniyor...")
                      try:
                          # Verileri koleksiyona ekle
                          chroma_collection.add(ids=ids_batch, embeddings=embeddings_batch, documents=documents_batch, metadatas=metadatas_batch)
                          total_added += len(ids_batch)
                          print(f"Batch {batch_index_chroma} eklendi. Toplam eklenen: {total_added}/{len(valid_ids)}")
                      except Exception as add_e:
                          print(f"❌ Hata (Chroma Batch {batch_index_chroma} eklenirken): {add_e}")

                 if total_added == len(valid_ids):
                     print(f"✅ Tüm veriler ({total_added}) yerel Chroma veritabanına başarıyla eklendi.")
                 else:
                     print(f"⚠️ Uyarı: Veri kaybı var. Eklenen: {total_added}/{len(valid_ids)}")

                 # 5. Veritabanı Test Sorgusu (Sanity Check)
                 # DB'nin çalışıp çalışmadığını kontrol etmek için basit bir arama yap
                 print("\nYerel DB testi: 'anayasa mahkemesi' kelimesi aranıyor...")
                 query_text = "anayasa mahkemesi"
                 try:
                      # Sorguyu da (farklı task_type ile) embed et
                      query_embedding = embed_content_with_retry(query_text, task_type='RETRIEVAL_QUERY')
                      if query_embedding:
                           # 'query' metodu ile en yakın 2 sonucu (n_results=2) getir
                           results = chroma_collection.query(query_embeddings=[query_embedding], n_results=2)
                           if results and results.get('ids') and results['ids'][0]:
                               print(f"✅ Veritabanı testi başarılı. {len(results['ids'][0])} sonuç bulundu.");
                               DB_CREATED_OR_LOADED = True # Her şey yolunda, RAG'a geçebiliriz.
                           else:
                               print("⚠️ Test sorgusu sonuç döndürmedi (DB boş olabilir).")
                      else:
                          print("❌ Test sorgusu embed edilemedi.")
                 except Exception as search_e:
                     print(f"❌ Veritabanı test sorgusu hatası: {search_e}")
             except Exception as db_e:
                 print(f"❌ Yerel Chroma veritabanı kurulum hatası: {db_e}"); chroma_client = None; chroma_collection = None
        else:
            print("❌ Veritabanına eklenecek geçerli embedding bulunamadı.")
    except Exception as e:
        print(f"❌ Embedding/DB (Adım 5) genel hatası: {e}")
else:
    print("❌ Önceki adımlardaki hatalar (Chunk/API) nedeniyle Adım 5 atlandı.")

if not DB_CREATED_OR_LOADED:
    print("\n❌ Adım 5 tamamlanamadı. Vektör Veritabanı hazır değil.")
print("-" * 50)


# ==============================================================================
# Adım 6: RAG Pipeline (Zincir) Kurulumu
# ==============================================================================
# Bu adımda, tüm bileşenleri (Retriever, Prompt, LLM) bir araya getirerek
# LangChain Expression Language (LCEL) kullanarak RAG zincirimizi kuruyoruz.
# Not: LLM olarak LangChain'in standart Gemini sarmalayıcısı (wrapper) yerine
# doğrudan 'google.generativeai' kütüphanesini kullanıyoruz. Zincir buna göre uyarlandı.
print("\n--- Adım 6: RAG Pipeline Kurulumu ---")

# Bu importlar LCEL zinciri için gereklidir (Adım 1'de yapıldı)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

rag_chain = None
RAG_READY = False
llm_model_name = "gemini-2.0-flash" # Hız ve maliyet için optimize edilmiş Gemini modeli

if DB_CREATED_OR_LOADED and chroma_collection is not None:
    print("RAG pipeline (zinciri) oluşturuluyor...")
    try:
        # 1. LLM'i Tanımla
        # Doğrudan 'genai' kütüphanesinden modeli çağırıyoruz.
        # 'temperature=0.3' ile modelin daha tutarlı ve daha az yaratıcı (daha az "halüsinasyon")
        # cevaplar vermesini sağlıyoruz. Hukuk metinleri gibi hassas konular için bu önemlidir.
        llm_model = genai.GenerativeModel(
            model_name=llm_model_name,
            generation_config={"temperature": 0.3}
        )
        print(f"✅ LLM ({llm_model_name}, T=0.3) tanımlandı.")

        # 2. Retriever Fonksiyonunu Tanımla
        # Bu fonksiyon, kullanıcı sorgusunu alır, ChromaDB'de arar
        # ve en ilgili 'k' adet metin parçasını (context) döndürür.
        def retrieve_context(query: str, k: int = 3):
            # print(f"   [Debug: Retriever çalıştı, sorgu: '{query[:50]}...']") # (Debug için)
            try:
                # Sorguyu, 'RETRIEVAL_QUERY' tipiyle embed et
                query_embedding = embed_content_with_retry(query, task_type='RETRIEVAL_QUERY')
                if not query_embedding:
                    return "Sorgu vektöre dönüştürülemedi."

                # ChromaDB'den en yakın 'k' sonucu (embedding'e göre) sorgula
                # Sadece 'documents' (metinler) bölümünü iste
                results = chroma_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=['documents']
                )

                # Sonuçları formatla (birleştir)
                if results and results.get('documents') and results['documents'][0]:
                    context = "\n\n".join(results['documents'][0])
                    # print(f"   [Debug: {len(results['documents'][0])} adet bağlam (context) bulundu.]") # (Debug için)
                    return context
                else:
                    # print("   [Debug: İlgili bağlam bulunamadı.]") # (Debug için)
                    return "İlgili bilgi bulunamadı."
            except Exception as e:
                return f"Bilgi alınırken hata oluştu: {e}"
        print("✅ Retriever (Bilgi Çekici) fonksiyonu tanımlandı.")

        # 3. Prompt Şablonunu Oluştur
        # Bu şablon, LLM'e tam olarak nasıl davranması gerektiğini söyler (System Prompt).
        # {context} ve {question} alanları zincir tarafından doldurulacaktır.
        template_str = """
        Yalnızca aşağıda verilen bağlamı (context) kullanarak soruyu Türkçe yanıtlayın.
        Eğer cevap bağlamda yoksa veya cevaptan emin değilseniz, "Bu konuda sağlanan bilgiler arasında bir cevap bulamadım." deyin.
        Cevabınızı doğrudan ve net bir şekilde verin.

        Bağlam:
        {context}

        Soru:
        {question}

        Yanıt:
        """
        prompt_template = PromptTemplate.from_template(template_str)
        print("✅ Prompt şablonu (System Prompt) oluşturuldu.")

        # 4. RAG Zincirini (LCEL) Kur
        # LCEL (LangChain Expression Language), '|' (pipe) operatörü ile
        # adımları birbirine bağlamamızı sağlar.
        # Zincir şu adımları izler:
        rag_chain = (
            # Adım A: Zincir, 'question' içeren bir dictionary ile başlar.
            # RunnablePassthrough() gelen ham sorguyu (string) alır ve 'question' anahtarı altına koyar.
            # Girdi: "Devletin şekli nedir?"
            # Çıktı: {"question": "Devletin şekli nedir?"}
            {"question": RunnablePassthrough()}

            # Adım B: 'context' anahtarını ekle.
            # 'retrieve_context' fonksiyonunu 'question' ile çalıştır ve sonucunu 'context'e ata.
            # Girdi: {"question": "..."}
            # Çıktı: {"question": "...", "context": "[ilgili hukuk metni...]"}
            | RunnablePassthrough.assign(context=lambda x: retrieve_context(x["question"]))

            # Adım C: Prompt'u formatla.
            # {'question': ..., 'context': ...} girdisini alıp prompt şablonuna yerleştirir.
            # Çıktı olarak formatlanmış tam bir metin (string) verir.
            # Girdi: {"question": "...", "context": "..."}
            # Çıktı: "Yalnızca aşağıda verilen bağlamı... Soru: ... Yanıt:"
            | RunnableLambda(lambda x: prompt_template.format(question=x["question"], context=x["context"]))

            # Adım D: LLM'i Çağır.
            # Formatlanmış metni (prompt) alır ve doğrudan genai modeline gönderir.
            # Modelden gelen '.text' yanıtını döndürür.
            # Girdi: "Yalnızca aşağıda verilen bağlamı..."
            # Çıktı: "Türkiye Devleti bir Cumhuriyettir."
            | RunnableLambda(lambda formatted_prompt: llm_model.generate_content(formatted_prompt).text)
        )
        print("✅ RAG zinciri (LCEL) başarıyla oluşturuldu.")

        # 5. Zinciri Test Et
        print("\nRAG zinciri test ediliyor (İlk sorgu)...")
        test_question = "Devletin şekli nedir?"
        print(f"Test Sorusu: {test_question}")
        try:
            start_time = time.time()
            # 'invoke' metodu ile zinciri çalıştır
            response = rag_chain.invoke(test_question)
            end_time = time.time()
            print(f"Test Cevabı ({end_time - start_time:.2f} s): {response}")
            RAG_READY = True
            if "bulunamadı" in response:
                print("⚠️ Test cevabı bilgi bulamadı (Bu durum, veritabanının ilgili bilgiyi içermemesi veya retriever'ın bulamaması durumunda normaldir).")
        except Exception as invoke_e:
            print(f"❌ RAG zinciri testi (invoke) sırasında hata: {invoke_e}")

    except Exception as e:
        print(f"❌ RAG pipeline (Adım 6) kurulum hatası: {e}"); rag_chain = None
else:
    print("❌ Önceki adımlarda (Veritabanı) hata olduğu için RAG zinciri kurulamadı.")

if not RAG_READY:
    print("\n❌ Adım 6 tamamlanamadı. Chatbot hazır değil.")
print("-" * 50)


# ==============================================================================
# ==============================================================================
# *** İKİNCİ HÜCRE BURADA BAŞLIYOR (Adım 7-9) ***
# ==============================================================================
# ==============================================================================


# ==============================================================================
# Adım 7: Streamlit Arayüz Kodu (app.py - Güvenlik Ayarları Eklendi)
# ==============================================================================
# Bu adım, RAG modelimizi interaktif bir web arayüzüne dönüştürecek olan
# 'app.py' dosyasının içeriğini hazırlar.
# Kod, bir Python string'i (f-string) olarak oluşturulur ve Colab ortamına
# bir dosya olarak yazılır.
print("\n--- Adım 7: Streamlit Arayüz Kodu Hazırlanıyor (Güvenlik Ayarları) ---")

# Colab notebook'undaki mevcut değişkenleri (model adı, DB yolu vb.)
# 'app.py' içine gömmek için burada yakalıyoruz.
APP_LLM_MODEL = "gemini-2.0-flash"
# 'DB_CREATED_OR_LOADED' bayrağını kontrol et, eğer DB yolu tanımlıysa onu kullan,
# değilse varsayılan (fallback) bir yol belirle.
APP_DB_PATH = db_path if 'DB_CREATED_OR_LOADED' in locals() and DB_CREATED_OR_LOADED else "./chroma_db_law_local_full"
APP_COLLECTION_NAME = collection_name if 'DB_CREATED_OR_LOADED' in locals() and DB_CREATED_OR_LOADED else "hukuk_tr_collection_full_local"
APP_SORU_COL = SORU_COLUMN if 'SORU_COLUMN' in locals() else 'Soru'
APP_DATASET_ID = dataset_name if 'dataset_name' in locals() else "Renicames/turkish-law-chatbot"

# Streamlit uygulamasının tam Python kodunu içeren f-string:
# *** DİKKAT: Bu f-string içindeki yorumlarda {} parantezleri kullanılmamalıdır. ***
streamlit_app_code = f"""
# ===============================================
#     *** TÜRK HUKUKU RAG CHATBOT - APP.PY ***
# ===============================================
# Bu dosya, Streamlit kütüphanesini kullanarak
# RAG modelimiz için bir web arayüzü sağlar.
# Not: Bu dosya, Colab notebook'u tarafından
# otomatik olarak oluşturulmuştur.
# ===============================================

import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # Gemini Güvenlik Ayarları için
import chromadb
from langchain_core.prompts import PromptTemplate
import time
import os
import shutil # Dosya işlemleri için

# --- Konfigürasyon ve Global Değişkenler ---

# API Anahtarını al: Önce ortam değişkenlerinden (os.environ) dener,
# bulamazsa Streamlit'in kendi 'secrets' (st.secrets) yönetiminden dener.
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except:
        st.error("Gemini API Anahtarı 'GEMINI_API_KEY' bulunamadı! Lütfen ayarlayın.")
        st.stop() # Anahtar yoksa uygulama durur.

# Colab'den enjekte edilen sabitler
DB_PATH = "{APP_DB_PATH}"
COLLECTION_NAME = "{APP_COLLECTION_NAME}"
MODEL_NAME_LLM = "{APP_LLM_MODEL}"
MODEL_NAME_EMBEDDING = "models/text-embedding-004"
SORU_COLUMN_NAME = "{APP_SORU_COL}"
DATASET_ID_STR = "{APP_DATASET_ID}"

# Gemini API'yi yapılandır
try:
    genai.configure(api_key=API_KEY)
except Exception as config_e:
    st.error(f"API yapılandırılamadı: {{config_e}}")
    st.stop()

# --- Yardımcı Fonksiyonlar ---

# Embedding Fonksiyonu (Colab notebook'u ile aynı)
# API hatalarına (örn: rate limit) karşı yeniden deneme (retry) mekanizması içerir.
def embed_content_with_retry(content, model=MODEL_NAME_EMBEDDING, task_type="RETRIEVAL_DOCUMENT", max_retries=3, initial_delay=1):
    delay = initial_delay; last_exception = None; is_batch = isinstance(content, list)
    for attempt in range(max_retries):
        try:
            current_task_type = task_type if isinstance(content, str) and task_type == 'RETRIEVAL_QUERY' else 'RETRIEVAL_DOCUMENT'
            result = genai.embed_content(model=model, content=content, task_type=current_task_type)
            embedding_key = 'embedding' if 'embedding' in result else ('embeddings' if 'embeddings' in result else None)
            if embedding_key: return result[embedding_key]
            else: raise ValueError("Embedding sonucu anahtar içermiyor.")
        except Exception as e:
            # Hata durumunda kullanıcıya 'toast' bildirimi göster
            st.toast(f"Embedding hatası (Deneme {{{{attempt + 1}}}}): {{{{e}}}}", icon="⚠️")
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(delay); delay *= 2 # Hata sonrası bekleme (exponential backoff)
            else:
                if is_batch: return [None] * len(content)
                else: raise last_exception
    return None

# Chroma Vektör Veritabanını Yükleme Fonksiyonu
# '@st.cache_resource': Streamlit'e bu fonksiyonun sonucunu (yani DB bağlantısını)
# önbelleğe almasını (cache) söyler. Bu sayede her kullanıcı etkileşiminde
# veritabanı tekrar tekrar yüklenmez, performans artar.
@st.cache_resource
def get_chroma_collection():
    try:
        # Streamlit deploy ortamlarında dosya yolları değişebilir.
        # Önce 'DB_PATH' var mı diye kontrol et.
        if not os.path.exists(DB_PATH):
            # Eğer yoksa, 'app.py' dosyasının bulunduğu dizine göre
            # göreceli (relative) yolu bulmaya çalış.
            script_dir = os.path.dirname(__file__)
            db_path_rel = os.path.join(script_dir, DB_PATH)
            if not os.path.exists(db_path_rel):
                 st.error(f"Veritabanı yolu bulunamadı: '{{DB_PATH}}' veya '{{db_path_rel}}'.")
                 return None
            effective_db_path = db_path_rel # Göreceli yolu kullan
        else:
            effective_db_path = DB_PATH # Orijinal yolu kullan

        # 'PersistentClient', veritabanının diskten (belirtilen yoldan)
        # kalıcı olarak yüklenmesini sağlar. (Colab'de oluşturduğumuz DB)
        client = chromadb.PersistentClient(path=effective_db_path)
        # Koleksiyonu isimle çağır
        collection = client.get_collection(name=COLLECTION_NAME)
        st.info(f"Yerel Chroma koleksiyonu '{{COLLECTION_NAME}}' yüklendi (Öğe Sayısı: {{collection.count()}}).")
        # DB'nin boş olup olmadığını kontrol et
        if collection.count() == 0:
            st.error("⚠️ Yerel Chroma koleksiyonu boş! Colab'de Adım 5'in çalıştığından emin olun.")
        return collection
    except Exception as e:
        st.error(f"Yerel Chroma yüklenirken/alınırken hata: {{e}}")
        st.error(f"Veritabanı dosyalarının ('{{DB_PATH}}' klasörü) 'app.py' ile aynı dizinde olduğundan emin olun.")
        return None

# Retriever Fonksiyonu (Bilgi Çekici)
# Bu fonksiyon, kullanıcı sorgusuna en çok benzeyen metin parçalarını (context) DB'den çeker.
# GÜNCELLEME: Bu versiyon, metinlere ek olarak kaynak (metadata) bilgisini de döndürür.
def retrieve_context(query: str, collection, k: int = 3):
    if collection is None:
        return "Veritabanı bağlantısı kurulamadı.", ""
    try:
        # 1. Kullanıcı sorgusunu vektöre dönüştür (Query embedding)
        query_embedding = embed_content_with_retry(query, task_type='RETRIEVAL_QUERY')
        if not query_embedding:
            return "Sorgu vektöre çevrilirken hata oluştu.", ""

        # 2. ChromaDB'de en yakın 'k' sonucu sorgula
        # 'include' ile hem metinleri (documents) hem de metaveriyi (metadatas) istiyoruz.
        results = collection.query(query_embeddings=[query_embedding], n_results=k, include=['documents', 'metadatas'])

        # 3. Sonuçları formatla
        if results and results.get('documents') and results['documents'][0]:
            context_list = []; sources = []
            # Bulunan her sonuç için metni ve metaveriyi ayıkla
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                # Metaveriden orijinal 'Soru' sütununu kaynak olarak al
                source_info = metadata.get(SORU_COLUMN_NAME, f"ID: {{metadata.get('source_id', 'Bilinmiyor')}}")
                context_list.append(f"Metin: {{doc}}")
                sources.append(source_info)

            # Bulunan metinleri LLM'in anlayacağı 'context' formatına getir
            context_str = "\\n\\n".join(context_list)
            # Kaynakları (tekrarları kaldırarak) formatla
            source_str = "\\n".join([f"- {{s}}" for s in set(sources)])

            return context_str, source_str # Hem bağlamı hem kaynakları döndür
        else:
            return "İlgili bilgi bulunamadı.", ""
    except Exception as e:
        st.error(f"Retriever hatası: {{e}}");
        return f"Bilgi alınırken hata oluştu.", ""

# Prompt Şablonu (RAG için)
# Bu şablon, LLM'e nasıl davranması gerektiğini söyler (Sistem Talimatı).
# Not: template_str_app içindeki {{{{ }}}} (4 küme parantezi)
# Python'un f-string formatlamasından (Colab tarafı) kaçmak içindir.
template_str_app = '''Sen Türk Hukuku alanında uzman bir yapay zeka asistanısın...
Bağlam:
{{{{context}}}}
Soru:
{{{{question}}}}
Yanıt:
'''
# *** HATANIN OLDUĞU YORUM SATIRI DÜZELTİLDİ ***
# Buradaki .replace() işlemi, şablonu LangChain'in anlayacağı
# standart formata (yani 'context' ve 'question' anahtarlarına) getirir.
prompt_template_lc = PromptTemplate.from_template(template_str_app.replace('{{{{', '{{').replace('}}}}', '}}'))

# --- Streamlit Arayüzü Başlangıcı ---
st.set_page_config(page_title="🇹🇷 Türk Hukuku RAG Chatbot", page_icon="⚖")
st.title("⚖ Türk Hukuku RAG Chatbot")
st.caption(f"Veri Seti: {{DATASET_ID_STR}} (HF) | Vektör DB: Yerel Chroma | Model: {{MODEL_NAME_LLM}}")

# --- YENİ: Gemini Güvenlik Ayarları (Safety Settings) ---
# Gemini modelleri varsayılan olarak katı güvenlik filtrelerine sahiptir.
# Hukuk metinleri bazen (örn: ceza hukuku) hassas/şiddet içeren terimler
# barındırabileceğinden, modelin cevap vermesini engellememesi için
# filtreleri 'BLOCK_ONLY_HIGH' (Sadece Yüksek Olasılıklı Zararı Engelle)
# seviyesine çekiyoruz.
safety_settings = {{
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}}
# --------------------------------------------------------

# LLM (Dil Modeli) Yükleme Fonksiyonu
# '@st.cache_resource': Tıpkı DB gibi, LLM modelini de önbelleğe alır.
# Bu, sayfa her yenilendiğinde modelin API'den tekrar çekilmesini engeller.
@st.cache_resource
def initialize_llm():
    try:
        # GÜNCELLENDİ: Modeli, belirlediğimiz 'safety_settings' ile başlatıyoruz.
        llm_model = genai.GenerativeModel(
            model_name=MODEL_NAME_LLM,
            generation_config={{"temperature": 0.3}}, # Yaratıcılığı düşük tut (tutarlı cevaplar için)
            safety_settings=safety_settings
            )
        st.success(f"Gemini modeli ({{MODEL_NAME_LLM}}) başarıyla yüklendi (Güvenlik: BLOCK_ONLY_HIGH).")
        return llm_model
    except Exception as e:
        st.error(f"LLM yüklenirken hata oluştu: {{e}}"); return None

# Ana bileşenleri (LLM ve DB) yükle
llm_model = initialize_llm()
chroma_collection = get_chroma_collection()

# Eğer LLM veya DB yüklenemezse, uygulama hata verip durur.
if llm_model is None or chroma_collection is None:
    st.error("Ana kaynaklar (LLM veya Vektör DB) yüklenemedi. Uygulama durduruluyor.")
    st.stop()

# RAG Cevap Fonksiyonu (Ana Mantık)
# Kullanıcı sorgusunu alıp RAG sürecini işleten fonksiyon.
def get_response_from_rag(user_query):
    try:
        # 1. Bilgiyi Çek (Retrieve)
        # DB'den ilgili metinleri (context) ve kaynakları (sources_str) al
        retrieved_context, sources_str = retrieve_context(user_query, chroma_collection)

        # 2. Prompt'u Hazırla (Augment)
        # Kullanıcı sorusu ve çekilen bilgiyi şablona yerleştir
        formatted_prompt = prompt_template_lc.format(question=user_query, context=retrieved_context)

        # 3. Cevap Üret (Generate)
        # Hazırlanan prompt'u LLM'e gönder
        response = llm_model.generate_content(formatted_prompt) # safety_settings=safety_settings)

        try:
            # Cevabın metnini al
            answer = response.text

            # GÜNCELLEME: Güvenlik Engeli Kontrolü
            # Eğer 'answer' boşsa VE 'prompt_feedback' bir 'block_reason' (Engelleme Nedeni)
            # içeriyorsa, bu, cevabın Gemini güvenlik filtresine takıldığını gösterir.
            if not answer and response.prompt_feedback.block_reason:
                 st.warning(f"⚠️ Yanıt güvenlik nedeniyle engellendi: {{response.prompt_feedback.block_reason}}")
                 return "Üzgünüm, ürettiğim yanıt güvenlik politikalarımız nedeniyle engellendi."

            # 4. Kaynakları Cevaba Ekle
            # Eğer kaynak bulunduysa ve cevap "bulunamadı" değilse, kaynakları cevabın sonuna ekle.
            if sources_str and "bulunamadı" not in answer:
                 answer += f"\\n\\n---\\n*Kaynaklar (İlgili Orijinal Sorular):*\\n{{sources_str}}"
            return answer

        except Exception as resp_e:
            st.error(f"Yanıt (response) işlenirken hata: {{resp_e}}")
            return "Yanıt alınırken bir hata oluştu."

    except Exception as e:
        st.error(f"RAG süreci hatası: {{e}}")
        return f"Üzgünüm, cevap üretilirken genel bir hata oluştu."

# --- Chat Arayüzü Mantığı ---

# Chat geçmişini Streamlit'in 'session_state' hafızasında tut
if "messages" not in st.session_state:
    st.session_state.messages = [{{ "role": "assistant", "content": "Merhaba! Türk hukuku hakkında ne öğrenmek istersiniz?" }}] # Başlangıç mesajı

# Geçmişteki mesajları ekrana yazdır
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni giriş (prompt) al
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # 1. Kullanıcının mesajını geçmişe ve ekrana ekle
    st.session_state.messages.append({{"role": "user", "content": prompt}})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Asistanın cevabını hazırla
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Cevap gelene kadar boş bir alan ayır
        with st.spinner("Düşünüyor... (Veritabanı taranıyor ve cevap üretiliyor)"):
            # Ana RAG fonksiyonunu çağır
            assistant_response = get_response_from_rag(prompt)

        # 3. Asistanın cevabını ekrana ve geçmişe ekle
        message_placeholder.markdown(assistant_response)
    st.session_state.messages.append({{"role": "assistant", "content": assistant_response}})
"""
# --- 'app.py' Dosyasını Yazma ---

# Yukarıda f-string ile hazırladığımız kod içeriğini ('streamlit_app_code')
# 'app.py' adında gerçek bir dosyaya yazıyoruz.
APP_PY_CREATED = False
try:
    # dataset_id'nin tanımlı olduğundan emin ol
    if 'dataset_id' not in locals(): dataset_id = "Renicames/turkish-law-chatbot"

    with open("app.py", "w", encoding="utf-8") as f: f.write(streamlit_app_code)
    print("✅ app.py dosyası başarıyla oluşturuldu/güncellendi (Hatalı yorum düzeltildi).")
    APP_PY_CREATED = True
except Exception as e: print(f"❌ app.py dosyası yazılırken hata oluştu: {e}")
print("-" * 50)


# ==============================================================================
# Adım 8: Streamlit Arayüzünü Çalıştırma (ngrok ile)
# ==============================================================================
# Bu adım, 'app.py' dosyasını çalıştırır ve 'ngrok' kullanarak
# Colab üzerinde çalışan bu uygulamaya herkesin erişebileceği
# geçici bir genel (public) URL oluşturur.
print("\n--- Adım 8: Streamlit Arayüzünü Çalıştırma ---")

# Gerekli kütüphaneler (Bu adım için)
import subprocess
import threading
from pyngrok import ngrok, conf
import time
# Colab secrets importu Adım 2'de yapıldı, userdata tanımlı olmalı
# from google.colab import userdata
import os

NGROK_READY = False
# Önceki adımların başarı bayraklarını (flag) kontrol et
db_created_flag = 'DB_CREATED_OR_LOADED' in locals() and DB_CREATED_OR_LOADED
rag_ready_flag = 'RAG_READY' in locals() and RAG_READY
app_py_created_flag = 'APP_PY_CREATED' in locals() and APP_PY_CREATED

# SADECE tüm önceki adımlar (DB, RAG, app.py) başarılıysa devam et
if db_created_flag and rag_ready_flag and app_py_created_flag:
    print("Tüm adımlar başarılı. Ngrok ve Streamlit başlatılıyor...")
    NGROK_AUTH_TOKEN = None

    # Ngrok'u kullanmak için bir 'Authtoken' gerekir.
    # Bu token, ngrok.com dashboard'undan ücretsiz alınabilir.
    # Token'ı koda yazmamak için kullanıcıdan manuel girmesini istiyoruz.
    NGROK_AUTH_TOKEN = input("Lütfen Ngrok Authtoken'ınızı Girin (URL: https://dashboard.ngrok.com/get-started/your-authtoken ): ")

    if not NGROK_AUTH_TOKEN:
        print("❌ Ngrok token girilmedi. Streamlit başlatılamıyor.")
    else:
        try:
            # Ngrok'u alınan token ile ayarla
            ngrok.set_auth_token(NGROK_AUTH_TOKEN); print("✅ ngrok authtoken ayarlandı.")

            # Colab ortamında kalabilecek eski işlemleri temizle
            try:
                subprocess.run(["killall", "-q", "streamlit"], check=False); print("Mevcut Streamlit işlemleri durduruldu.")
                ngrok.kill(); print("Mevcut Ngrok tünelleri kapatıldı."); time.sleep(3) # Sistemlerin kapanması için kısa bekleme
            except:
                pass # Hata verirse (örn: işlem yoksa) es geç

            # Streamlit'i Ayrı Bir Thread'de (İş Parçacığı) Başlatma
            # Streamlit sunucusu 'run' komutuyla başlatıldığında ana Colab
            # hücresini kilitler. Bunu engellemek için 'threading' kullanıyoruz.
            def run_streamlit():
                print("Streamlit sunucusu arka planda (thread) başlatılıyor...")
                if os.path.exists("app.py"):
                    # 'subprocess.Popen' ile streamlit komutunu çalıştır
                    # '--server.port=8501' : Streamlit'in çalışacağı port
                    # '--server.headless=true' : Tarayıcıyı otomatik açmamasını söyler
                    # 'stdout/stderr=subprocess.DEVNULL' : Çıktıları gizler (Colab'i kirletmemesi için)
                    subprocess.Popen(["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    print("❌ app.py dosyası bulunamadı, Streamlit başlatılamadı.")

            thread = threading.Thread(target=run_streamlit);
            thread.start() # Thread'i başlat

            print("Streamlit thread'i başlatıldı, ngrok tüneli için 7sn bekleniyor...");
            time.sleep(7) # Streamlit sunucusunun ayağa kalkması için zaman tanıyoruz

            # Ngrok Tünelini Aç
            # Ngrok'a, yerelde 8501 portunda çalışan uygulamayı dışarıya açmasını söylüyoruz.
            public_url = ngrok.connect(8501)
            print("="*70 + f"\n✅✅✅ Streamlit arayüzü hazır! Erişmek için tıklayın: {public_url}\n" + "="*70)
            NGROK_READY = True

        except Exception as e:
            print(f"❌ ngrok veya Streamlit başlatılırken kritik hata: {e}")
else:
    # Eğer önceki adımlardan biri başarısızsa, nedenini listele
    print("❌ Önceki adımlarda hata olduğu için Streamlit başlatılamıyor.")
    if not db_created_flag:
        print("   - Neden: Adım 5 (Vektör Veritabanı) başarıyla tamamlanmadı.")
    if not rag_ready_flag:
        print("   - Neden: Adım 6 (RAG Pipeline) başarıyla tamamlanmadı.")
    if not app_py_created_flag:
        print("   - Neden: Adım 7 (app.py oluşturma) başarıyla tamamlanmadı.")
print("-" * 50)

# ==============================================================================
# Adım 9: requirements.txt Dosyası Oluşturma
# ==============================================================================
# Bu adım, projenin çalışması için gereken tüm Python paketlerini
# ve versiyonlarını içeren bir 'requirements.txt' dosyası oluşturur.
# Bu dosya, projenin başka bir ortamda (örn: sunucu)
# kolayca kurulabilmesi için standart bir yöntemdir.
if NGROK_READY:
     print("\n--- Adım 9: requirements.txt Dosyası Oluşturuluyor ---")
     # Projede kullanılan ana kütüphanelerin listesi ve spesifik versiyonları
     requirements_content = """google-generativeai~=0.8.5
chromadb~=0.5.4
datasets~=2.20.0
langchain-text-splitters~=0.3.11
langchain-core~=0.3.11
langchain-community~=0.2.19
streamlit~=1.39.0
sentence_transformers~=3.0.1
langchain~=0.2.22
python-dotenv~=1.0.1
pyngrok~=7.1.6
pandas"""
     try:
         # İçeriği 'requirements.txt' dosyasına yaz
         with open("requirements.txt", "w") as f:
             f.write(requirements_content.strip())
         print("✅ requirements.txt dosyası başarıyla oluşturuldu.")
         # 'cat' komutu ile dosyanın içeriğini Colab'de göster (kontrol amaçlı)
         get_ipython().system('cat requirements.txt')
     except Exception as e:
         print(f"❌ requirements.txt dosyası yazılırken hata: {e}")
     print("-" * 50)
     print("🏁 Tüm Adımlar Tamamlandı.")
     print("➡️ Chatbot'u kullanmak için yukarıdaki Streamlit (ngrok) linkini kullanabilirsiniz.")
else:
    print("🏁 Adımlar tamamlandı ancak Streamlit arayüzü başlatılamadı (Detaylar Adım 8'de).")

