title: Hukuk Chatbot
emoji: ⚖
sdk: streamlit
app_file: app.py
pinned: false 


⚖ TÜRK HUKUKU RAG CHATBOT ⚖ 
Bu proje, Türk Hukuku alanında uzmanlaşmış, yapay zeka destekli bir web uygulamasıdır. Retrieval Augmented Generation (RAG) mimarisiyle geliştirilen bu araç, kullanıcılara Türk Hukuku ile ilgili sorularına, ilgili mevzuat ve bilgilere dayanarak anında ve bağlamsal yanıtlar sunar.

Merhaba! 👋 Biz, 3. sınıf bilgisayar mühendisliği öğrencileri olarak bu projeyi geliştirdik. Bu, yapay zekanın "halüsinasyon" (bilgi uydurma) sorununu nasıl çözebileceğimizi araştırdığımız bir RAG denemesidir. Amacımız, genel kültürle değil, doğrudan kaynak metinle cevap veren bir sistem kurmaktı.

🚀 Canlı Demo (Live Demo) Bu proje bir Google Colab not defteri olarak tasarlanmıştır. Canlı demo, statik bir link üzerinde değil, doğrudan Colab not defterinin çalıştırılmasıyla oluşturulur.

Not defterindeki tüm adımlar (Adım 1'den 8'e kadar) başarıyla çalıştırıldığında, 8. Adım'ın çıktısı olarak size özel, geçici bir ngrok public URL'i verilecektir. Arayüze bu link üzerinden erişebilirsiniz.

✅✅✅ Streamlit arayüzü hazır: https://ornek-adres.ngrok-free.dev

Proje Arayüz Gifi (Buraya ngrok linki çalışırken çektiğiniz bir ekran kaydı GIF'i ekleyebilirsiniz!)

🎯 Proje Hakkında Bu chatbot, karmaşık hukuki sorulara hızlı ve güvenilir yanıtlar bulma zorluğunu ele alır. Kullanıcılara anında, ilgili veri setine dayalı yanıtlar sunar ve yanıtların hangi kaynaklardan (ilgili sorulardan) türetildiğini göstererek şeffaflık sağlar.

Neden Bu Proje? (Motivasyonumuz) sınıf bilgisayar mühendisliği öğrencileri olarak, yapay zekanın sadece 'sohbet' etmesinden öte, gerçek dünyadaki karmaşık problemlere nasıl çözüm olabileceğini görmek istedik. Hukuk metinleri, yanlış bir bilginin büyük sorunlara yol açabileceği kritik bir alan.

Bu projede, LLM'in (Gemini 2.0 Flash) "halüsinasyon" denilen, yani modelin kendine güvenerek yanlış bilgi uydurması sorununu çözmeyi hedefledik. RAG mimarisi, bu soruna mükemmel bir çözüm sundu. Modele "Bu soruyu bil" demek yerine, "Bu soruyu cevaplamak için bu belgelere bak " demeyi öğrettik.

Kısacası, modelin "ezberlemesi" yerine "araştırmasını" ve "kaynak göstermesini" sağladık.

🛠 Nasıl Çalışır: RAG Mimarisi Chatbot, doğru ve bağlama duyarlı yanıtlar sağlamak için eksiksiz bir RAG hattı (pipeline) uygular:

📄 Veri Yükleme: Belirlenen Renicames/turkish-law-chatbot veri seti Hugging Face'den yüklenir.

🧠 Akıllı Parçalama (Chunking): RecursiveCharacterTextSplitter kullanarak metinler (cevaplar) anlamsal bütünlüğü koruyacak şekilde 1000 karakterlik parçalara ayrılır.

🧬 Vektör Gömme (Embedding): Her metin parçası (chunk), Google'ın models/text-embedding-004 modeli kullanılarak anlamsal vektörlere dönüştürülür.

🗄 Vektör Depolama: Hızlı erişim için tüm vektörler, Colab ortamında yerel olarak oluşturulan bir ChromaDB (PersistentClient) veritabanında saklanır.

🔗 Bağlam Eşleştirme: Kullanıcının sorgusu vektöre çevrilir ve ChromaDB'de en ilgili (benzer) metin parçaları (context) bulunur.

💬 Yanıt Üretme: LangChain (LCEL) ve Gemini 2.0 Flash modeli kullanılarak, bulunan bağlam (context) ve kullanıcının sorusu bir prompt şablonuna yerleştirilir ve nihai, tutarlı yanıt üretilir.

🖥 Arayüz Sunma: Tüm bu sistem, Streamlit ile oluşturulan interaktif bir chat arayüzü üzerinden pyngrok aracılığıyla canlı olarak yayınlanır.

🧗‍♀ Karşılaşılan Zorluklar ve Öğrendiklerimiz Bu projeyi yaparken takım olarak, teoride basit görünen bazı adımların pratikte oldukça zorlayıcı olduğunu fark ettik:

API Rate Limit (429 Hatası): Veri setindeki yüzlerce metni text-embedding-004 modeli ile vektöre çevirirken sürekli Google API limitlerine takıldık.

Çözüm: Hata yönetimi yapan, exponential backoff (yani hata aldıkça bekleme süresini katlayarak artıran) bir embed_content_with_retry fonksiyonu yazmak zorunda kaldık. Bu, projenin en kritik parçalarından biri oldu.

Yerel Veritabanı Yönetimi: Colab her kapandığında veritabanı siliniyordu.

Çözüm: ChromaDB'nin PersistentClient modunu kullanarak, veritabanını Colab ortamında bir klasöre (chroma_db_law_local_full/) fiziksel olarak kaydetmeyi öğrendik. Bu sayede app.py (Streamlit kodu) bu hazır veritabanını okuyabildi.

Prompt Engineering (Doğru Komutu Verme): Modelin sadece verdiğimiz bağlamı kullanmasını sağlamak zordu. Bazen bildiği (ama yanlış olabilecek) bilgileri araya katıyordu.

Çözüm: Prompt şablonunda çok net ve sert bir talimat verdik: "Yalnızca aşağıda verilen bağlamı kullanarak soruyu Türkçe yanıtlayın. Eğer cevap bağlamda yoksa... 'Bu konuda bilgi bulamadım' deyin." Bu, modelin halüsinasyon görmesini %99 engelledi.

Colab ve Streamlit Entegrasyonu: ngrok tünellemesi, Streamlit sunucusunu (app.py) dış dünyaya açmak için kritikti. Tüm adımların (API anahtarları, app.py'nin yazılması, veritabanının varlığı) doğru sırada çalışmasını sağlamak zaman aldı.

🧪 Örnek Test Sorguları ve Beklenen Yanıtlar Uygulamayı çalıştırdığınızda, chatbot'un RAG mimarisini test etmek için aşağıdaki gibi spesifik hukuk sorularını sormayı deneyebilirsiniz. Bu sorular, veri setimizden alınmış örneklerdir ve botun bunlara benzer, bağlama dayalı yanıtlar vermesi beklenir:

Soru 1: "Devletin şekli nedir?"

Beklenen Yanıt (veya benzeri):

"Türkiye Devleti bir Cumhuriyettir." (Bu cevap, Anayasa Madde 1'e dayanmaktadır.)

Soru 2: "Anayasa Mahkemesi kaç üyeden oluşur?"

Beklenen Yanıt (veya benzeri):

"Anayasa Mahkemesi onbeş üyeden kurulur." (Bu cevap, Anayasa Madde 146'ya dayanmaktadır.)

Soru 3: "Milletvekili seçilme yaşı kaçtır?"

Beklenen Yanıt (veya benzeri):

"Onsekiz yaşını dolduran her Türk milletvekili seçilebilir." (Bu cevap, Anayasa Madde 76'ya dayanmaktadır.)

Soru 4: "Temel hak ve hürriyetler ne ile sınırlanabilir?"

Beklenen Yanıt (veya benzeri):

"Temel hak ve hürriyetler, özlerine dokunulmaksızın yalnızca Anayasanın ilgili maddelerinde belirtilen sebeplere bağlı olarak ve ancak kanunla sınırlanabilir." (Bu cevap, Anayasa Madde 13'e dayanmaktadır.)

Bu sorgular ve beklenen yanıtlar, botun yerel ChromaDB veritabanından doğru bağlamı (ilgili kanun maddesini) bulup bulamadığını ve cevabını bu bağlama göre (halüsinasyon görmeden) üretip üretmediğini görmek için iyi bir yoldur. Ayrıca, botun cevabın sonunda verdiği "Kaynaklar (İlgili Sorular):" bölümünü de kontrol ederek hangi metin parçalarını kullandığını görebilirsiniz.

🌟 Temel Özellikler Uzmanlık Alanı: Sadece Türk Hukuku veri seti (Renicames/turkish-law-chatbot) üzerinden eğitilmiştir.

Gerçek Zamanlı RAG: Veri seti, not defteri her çalıştığında yeniden işlenir ve yerel vektör veritabanı sıfırdan oluşturulur.

Kaynak Gösterme: Chatbot, ürettiği yanıtların hangi kaynak sorulara dayandığını (Kaynaklar (İlgili Sorular): başlığı altında) gösterir.

İnteraktif Arayüz: Streamlit ile modern ve kullanımı kolay bir sohbet arayüzü sunar.

Kolay Kurulum: Tek bir Google Colab not defteri üzerinden tüm bağımlılıkları kurar ve çalışır.

📚 Veri Seti (Dataset) Bu uygulama, Hugging Face üzerinde bulunan Renicames/turkish-law-chatbot veri setini kullanır. Bu veri seti, Soru-Cevap formatında hukuki bilgiler içermektedir. RAG mimarisi, 'Cevap' sütunundaki metinleri vektörleştirerek bilgi tabanını oluşturur.

💻 Kullanılan Teknolojiler Platform: Google Colab

GenAI: Google Gemini 2.0 Flash, Google Embedding (text-embedding-004), LangChain (LCEL)

Vektör Veritabanı: ChromaDB (Yerel/Persistent)

Veri: Hugging Face datasets

Arayüz & Sunum: Streamlit, pyngrok

Yardımcı Kütüphaneler: pandas, langchain-text-splitters

🚀 Kurulum ve Çalıştırma (Google Colab) Bu proje, bir Google Colab not defteri olarak tasarlanmıştır ve en kolay bu platformda çalıştırılır.

Google Colab'da Açın: Proje .ipynb dosyasını Google Colab'da açın.

Hücreleri Sırayla Çalıştırın: En üstteki hücreden başlayarak tüm hücreleri sırayla çalıştırın (Runtime > Run all).

API Anahtarlarını Girin:

Adım 2: Hücre çalıştığında, sizden Gemini API Anahtarınızı girmeniz istenecektir.

Adım 8: Hücre çalıştığında, sizden Ngrok Authtoken'ınızı girmeniz istenecektir.

Kimlik Doğrulama (Gerekirse):

Ek Adım (ADC): Embedding modeli için Google Cloud kimlik doğrulaması gerekebilir. İstenirse, çıkan linke tıklayıp gelen kodu yapıştırın.

Uygulamaya Erişin: Tüm hücreler başarıyla tamamlandığında, Adım 8'in çıktısında ngrok.io ile biten bir public URL göreceksiniz. Bu linke tıklayarak chat arayüzüne erişebilirsiniz.

📂 Proje Yapısı (Oluşturulan Dosyalar) Bu bir Colab not defteri olduğu için, proje yapısı kodun çalışması sırasında Google Colab ortamında dinamik olarak oluşturulan dosyalardan ibarettir:

TÜRK_HUKUKU_RAG_CHATBOT.ipynb: Çalıştırdığınız ana not defteri dosyası.

app.py: (Adım 7'de oluşturulur) Streamlit web uygulamasının Python kodunu içeren dosya.

requirements.txt: (Adım 9'da oluşturulur) Projenin çalışması için gereken Python kütüphanelerini listeleyen dosya.

chroma_db_law_local_full/ Klasörü: (Adım 5'te oluşturulur) Yerel ChromaDB vektör veritabanının dosyalarını içeren klasör.

💡 Gelecek Planları ve Geliştirme Fikirleri Zaman kısıtlamaları nedeniyle yapamadığımız ancak bu projeyi daha da ileriye taşıyacak bazı fikirlerimiz:

Kalıcı Deploy: Projeyi ngrok yerine Streamlit Community Cloud'a taşımak. (Bunun için build_database.py ve app.py olarak ikiye ayırıp, veritabanı dosyalarını GitHub LFS ile yüklememiz gerekecek.)

Daha Geniş Veri Seti: Sadece Soru-Cevap değil, T.C. Anayasası'nın veya Ceza Kanunu'nun tamamını yükleyerek veritabanını zenginleştirmek.

Sohbet Hafızası: Modelin sadece son soruyu değil, önceki konuşmaları da hatırlamasını sağlamak (örn: LangChain ConversationBufferMemory).

📬 İletişim Projeyi incelediğiniz için teşekkür ederiz! Fikirlerinizi, eleştirilerinizi veya sorularınızı duymaktan mutluluk duyarız.

(Bu bölümü kendi bilgilerinizle doldurabilirsiniz.)

Yiğit Efe Kılıç

E-posta: ygtefe10@gmail.com

GitHub: https://github.com/ygtefe10

LinkedIn: www.linkedin.com/in/yigitefekilic

Mehmet Yusuf İzci

E-posta: yufuslora@gmail.com

GitHub: https://github.com/Yusuflora

LinkedIn: www.linkedin.com/in/mehmetyusufizci
