# Araç Tespiti ve Sayımı (YOLOv8 + Deep SORT)

Bu proje, video görüntüleri üzerinde araçları tespit edip sınıflandırmak ve her bir aracı sadece bir kez saymak amacıyla geliştirilmiştir. YOLOv8 nesne algılama modeli ve Deep SORT takip algoritması kullanılmıştır. Sayım sonuçları bir video çıktısı ve CSV dosyası olarak dışa aktarılır.


# Projenin Amacı

Bu çalışmanın amacı, video kayıtları üzerinden otomatik olarak araçları:

- Tespit etmek (car, bus, truck, motorbike, ev),
- Her bir aracı tekil olarak takip etmek,
- Türlerine göre saymak,
- Çıktı olarak işlenmiş bir video ve sayım verisi üretmektir.

Bu sistem, trafik analizi, otopark izleme, güvenlik uygulamaları gibi birçok alanda temel bir çözüm sunar.


# Kullanılan Teknolojiler

- Ultralytics YOLOv8– Nesne algılama
- Deep SORT– Nesne takibi
- OpenCV – Video işleme ve görselleştirme
- Pandas – Sayım verisinin CSV'ye yazılması
- Roboflow – Eğitim veri setinin alınması


# Kurulum

Aşağıdaki kütüphanelerin kurulu olması gerekir:

pip install ultralytics
pip install roboflow
pip install deep_sort_realtime
pip install opencv-python pandas

#  Veri Seti
Veri seti Roboflow üzerinden alınmıştır. Aşağıdaki gibi indirilip YOLOv8 formatında kullanıma hazır hale getirilir:

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("datasets-xd3z8").project("final_detection-wagxz")
dataset = project.version(1).download("yolov8")

#  Model Eğitimi
Eğitim, YOLOv8’in yolov8n.yaml yapılandırması ile gerçekleştirilmiştir:

from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

model.train(
    data="path/to/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.0005,
    patience=10,
    name="last_model"
)
Eğitim sonrası best.pt modeli kullanıma hazırdır.

# Araç Takibi ve Sayımı
Model, bir video üzerinde tahmin yapmak için kullanılır. Deep SORT algoritmasıyla her nesne takip edilir ve yalnızca bir kez sayılır. Her sınıf için sayım bilgileri video üzerine yazılır ve ayrı olarak CSV dosyasına kaydedilir.

Desteklenen sınıflar:

-car
-bus
-truck
-ev
-motorbike

Kod, gerçek zamanlı olarak video çıktısı üretir ve bu çıktıyı .mp4 formatında kaydeder.

#  Proje Dosya Yapısı
.
├── yolov8n.yaml                     # Model yapılandırması
├── best.pt                         # Eğitilmiş YOLOv8 ağırlıkları
├── video_input.mp4                 # Giriş videosu
├── output_annotated.mp4            # İşlenmiş video çıktısı
├── vehicle_counts.csv              # Araç sayım sonuçları
├── count_vehicles()                # Takip ve sayım fonksiyonu
└── README.md                       # Bu dosya


# Notlar
Daha doğru tespitler için yolov8s, yolov8m gibi daha büyük modeller tercih edilebilir.
Deep SORT parametreleri ayarlanarak izleme performansı optimize edilebilir.
Eğitim setiniz farklıysa, sınıf adlarını kodda güncellemeniz gerekebilir.
