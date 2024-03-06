Projekt dotyczący klasyfikacji zdjęć samochodów osobowych jako uszkodzone lub nie.
Zdjęcia zostały pobrane z otomoto.pl.
Na pierwszym planie był oceniany samochód.
Cały zbiór danych zawierał: 1389 zdjęć aut dobrych i 884 zdjęć aut uszkodzonych.
Do treningu użyto architektury modelu VGG16 z wstępnie wytrenowanymi wagami na zbiorze 'imagenet'.


Przedstawione pliki zawierają:
01_prepare_file_names.py  ->  ujednolicenie plików
02_model_testing.py  ->  sprawdzenie niewytrenowanego modelu (weryfikacja przygotowania zdjęć i wyników modelu)
03_data_splitting.py  ->  podział danych
04_train.py  ->  trening na różnych modelach
05_check_model.py  ->  ocena modelu na zbiorze testowym
06_predict_class.py  ->  klasyfikacja pojedynczego zdjęcia
07_predict_with_heatmap.py  ->  klasyfikacja pojedynczego zdjęcia po segmentacji pojazdu


Przed uruchomieniem:
Zainstaluj wymagane biblioteki z pliku requirements.txt
(pip install -r requirements.txt)


pobierz plik z modelem i etykietami z poniższego adresu:
https://huggingface.co/AdamGie/CV_damaged_car_detection/tree/main
Następnie przenieś je do katalogu "CV_damaged_car_classification/output"
w wyniku powinniśmy otrzymać pliki "model_custom_VGG16_2_classes.hdf5" oraz "labels_2_classes.pickle" w tym katalogu


pobierz plik do detekcji objektów z poniższego adresu:
https://pjreddie.com/media/files/yolov3.weights
Następnie przenieś go do katalogu "CV_damaged_car_classification/functions/YOLO_detection"
w wyniku powinniśmy otrzymać plik "yolov3.weights" w tym katalogu


pobierz plik z poniższego adresu:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
Następnie przenieś go do katalogu "CV_damaged_car_classification/functions/SAM_segmentation"
w wyniku powinniśmy otrzymać plik "sam_vit_h_4b8939.pth" w tym katalogu


Po wykonaniu powyższyk kroków można uruchomić pliki do klasyfikacji.
'06_predict_class.py'  lub  '07_predict_with_heatmap.py'


Przykładowe uruchomienie w terminalu (po przejściu do katalogu projektu):
(python nazwa_pliku -i ścieżka_do_zdjęcia)
python 06_predict_class.py -i example_images/damaged_01.jpg
python 07_predict_with_heatmap.py -i example_images/damaged_01.jpg