# Laporan Proyek Machine Learning - Alfia N. Rakhmatika

## Domain Proyek

Human resources/capital adalah modal utama dari suatu perusahaan untuk berkinerja baik dalam bisnisnya. Bagian dari pengelolaan HR adalah proses rekrutmen (hiring). Proses rekrutmen hadir dari kebutuhan perusahaan untuk mendapatkan talent atau tenaga kerja baru. Namun demikian, walaupun proses ini sangat penting untuk kelangsungan perusahaan, proses rekrutmen tidak menghasilkan income/revenue langsung untuk perusahaan. Kenyataannya, pemilik bisnis ukuran kecil (small business owners) menghabiskan 40% waktu kerjanya untuk hal-hal yang tidak secara langsung menghasilkan income, termasuk rekrutmen. Perusahaan menghabiskan 15%-20% dari gaji karyawan untuk menjalankan proses rekruitmen. Selain itu, proses rekrutmen yang tidak dapat diselesaikan dengan cepat membuat proses ini menjadi proses yang cukup mahal, relatif terhadap revenue yang didapat. 

Beberapa hal yang dapat diterapkan untuk mengefisienkan proses ini dari membuat referral program internal hingga efisiensi dari tahapan proses misalnya dengan melakukan wawancara secara online. Satu hal yang perlu menjadi perhatian bahwa proses rekrutmen hadir dari kebutuhan, sedangkan kebutuhan untuk tenaga kerja baru dapat muncul dari adanya karyawan yang resign. Dari perspektif tersebut, perusahaan dapat juga membuat program untuk meningkatkan retensi dari karyawan. Program retensi yang tepat dapat secara efektif mengurangi kebutuhan rekrutmen.

Program retensi sendiri membutuhkan diagnosis yang tepat terkait dengan penyebab dari turnover karyawan. Hal ini bisa jadi sangat rumit dan time consuming. Salah satu cara yang dapat diterapkan adalah dengan membuat model machine learning yang dapat memprediksi karyawan mana yang diperkirakan akan resign dan HR akan memberikan program retensi yang tepat sasaran untuk hasil dari machine learning tersebut.

1. [Is It Time to Outsource Human Resources?](https://www.entrepreneur.com/article/217866) 
2. [What Does It Cost to Hire an Employee?](https://www.businessnewsdaily.com/16562-cost-of-hiring-an-employee.html) 
3. [How much do recruitment agencies charge?](https://www.hiringpeople.co.uk/blog/how-much-do-recruitment-agencies-charge/)
4. [Employee Retention Strategies: IT Industry](https://scholar.google.co.id/scholar_url?url=https://www.academia.edu/download/30315431/scms_journal_july-september_2012.pdf%23page%3D81&hl=en&sa=X&ei=yIHyYpfbLcm8ywTT0rmoAw&scisig=AAGBfm1bl29qH-zEwHoBwDu48chcFkRmAw&oi=scholarr)

## Business Understanding

### Problem Statements

Pernyataan masalah pada project ini adalah kesulitan untuk memperkirakan karyawan yang akan resign untuk menerapkan program retensi yang tepat dalam rangka menurunkan kebutuhan rekrutmen dan efisiensi biaya operasional HR. 

### Goals

Tujuan project ini adalah membuat model machine learning yang dapat memberikan prediksi karyawan yang akan resign.

### Solution statements

Model machine learning yang akan digunakan adalah:
- Logistic Regression
- Random Forrest
- Artificial Neural Network

Ketiganya akan dievaluasi dengan metrik evaluasi recall karena hasil klasifikasi resign/tidak resign diharapkan dapat dengan tepat memprediksi yang akan resign dan akan menambah biaya rekrutmen jika salah klasifikasi yang akan resign menjadi tidak resign. Dengan demikian, model terbaik adalah model dengan nilai recall paling tinggi, artinya minimum false negatif. Perlu dilakukan juga pemeriksaan lebih lanjut apakah dataset memiliki target imbalance. Jika target tidak seimbang, maka metrik akurasi tidak dapat memberikan gambaran yang objektif/missleading sehingga recall menjadi lebih relevan.

## Data Understanding
Dataset yang digunakan adalah dataset fiksi yang dibuat oleh IBM data scientists.
[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

### Variabel-variabel pada IBM HR Analytics Employee Attrition & Performance dataset adalah sebagai berikut:
- Age : Usia karyawan
- Attrition : Target, apakah karyawan akan resign atau tidak
- BusinessTravel : Seberapa sering karyawan bepergian karena pekerjaan
- DailyRate : Nilai rate harian
- Department : Departement karyawan
- DistanceFromHome : Jarak kantor dari rumah
- Education : tingkat pendidikan
- EducationField : Bidang pendidikan karyawan
- EmployeeCount : Jumlah karyawan
- EmployeeNumber : nomor karyawan
- EnvironmentSatisfaction : tingkat kepuasan terhadap lingkungan
- Gender : Jenis kelamin karyawan
- HourlyRate : Nilai rate per jam
- JobInvolvement : tingkat keterlibatan dalam pekerjaan
- JobLevel : Level pekerjaan
- JobRole : Peran dalam pekerjaan
- JobSatisfaction : tingkat kepuasan pada pekerjaan
- MaritalStatus : Status pernikahan
- MonthlyIncome : Pendapatan bulanan
- MonthlyRate : Nilai rate per bulan
- NumCompaniesWorked : Jumlah perusahaan yang pernah bekerja
- Over18 : Apakah karyawan berusia di atas 18 tahun
- OverTime : Apakah karyawan melaksanakan overtime/lembur
- PercentSalaryHike : Persentasi kenaikan gaji
- PerformanceRating  : hasil penilaian kinerja
- RelationshipSatisfaction  : tingkat kepuasan dalam hubungan
- StandardHours : Jam kerja standard
- StockOptionLevel : Opsi stok karyawan
- TotalWorkingYears  : Lama karyawan bekerja
- TrainingTimesLastYear : Waktu training tahun lalu
- WorkLifeBalance : tingkat keseimbangan antara kehidupan dan pekerjaan
- YearsAtCompany : Lama bekerja di perusahaan
- YearsInCurrentRole : Lama bekerja pada peran saat ini
- YearsSinceLastPromotion : Lama sejak promosi terakhir
- YearsWithCurrManager : Lama waktu dengan manajer yang sekarang

### Exploratory Data Analysis
Tahapan eksplorasi data yang dilakukan:
1. Check Missing value, untuk memastikan apakah diperlukan imputasi. Tidak ditemukan adanya missing value.

![Capture](https://user-images.githubusercontent.com/115720444/195632334-b8271bf4-f903-4786-810c-9fd97e059ca2.JPG)

2. Mengubah nilai pada beberapa kolom dengan nilai integer 0 atau 1 pada kolom yang nilainya Yes dan No untuk memudahkan proses dan visualisasi data.
3. Visualisasi data dengan histogram
4. Menampilkan semua unique valus masing-masing kolom untuk memastikan tidak ada data kotor.
5. Menampilkan statistik deskriptif dari masing-masing target ditemukan pada karyawan yang tidak resign rata-rata usia lebih tinggi, daily rate lebih tinggi, DistanceFromHome lebih dekat/kecil, EnvironmentSatisfaction dan JobSatisfaction lebih tinggi, dan StockOptionLevel lebih tinggi.
6. Visualisasi korelasi tiap data dengan heatmap. Beberapa kolom yang berkorelasi tinggi diantaranya Job level dan total working hours, Monthly income dan Job level, Monthly income dan total working hours, dan Age dan monthly income.
7. Visualisasi data Monthly Income dengan boxplot untuk setiap Job Role. Terlihat setiap Job ROle memiliki rentang Monthly Income yang berbeda-beda.
8. Visualisasi target dengan barchart. Dataset memiliki target yang imbalance.

## Data Preparation
Teknik preparasi data yang dilakukan sebagai berikut:
1. Menghapus kolom yang hanya berisi 1 nilai dan kolom Job Level yang memiliki korelasi sangat tinggi dengan Monthly Income. Kolom-kolom ini hanya akan menambahkan dimensi yang tidak perlu.
2. Memisahkan data kategorikal dan melakukan one hot encoding. Hal ini perlu karena model hanya memahami angka.
3. Memisahkan dataset features dan Target.
4. Melakukan scaling data dengan min max scaler. Hal ini perlu dilakukan karena terdapat perbedaan skala/rentang angka cukup besar pada beberapa kolom, misal kolom data DailyRate angkanya sampai ribuan namun kolom data DistanceFromHome di angka puluhan. Scaling memastikan model akan memperlakukan setiap kolom dengan seimbang.
5. Memisahkan dataset menjadi 80% data train dan 20% data test untuk training model.

## Modeling
Modeling yang dilakukan:
1. Logistic Regression

Model logistic regression memiliki kelebihan:
- Termasuk model yang relatif sederhana, tidak membutuhkan banyak resource komputasi
- Memberikan output nilai probabilitas, bukan hanya hasil klasifikasi saja
- Efisien ketika dataset memiliki feature yang dapat dipisahkan secara linear

Sedangkan kekurangannya adalah:
- Sulit menangkap keterhubungan yang rumit dengan logistic regression
- Memerlukan multicollinearitas yang moderat atau tidak ada sama sekali pada feature data training.
- Sensitif dengan outliers

Hasil metrik dari model Logistic Regression adalah sebagai berikut:
- Accuracy :  0.88
- Precision :  0.81
- Recall :  0.67
- F1 Score :  0.71

2. Random Forrest

Model Random Forrest memiliki kelebihan:
- Cenderung tidak overfit jika cukup banyak tree pada model
- Dapat menangani dataset yang besar dengan dimensionalitas yang besar
- Memiliki metode untuk melakukan balancing error terutama jika dataset imbalance

Sedangkan kekurangannya adalah:
- Banyaknya tree membuat waktu prediksi relatif lama dan kurang tepat diterapkan untuk prediksi realtime
- Membutuhkan resource komputasi yang besar, terutama jika dataset besar

Hasil metrik dari model Random Forrest adalah sebagai berikut:
- Accuracy :  0.86
- Precision :  0.84
- Recall :  0.59
- F1 Score :  0.62

3. Artificial Neural Network

Model Artificial Neural Network memiliki kelebihan:
- Lebih fault tolerant
- Masih dapat bekerja walaupun model tidak mendapatkan data yang lengkap
- Paralel processing

Sedangkan kekurangannya adalah:
- Membutuhkan resource komputasi yang besar
- Sulit menentukan struktur untuk mendapatkan model terbaik

Hasil metrik dari model Artificial Neural Network adalah sebagai berikut:
- Accuracy :  0.86
- Precision :  0.75
- Recall :  0.7
- F1 Score :  0.72


Dari ketiga model tersebut, dari metrik yang telah ditetapkan sebelumnya, maka model terbaik adalah model artificial neural network dengan nilai recall paling baik diantara ketiganya.


## Evaluation
Dari beberapa metrik yang ditampilkan, sebagaimana goal yang telah ditetapkan, maka metrik yang dipilih adalah recall karena pada kasus ini perlu nilai false negatif yang minimal (karyawan diprediksi tidak akan resign padahal akan resign).

Rumus perhitungan recall sebagai berikut:

Recall = TP/ Actual TRUE = TP/ (TP+FN)

Keterangan:
True positives (TP): karyawan akan resign dan diprediksi akan resign 
True negatives (TN): karyawan tidak akan resign dan diprediksi tidak akan resign 
False positives (FP) (Type I error): karyawan tidak akan resign tetapi diprediksi akan resign 
False negatives (FN) (Type II error): karyawan akan resign tetapi diprediksi tidak akan resign 

Dengan demikian, pada project ini, model terbaik adalah model dengan Artificial Neural Network dengan recall paling tinggi yaitu 0.7.

**---Ini adalah bagian akhir laporan---**
