# Tugas Besar Pemrosesan Paralel

## Judul
Perkalian Matriks Menggunakan CUDA dengan Grid dan Thread.

## Konsep
Program memanfaatkan GPU NVIDIA untuk melakukan perkalian matriks secara paralel menggunakan CUDA.
Sebagai perbandingan tambahan, digunakan PySpark (Hybrid) untuk membandingkan metode pemrosesan paralel.

## Struktur Paralel
- Grid : 2 Dimensi (64 x 64 = 4.096 blocks)
- Block : 16 x 16 Thread (256 threads per block)
- Data : Matriks 1024 x 1024
- Total Thread : 1.048.576

## Struktur Folder
```
tugas-besar-paralel/
│
├── Tugas_Besar_Paralel_Hybrid.ipynb
│
├── src/
│   └── matrix_multiplication_cuda.cu
│
├── build/
│   └── libmatrix.so
│
├── data/
│   └── matrix_input.txt
│
├── output/
│   ├── hasil_perkalian.txt
│   └── benchmark.txt
│
├── screenshots/
│   └── hasil_running.png
│
└── README.md
```

## Penjelasan Paralelisme

### Grid
```c
dim3 blocksPerGrid(64, 64);
```
- Grid adalah kumpulan dari semua block yang akan dieksekusi
- Berukuran 2 dimensi: 64 x 64 = **4.096 block**
- Dihitung dari: `(1024 + 16 - 1) / 16 = 64` untuk setiap dimensi
- Meng-cover seluruh matriks 1024 x 1024

### Thread
```c
dim3 threadsPerBlock(16, 16);
```
- Setiap block memiliki 16 x 16 = **256 thread**
- Thread disusun 2 dimensi sesuai struktur matriks
- `threadIdx.x` → kolom, `threadIdx.y` → baris

### Setiap Thread Menghitung:
```
C[row][col] = Σ (A[row][k] × B[k][col])  untuk k = 0 sampai 1023
```

Setiap thread menghitung **satu elemen** matriks hasil C secara paralel.
Dengan total **1.048.576 thread**, seluruh elemen matriks dihitung secara bersamaan.

## Compile

Compile executable:
```bash
nvcc src/matrix_multiplication_cuda.cu -o build/matrix_mul
```

Compile shared library:
```bash
nvcc -shared -Xcompiler -fPIC src/matrix_multiplication_cuda.cu -o build/libmatrix.so
```

## Menjalankan Program

Linux:
```bash
./build/matrix_mul
```

Windows:
```bash
build\matrix_mul.exe
```
