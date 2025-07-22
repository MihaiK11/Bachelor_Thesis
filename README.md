
# [RO] Licență/Disertație 2024

Scopul acestui depozit este colectarea lucrărilor de licență.  
Astfel, în cadrul acestui depozit veți găsi următoarele directoare:

- `proiect/` – acest director trebuie să conțină codul sursă al aplicației voastre;
- `lucrare/` – acest director trebuie să conțină lucrarea voastră în format PDF.

## Important:

- **NU** este permisă adăugarea de sub-repository-uri;
- Vă rugăm ca în `proiect/` să **NU încărcați arhive** (`.zip`, `.rar` etc.), ci **DOAR codul sursă**. Nerespectarea acestei instrucțiuni poate duce la ignorarea codului.

## Instrucțiuni

Pentru actualizarea conținutului acestui repository, vă rugăm să realizați o clonă locală folosind `git`, să adăugați/modificați datele local și apoi să realizați `push` înapoi pe server.

---

# Descrierea proiectului: Rezolvarea sistemelor de ecuații liniare cu algoritmi metaeuristici

Acest proiect implementează două algoritmi metaeuristici inspirați din natură – **Ant Lion Optimizer (ALO)** și **Monkey Algorithm (MA)** – pentru rezolvarea sistemelor de ecuații liniare. Problema este tratată ca o problemă de optimizare, în care se urmărește minimizarea erorii pătratice medii dintre soluția estimată și cea exactă a sistemului Ax = b.

## Structura proiectului

- `proiect/antlion.py` – implementarea algoritmului ALO;
- `proiect/monkey.py` – implementarea algoritmului MA;
- `proiect/extracted_matrices/` – folderul în care trebuie plasate fișierele de input (ex. `exemplu.txt`);
- `lucrare/Finala.pdf` – documentația completă (lucrarea de licență).

## Format fișier input (`exemplu.txt`)

```
a11 a12 a13 ...
a21 a22 a23 ...
...
---
b1 b2 ...
```

## Rulare

> Asigurați-vă că fișierul `exemplu.txt` este plasat în `proiect/extracted_matrices/`.

Rulare ALO:
```bash
python proiect/antlion.py
```

Rulare MA:
```bash
python proiect/monkey.py
```

## Rezultate

- Rezultatele detaliate pot fi consultate în `lucrare/Finala.pdf`.

---

# [EN] BSc/MSc Thesis

The purpose of this repository is to collect your graduation theses.  
In this repository, you will find the following directories:

- `proiect/` – this directory **MUST** contain the source code of your application;
- `lucrare/` – this directory **MUST** contain your thesis in PDF format.

## Important:

- You are **NOT ALLOWED** to use sub-repositories;
- Please **DO NOT upload archives**. Upload just the source code. Failure to comply with this instruction might result in your code being ignored.

## Instructions

To update the contents of this repository, please clone it locally using `git`, add or modify your data, and then push the changes.

## References

- https://git-scm.com/docs/gittutorial  
- https://docs.gitlab.com/ee/gitlab-basics/  
- https://www.sourcetreeapp.com/

---

# Project description: Solving Linear Systems with Metaheuristic Algorithms

This project implements two nature-inspired metaheuristics – **Ant Lion Optimizer (ALO)** and **Monkey Algorithm (MA)** – for solving systems of linear equations. The problem is approached as an optimization task by minimizing the mean squared error (MSE) between the predicted solution and the exact result.

## Project structure

- `proiect/antlion.py` – implementation of the ALO algorithm;
- `proiect/monkey.py` – implementation of the MA algorithm;
- `proiect/extracted_matrices/` – directory containing input systems (e.g., `exemplu.txt`);
- `lucrare/Finala.pdf` – the full thesis (PDF).

## Input file format (`exemplu.txt`)

```
a11 a12 a13 ...
a21 a22 a23 ...
...
---
b1 b2 ...
```

## How to run

> Make sure the `exemplu.txt` file is placed inside `proiect/extracted_matrices/`.

Run ALO:
```bash
python proiect/antlion.py
```

Run MA:
```bash
python proiect/monkey.py
```

## Results

- Detailed performance and comparisons are included in `lucrare/Finala.pdf`.
