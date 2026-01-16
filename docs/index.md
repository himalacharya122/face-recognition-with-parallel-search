# Parallel Face Recognition System

Welcome to the documentation for the **Parallel Face Recognition System**, a project designed to compare and optimize face matching using both **serial** and **parallel** processing techniques in Python.

This system uses **multiprocessing** to improve performance when searching and matching faces across image datasets, while also providing a serial baseline for benchmarking and analysis.

---

## Key Features

- Parallel and serial implementations for face search  
- Smart worker allocation based on CPU and workload  
- Caching system to avoid redundant face encoding  
- Benchmarking tools for performance comparison  
- Efficient image dataset processing  

---

## Purpose of the Project

The goal of this project is to show how parallel computing can significantly reduce execution time for computationally expensive tasks such as face recognition, and to provide a clean, reusable API for experimentation and learning.

---

## Getting Started

Use the navigation menu to explore the API reference for each module:

- `benchmark` — Performance measurement utilities  
- `utils` — Helper functions and cache system  
- `parallel_face_search` — Multiprocessing implementation  
- `serial_face_search` — Sequential implementation  
- `main` — Entry point and orchestration  