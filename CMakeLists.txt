cmake_minimum_required(VERSION 3.29)
project(Elaborato_2_Giovanni_Stefanini LANGUAGES CXX CUDA)

# Imposta il percorso di CUDA manualmente (se necessario)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")

# Abilita CUDA e C++
enable_language(CUDA)
enable_language(CXX)

# Specifica il percorso di OpenCV
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

# Aggiunge il codice CUDA al progetto
add_executable(Elaborato_2_Giovanni_Stefanini kernel.cu main.cpp)

# Imposta il compilatore C++ e CUDA per usare C++17
set_target_properties(Elaborato_2_Giovanni_Stefanini PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 20
)

# Linka OpenCV
target_include_directories(Elaborato_2_Giovanni_Stefanini PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Elaborato_2_Giovanni_Stefanini ${OpenCV_LIBS})


