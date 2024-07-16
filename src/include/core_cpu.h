#ifndef CORE_CPU_H
#define CORE_CPU_H

#include <iostream>
#include<fstream>
#include <string>
#include <numeric>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <sstream>

#define BLOCK_SIZE 256

namespace liteqwen{

    enum IndexType {
        INT32_IDX = 0, UINT32_IDX = 1, LONG_IDX=2, ULONG_IDX = 3
    };
    IndexType get_index_type(size_t max_index, bool is_signed);

    enum DataType {
        FLOAT16 = 0, INT32 = 1, INT64 = 2, INT16 = 3, INT8 = 4, INT4 = 5, INT2 = 6, BIT = 7, BFLOAT16=8, FLOAT32 = 9,
        INT4_NOZERO = 10, // 不用zeroPoint的int4, floatValue = min + uint4Value * scale
        INT32PARAM = 100 // int32的参数，这种类型的数据永远存在CPU上
    };
    inline std::string DtypeToString(DataType v);


    std::pair<int, int> GetUintInfo(liteqwen::DataType dtype);

    class Data {
        public:
        DataType dtype = DataType::FLOAT16;
        int unitSize, unitSizeDiv = 1; // 单个元素的字节数 = unitSIze / unitSizeDiv
        std::vector <int> shape;
        std::vector <size_t> strides;
        int gpu_id = -1;
        uint8_t *cpuData; // 数据指针
	    void *cudaData; 
        bool is_nested = false; // 是否依赖于其他Data，如果是，则属于浅拷贝。
        bool managed = true; // 参与预分配管理
        Data () {};
        Data (DataType type);
        Data (DataType type, const std::vector <int> &shape);
        Data (DataType type, const std::vector <int> &shape, int gpu_id, bool managed);
        Data (DataType type, const std::vector <int> &shape, int gpu_id, void* origData, size_t original_offset); // 浅拷贝其他Data，并设置偏移。不需要重复Allocate。
        Data (const Data& ori);
        Data (const Data& ori, bool shallow);
        void CopyFrom(const Data &ori);
        void CopyFrom(const Data& ori, bool shallow);
        ~Data();
        void Init(DataType type, const std::vector <int> &shape, int gpu_id);
        void Init(DataType type, const std::vector <int> &shape, int gpu_id, bool managed);
        void CpuDelete();

        void print();
        void print(std::string data_name);
        void print(std::string data_name, bool should_prt);
        void print(std::string data_name, int prt_len);
        void const_print(std::string data_name) const;

        void ToDevice(int new_gpu_id);
        void UpdateUnitSize();
        void exportFloatMatrix(float *array, int row, int col, std::string file_name);
        void export_2d(std::string filepath);
        size_t numel();
        size_t numel() const;
        size_t get_stride(int dim);
        size_t get_stride(int dim) const;
        void reshape(const std::vector <int> &shape);
        void unsafe_reshape(std::vector<int> shape);

        void Allocate(int device_id, DataType dtype, size_t numel);
        void Allocate(size_t numel);
        void Allocate();
        void Reallocate(int new_gpu_id);
        void Reallocate(int new_gpu_id, bool do_allocate);
        void Reallocate(int new_gpu_id, bool do_allocate, size_t numel);
        void UploadIntValues(size_t numel, size_t offset, int* cpu_values);
        void UploadValues(size_t numel, size_t offset, uint8_t* cpu_values, DataType src_dtype);

        // void fill_logits_mask(float value, std::vector<int> except_ids, float except_value, bool protect_eos);
        void const_values_fill(double value);

        void InplaceAppend(Data delta_value, size_t offset);
        void InplaceAppend(Data delta_value, size_t offset, size_t read_offset, size_t copy_len);
        void Fp32CpuToFp16Upload(int gpu_id, float* values);
        void gpu_arange(int limit);
        void Free();
        void Free(bool should_free);
        void check_value(std::string tag);
        
        private:
        size_t shape_prod(const std::vector <int> &shape, int start_dim);
    };

    std::string join(std::vector<std::string> const &strings, std::string delim);
    std::string trim(std::string in);

    template<typename ... Args>
    std::string string_format( const std::string& format, Args ... args );

    void print_cpu_row(std::string row_info, liteqwen::DataType dtype, uint8_t* data, size_t row_id, int cols, int print_width);

    int ceil_divide(int a, int b);
    
    void setup_gpu_cublas_handler(int gpu_id);
    void SetEmbeddingBuffer(int max_BL, int hidden_size);
    uint8_t* cpu_embedding_copy(uint8_t* read_row, int* cpu_input_ids, int lookup_len, int channel);
    
}

#endif // CORE_CPU_H