#include "core_cpu.h"
#include "core_gpu.cuh"

#include <iostream>
#include <cstdlib>
#include "json11.h"
#include <algorithm>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace liteqwen{

    std::map<int, std::vector<int>> logits_mask_template_except_ids;
    std::map<int, std::string> logits_mask_names;
    std::map<int, float> logits_mask_template_base_values;
    std::map<int, float> logits_mask_template_except_values;

    int ceil_divide(int a, int b) {
        return a/b + (a % b != 0);
    }

    std::string join(std::vector<std::string> const &strings, std::string delim)
    {
        if (strings.empty()) {
            return std::string();
        }

        return std::accumulate(strings.begin() + 1, strings.end(), strings[0],
            [&delim](std::string x, std::string y) {
                return x + delim + y;
            }
        );
    }

    std::string trim(std::string str) {
        const std::string whiteSpaces = " \t\n\r\f\v";
        // Remove leading whitespace
        size_t first_non_space = str.find_first_not_of(whiteSpaces);
        str.erase(0, first_non_space);
        // Remove trailing whitespace
        size_t last_non_space = str.find_last_not_of(whiteSpaces);
        str.erase(last_non_space + 1);
        return str;
    }

    template<typename ... Args>
    std::string string_format( const std::string& format, Args ... args )
    {
        int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
        if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
        auto size = static_cast<size_t>( size_s );
        std::unique_ptr<char[]> buf( new char[ size ] );
        std::snprintf( buf.get(), size, format.c_str(), args ... );
        return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
    }


    IndexType get_index_type(size_t max_index, bool is_signed=false) {
        if (is_signed) {
            if (max_index > 2147483647) {
                return liteqwen::IndexType::LONG_IDX;
            } else {
                return liteqwen::IndexType::INT32_IDX;
            }
        } else {
            if (max_index > 4294967295) {
                return liteqwen::IndexType::ULONG_IDX;
            } else {
                return liteqwen::IndexType::UINT32_IDX;
            }
        }
    }

    inline std::string DtypeToString(DataType v)
    {
        switch (v)
        {
            case DataType::FLOAT32:   return std::string("fp32");
            case DataType::FLOAT16:   return std::string("fp16");
            case DataType::INT32: return std::string("int");
            default:      return std::string("other");
        }
    }

    std::pair<int, int> GetUintInfo(liteqwen::DataType dtype) {
        int unitSize;
        int unitSizeDiv;
        if (dtype == liteqwen::DataType::FLOAT32) {
            unitSize = 4;
            unitSizeDiv = 1;
        } else if (dtype == liteqwen::DataType::BFLOAT16 || dtype == liteqwen::DataType::INT16 || dtype == liteqwen::DataType::FLOAT16) {
            unitSize = 2;
            unitSizeDiv = 1;
        } else if (dtype == liteqwen::DataType::INT32PARAM || dtype == liteqwen::DataType::INT32) {
            unitSize = 4;
            unitSizeDiv = 1;
        } else if (dtype == liteqwen::DataType::INT8) {
            unitSize = 1;
            unitSizeDiv = 1;
        } else if (dtype == liteqwen::DataType::INT64) {
            unitSize = 8;
            unitSizeDiv = 1;
        } else if (dtype == liteqwen::DataType::INT4 || dtype == liteqwen::DataType::INT4_NOZERO) {
            unitSize = 1;
            unitSizeDiv = 2;
        } else if (dtype == liteqwen::DataType::INT2) {
            unitSize = 1;
            unitSizeDiv = 4;
        } else if (dtype == liteqwen::DataType::BIT) {
            unitSize = 1;
            unitSizeDiv = 8;
        } 
        return std::pair<int, int>(unitSize, unitSizeDiv);
    }

    void Data::UpdateUnitSize() {
        std::pair<int, int> uint_info = GetUintInfo(this->dtype);
        this->unitSize = uint_info.first;
        this->unitSizeDiv = uint_info.second;

        // if (this->dtype == DataType::FLOAT32) {
        //     this->unitSize = 4;
        //     this->unitSizeDiv = 1;
        // } else if (this->dtype == DataType::BFLOAT16 ||
        //         this->dtype == DataType::INT16 ||
        //         this->dtype == DataType::FLOAT16) {
        //     this->unitSize = 2;
        //     this->unitSizeDiv = 1;
        // } else if (this->dtype == DataType::INT32PARAM || this->dtype == DataType::INT32) {
        //     this->unitSize = 4;
        //     this->unitSizeDiv = 1;
        // } else if (this->dtype == DataType::INT8) {
        //     this->unitSize = 1;
        //     this->unitSizeDiv = 1;
        // } else if (this->dtype == DataType::INT4 || this->dtype == DataType::INT4_NOZERO) {
        //     this->unitSize = 1;
        //     this->unitSizeDiv = 2;
        // } else if (this->dtype == DataType::INT2) {
        //     this->unitSize = 1;
        //     this->unitSizeDiv = 4;
        // } else if (this->dtype == DataType::BIT) {
        //     this->unitSize = 1;
        //     this->unitSizeDiv = 8;
        // }  else if (this->dtype == DataType::INT64) {
        //     this->unitSize = 8;
        //     this->unitSizeDiv = 1;
        // }
    }

    void print_cpu_row(std::string row_info, liteqwen::DataType dtype, uint8_t* data, size_t row_id, int cols, int print_width) {
        size_t offset = row_id * cols;
        bool only_one_num = (print_width == 0);
        if (only_one_num) {
            print_width=1;
        }

        printf((row_info+std::string("|")).c_str());
        if (dtype == liteqwen::DataType::FLOAT32) {
            float* generic_data = reinterpret_cast<float*>(data);
            for (int i=0; i< print_width; i++) {
                printf("%f|", generic_data[offset + i]);
            }
            if (print_width < cols && !only_one_num) {
                printf("...");
                size_t end_offset = offset + cols - print_width;
                for (int j=0; j< print_width; j++) {
                    printf("%f|", generic_data[end_offset + j]);
                }
            }
            printf("\n");
        } else if (dtype == liteqwen::DataType::INT32) {
            int* generic_data = reinterpret_cast<int*>(data);
            for (int i=0; i< print_width; i++) {
                printf("%i|", generic_data[offset + i]);
            }
            if (print_width < cols && !only_one_num) {
                printf("...");
                size_t end_offset = offset + cols - print_width;
                for (int j=0; j< print_width; j++) {
                    printf("%i|", generic_data[end_offset + j]);
                }
            }
            printf("\n");
        } else if (dtype == liteqwen::DataType::FLOAT16) {
            __half* generic_data = reinterpret_cast<__half*>(data);
            for (int i=0; i< print_width; i++) {
                printf("%f|", __half2float(generic_data[offset + i]));
            }
            if (print_width < cols && !only_one_num) {
                printf("...");
                size_t end_offset = offset + cols - print_width;
                for (int j=0; j< print_width; j++) {
                    printf("%f|", __half2float(generic_data[end_offset + j]));
                }
            }
            printf("\n");            
        } else {
            printf("cpu print only supports fp16, fp32 and int\n");
            return;
        }
    }
    
    Data::Data(liteqwen::DataType type) {
        this->dtype = type;
        this->strides = std::vector<size_t>();
        this->UpdateUnitSize();
        this->shape = std::vector<int>();
    }

    Data::Data(liteqwen::DataType type, const std::vector<int> &shape) {
        this->dtype = type;
        this->strides = std::vector<size_t>();
        this->UpdateUnitSize();
        reshape(shape); // shapes must be > 0
        this->gpu_id=-1;
    }

    Data::Data (DataType type, const std::vector <int> &shape, int gpu_id, bool managed) {
        this->dtype = type;
        this->strides = std::vector<size_t>();
        this->UpdateUnitSize();
        reshape(shape); // shapes must be > 0
        this->gpu_id = gpu_id;
        this->managed = managed;
    }

    Data::Data (DataType type, const std::vector <int> &shape, int gpu_id, void* origData, size_t original_offset) {
        this->dtype = type;
        this->strides = std::vector<size_t>();
        this->UpdateUnitSize();
        reshape(shape); // shapes must be > 0
        this->gpu_id = gpu_id;
        this->is_nested = true;
        size_t int8_offset = original_offset * this->unitSize / this->unitSizeDiv;
        if (this->gpu_id < 0) {
            this->cpuData = (uint8_t*)(origData) + int8_offset;
        } else {
            this->cudaData = (uint8_t*)origData + int8_offset;
        }
    }

    Data::Data(const Data& ori, bool shallow) {
        this->is_nested = shallow;
        CopyFrom(ori);
    }

    Data::Data(const Data &ori) {
        CopyFrom(ori);
    }

    void Data::CopyFrom(const Data& ori, bool shallow) {
        this->is_nested = shallow;
        this->CopyFrom(ori);
    }

    void Data::CopyFrom(const Data &ori) {

        this->dtype = ori.dtype;
        this->gpu_id = ori.gpu_id;
        this->managed = ori.managed;
        this->UpdateUnitSize();

        if (ori.shape != this->shape) {
            this->reshape(ori.shape);
        }

        if (this->is_nested) {
            this->cudaData = ori.cudaData;
            this->cpuData = ori.cpuData;
            return;
        }

        size_t numel = this->numel();
        if (numel > 0) {
            this->Allocate();
            size_t cpy_size = sizeof(uint8_t) * this->unitSize / this->unitSizeDiv * numel;
            if (this->gpu_id < 0) {
                // printf("copying cpu Data: device %i, uint8_size=%i, copying size %i=%i x %i, %p->%p\n", this->gpu_id, (int)sizeof(uint8_t), (int)cpy_size, (int)(sizeof(uint8_t) * this->unitSize / this->unitSizeDiv), numel, ori.cpuData, this->cpuData);
                std::memcpy(this->cpuData, ori.cpuData, cpy_size);
            } else {
                printf("copying cuda Data: device %i, uint8_size=%lu, copying size %i=%i x %lu, %p->%p\n", this->gpu_id, (int)sizeof(uint8_t), (size_t)cpy_size, (int)(sizeof(uint8_t) * this->unitSize / this->unitSizeDiv), numel, ori.cudaData, this->cudaData);
                CopyGPUData(this->dtype, this->cudaData, ori.cudaData, this->gpu_id, 0, 0, this->numel(), false);
            }
        } 
        // else {
            // printf("copying empty data: device %i, uint8_size=%i\n", this->gpu_id, (int)sizeof(uint8_t));
        // }
    }

    Data::~Data() {
        // std::cout << "delete, cpu addr: " << this->cpuData << ", cuda addr: " << this->cudaData << std::endl;
        if (this->is_nested) {
            return;
        }
        // if (this->gpu_id==-1 && this->cpuData != nullptr) {
        //     delete[] this->cpuData;
        // }

        // 自动清除tensor理论上不应被调用，所有cudaData应该被手动释放，以确保高效。
        if (this->gpu_id>=0 && this->cudaData != nullptr) {
            size_t numel = this->numel();
            int64_t num_dim = this->shape.size();
            std::vector<std::string> tensor_shape;
            std::string delim = ",";
            for (int64_t di=0; di<num_dim; di++){
                int size = shape.at(di);
                tensor_shape.push_back(std::to_string(size));
            }
            std::string shape_print = liteqwen::join(tensor_shape, delim);
            printf("destroying data?=%i, gpu_id=%i, cuda_ptr=%p, shape=[%s]\n", 1-(int)(this->is_nested), this->gpu_id, this->cudaData, shape_print.c_str());
            CudaFree(this->cudaData, this->managed, this->gpu_id);
        }
    }

    size_t Data::shape_prod(const std::vector <int> &shape, int start_dim) {
        size_t prod = 0;
        for (int i=0; i<(int)(shape.size()); i++) {
            if (i==start_dim) {
                prod = static_cast<size_t>(shape[i]);
            } else {
                prod = prod* static_cast<size_t>(shape[i]);
            }
        }
        return prod;
    }

    void Data::Init(DataType type, const std::vector <int> &shape, int gpu_id){
        this->Init(type, shape, gpu_id, true);
    }

    void Data::Init(DataType type, const std::vector <int> &shape, int gpu_id, bool managed) {
        this->dtype = type;
        this->strides = std::vector<size_t>();
        this->UpdateUnitSize();
        reshape(shape); // shapes must be > 0
        this->gpu_id = gpu_id;
        this->managed = managed;
    }

    void Data::Reallocate(int new_gpu_id) {
        if (this->gpu_id != new_gpu_id) {
            int old_id = this->gpu_id;
            if (this->cudaData != nullptr && old_id >= 0) {
                // printf("reallocate freeing old cuda memory at %i, ", old_id);
                SetDevice(old_id);
                if (!this->is_nested) {
                    CudaFree(this->cudaData, this->managed, old_id);
                }
            }
            if (new_gpu_id >= 0) {
                // printf("reallocate new cuda memory at %i\n", new_gpu_id);
                SetDevice(new_gpu_id);
                this->gpu_id = new_gpu_id;
                this->Allocate();
            }
        } else if (new_gpu_id>=0){
            // printf("reallocate new cuda memory at %i\n", new_gpu_id);
            // SetDevice(new_gpu_id);
            this->gpu_id = new_gpu_id;
            this->Allocate();
        }
    }

    void Data::Reallocate(int new_gpu_id, bool do_allocate) {
        if (this->gpu_id != new_gpu_id) {
            int old_id = this->gpu_id;
            if (this->cudaData != nullptr && old_id >= 0) {
                // printf("reallocate freeing old cuda memory at %i, ", old_id);
                SetDevice(old_id);
                if (!this->is_nested) {
                    CudaFree(this->cudaData, this->managed, old_id);
                }
            }
            if (new_gpu_id >= 0) {
                // printf("reallocate new cuda memory at %i\n", new_gpu_id);
                SetDevice(new_gpu_id);
                this->gpu_id = new_gpu_id;
                this->Allocate();
            }
        } else if (new_gpu_id>=0 && do_allocate){
            // printf("reallocate new cuda memory at %i\n", new_gpu_id);
            // SetDevice(new_gpu_id);
            this->gpu_id = new_gpu_id;
            this->Allocate();
        }
    }

    void Data::Reallocate(int new_gpu_id, bool do_allocate, size_t numel) {
        if (this->gpu_id != new_gpu_id) {
            int old_id = this->gpu_id;
            if (this->cudaData != nullptr && old_id >= 0) {
                // printf("reallocate freeing old cuda memory at %i, ", old_id);
                SetDevice(old_id);
                if (!this->is_nested) {
                    CudaFree(this->cudaData, this->managed, old_id);
                }
            }
            if (new_gpu_id >= 0) {
                // printf("reallocate new cuda memory at %i\n", new_gpu_id);
                SetDevice(new_gpu_id);
                this->gpu_id = new_gpu_id;
                this->Allocate(this->gpu_id, this->dtype, numel);
            }
        } else if (new_gpu_id>=0 && do_allocate){
            // printf("reallocate new cuda memory at %i\n", new_gpu_id);
            // SetDevice(new_gpu_id);
            this->gpu_id = new_gpu_id;
            this->Allocate(this->gpu_id, this->dtype, numel);
        }
    }

    void Data::ToDevice(int new_gpu_id) {
        if (new_gpu_id == this->gpu_id || new_gpu_id < -1) {
            return;
        }
        if (new_gpu_id >=0 && this->gpu_id >= 0) {
            int old_gpu_id = this->gpu_id;
            void* new_device_tensor = GetDtypeCudaMalloc(new_gpu_id, this->dtype, this->numel(), this->managed);
            CopyBetweenGPUs(this->dtype, new_gpu_id, new_device_tensor, this->gpu_id, this->cudaData, this->numel());
            SetDevice(old_gpu_id);
            if (!this->is_nested) {
                CudaFree(this->cudaData, this->managed, old_gpu_id);
            }
            this->cudaData = new_device_tensor;
            this->gpu_id = new_gpu_id;
            // 切换device后deepcopy了内容，旧依赖关系消失。
            this->is_nested = false;
            SetDevice(new_gpu_id);
        } else if (this->gpu_id < 0 && new_gpu_id >=0) {
            UploadData(this->dtype, this->cudaData, this->cpuData, new_gpu_id, 0, 0, this->numel());
            // if (! this->is_nested) {
            //     delete[] this->cpuData;
            // }
            this->gpu_id = new_gpu_id;
            this->cpuData = nullptr;
        } else {
            DownloadData(this->dtype, this->cpuData, this->cudaData, 0, 0, this->numel());
            if (! this->is_nested) {
                CudaFree(this->cudaData, this->managed, this->gpu_id);
            }
            this->gpu_id = new_gpu_id;
            this->cudaData = nullptr;
        }
    }

    void Data::reshape(const std::vector <int> &shape) {
        size_t shape_numel = this->shape_prod(shape, 0);
        size_t cur_numel = this->numel();
        if (cur_numel > 0 && cur_numel != shape_numel) {
            throw(std::runtime_error("not same shape exception, could not be reshaped."));
        }
        while(!this->strides.empty()) {
            this->strides.pop_back();
        }
        
        for (int i=0; i<(int)(shape.size())-1;i++) {
            size_t stride = this->shape_prod(shape, i+1);
            this->strides.push_back(stride);
        }
        this->strides.push_back(1);
        this->shape = shape;
    }

    void Data::unsafe_reshape(std::vector<int> shape) {
        while(!this->strides.empty()) {
            this->strides.pop_back();
        }
        
        for (int i=0; i<(int)(shape.size())-1;i++) {
            size_t stride = this->shape_prod(shape, i+1);
            this->strides.push_back(stride);
        }
        this->strides.push_back(1);
        this->shape = shape;  
    }

    size_t Data::numel() {
        // return this->shape_prod(this->shape, 0);
        if (this->shape.size() > 0) {
            return this->strides[0] * this->shape[0];
        } else {
            return 0;
        }
    }
    size_t Data::numel() const {
        // return this->shape_prod(this->shape, 0);
        if (this->shape.size() > 0) {
            return this->strides[0] * this->shape[0];
        } else {
            return 0;
        }
    }
    size_t Data::get_stride(int dim) {
        return this->strides[dim];
    }
    size_t Data::get_stride(int dim) const {
        return this->strides[dim];
    }

    void Data::Allocate(int gpu_id, DataType dtype, size_t numel) {
        if (this->is_nested) {
            return;
        }        
        if (numel < this->numel()) {
            throw(std::runtime_error("unable to allocate memory smaller than a tensor's numel*sizeof(dtype)"));
        }
        if (gpu_id >= 0) {
            this->cudaData = GetDtypeCudaMalloc(gpu_id, dtype, numel, this->managed);
        } else {
            this->cpuData = (new uint8_t[numel * this->unitSize / this->unitSizeDiv]);
            // std::cout << "allocating, cpu addr: " << this->cpuData << std::endl;
        }
    }

    void Data::Allocate(size_t numel) {
        if (this->is_nested) {
            return;
        }        
        if (numel < this->numel()) {
            throw(std::runtime_error("unable to allocate memory smaller than a tensor's numel*sizeof(dtype)"));
        }
        if (gpu_id >= 0) {
            this->cudaData = GetDtypeCudaMalloc(this->gpu_id, this->dtype, numel, this->managed);
        } else {
            this->cpuData = (new uint8_t[numel * this->unitSize / this->unitSizeDiv]);
            // std::cout << "allocating, cpu addr: " << this->cpuData << std::endl;
        }       
    }

    void Data::Allocate() {
        if (this->is_nested) {
            return;
        }
        if (this->gpu_id >= 0) {
            this->cudaData = GetDtypeCudaMalloc(this->gpu_id, this->dtype, this->numel(), this->managed);
        } else {
            size_t ten_size = this->numel() * this->unitSize / this->unitSizeDiv * sizeof(uint8_t);
            this->cpuData = (new uint8_t[this->numel() * this->unitSize / this->unitSizeDiv]);
            // std::cout << "allocating, cpu addr: " << this->cpuData << std::endl;
        }
    }

    void Data::InplaceAppend(Data delta_value, size_t offset) {
        if (delta_value.gpu_id != this->gpu_id) {
            printf("appending tensors on two devices %i and %i not allowed.", this->gpu_id, delta_value.gpu_id);
            throw("device error");
        }
        if (delta_value.dtype != this->dtype) {
            printf("appending tensors of two dtypes %i and %i not allowed.", (int)(this->dtype), (int)(delta_value.dtype));
            throw("dtype error");
        }
        if (delta_value.get_stride(0) + offset > this->get_stride(0)) {
            printf("appending leads to memory leak, numel= %lu+%lu > %i not allowed.", delta_value.get_stride(0), offset, this->gpu_id);
            throw("memory error");
        }
        if (this->gpu_id >= 0) {
            CopyGPUData(this->dtype, this->cudaData, delta_value.cudaData, this->gpu_id, offset, 0, delta_value.numel(), false);
        } else {
            printf("not implemented cpu append");
            throw("not implemented error");
        }
    }

    void Data::InplaceAppend(Data delta_value, size_t offset, size_t read_offset, size_t copy_len) {
        if (delta_value.gpu_id != this->gpu_id) {
            printf("appending tensors on two devices %i and %i not allowed.", this->gpu_id, delta_value.gpu_id);
            throw("device error");
        }
        if (delta_value.dtype != this->dtype) {
            printf("appending tensors of two dtypes %i and %i not allowed.", (int)(this->dtype), (int)(delta_value.dtype));
            throw("dtype error");
        }
        if (delta_value.get_stride(0) + offset > this->get_stride(0)) {
            printf("appending leads to memory leak, numel= %lu+%lu > %i not allowed.", delta_value.get_stride(0), offset, this->gpu_id);
            throw("memory error");
        }
        if (this->gpu_id >= 0) {
            CopyGPUData(this->dtype, this->cudaData, delta_value.cudaData, this->gpu_id, offset, read_offset, copy_len, false);
        } else {
            printf("not implemented cpu append");
            throw("not implemented error");
        }
    }

    void Data::UploadIntValues(size_t numel, size_t offset, int* cpu_values) {
        if (this->dtype != DataType::INT32) {
            printf("unsupported upload from int32 to other dtype tensors");
            throw("");
        }
        if (gpu_id >= 0) {
            uint8_t* casted = reinterpret_cast<uint8_t*>(cpu_values);
            // UploadData(DataType::INT32, this->cudaData, casted, this->gpu_id, offset, 0, numel);
            // offset+numel之后的内容pad为0，防止embedding出错。
            UploadInt32(this->cudaData, casted, this->gpu_id, offset, 0, numel, this->numel(), true);
        } else {
            int* casted = reinterpret_cast<int*>(this->cpuData);
            for (size_t i=0; i<numel; i++) {
                casted[i] = cpu_values[i];
            }
        }
    }

    void Data::UploadValues(size_t numel, size_t offset, uint8_t* cpu_values, DataType src_dtype) {
        // printf("this dtype=%i, parsing dtype=%i\n", (int)this->dtype, (int)src_dtype);
        if (this->dtype != DataType::FLOAT32 && this->dtype != DataType::FLOAT16 && this->dtype != DataType::INT32) {
            printf("not implemented conversion from float input to tensor dtype, quantization load TODO in the future");
            throw("");
        }
        if (this->dtype != src_dtype) {
            printf("loading dtype is not same as data dtype, please make convert before uploading.");
            throw("");
        }

        if (gpu_id >= 0) {
            UploadData(this->dtype, this->cudaData, cpu_values, this->gpu_id, offset, 0, numel);
        } else {
            size_t int8_numel = this->unitSize / this->unitSizeDiv * numel;
            for (int i=0; i<int8_numel;i++) {
                this->cpuData[i] = cpu_values[i];
            }
        }        
    }

    void Data::Fp32CpuToFp16Upload(int gpu_id, float* values) {
        if (values == nullptr) {
            printf("upload source data cannot be nullptr\n");
            throw("data error");
        }
        if (this->cudaData == nullptr || this->dtype != DataType::FLOAT16) {
            printf("unable to upload to an tensor not allocated with gpu of dtype fp16");
            throw("data error");
        }
        UploadCastFp32ToFp16Data(this->cudaData, values, gpu_id, 0, 0, this->numel());
        this->gpu_id = gpu_id;
        this->dtype = DataType::FLOAT16;
    }

    void Data::gpu_arange(int limit) {
        if (this->gpu_id <0) {
            printf("error: gpu_arange should be called by tensor on gpus.\n");
            throw("device error");
        }
        SetDevice(this->gpu_id);
        FillArange(this->cudaData, limit);
    }

    void Data::const_values_fill(double value) {
        if (this->gpu_id <0) {
            printf("error: gpu_arange should be called by tensor on gpus.\n");
            throw("device error");
        }
        SetDevice(this->gpu_id);
        ConstantFill(this->cudaData, this->dtype, this->numel(), value);
    }

    // void Data::fill_logits_mask(float value, std::vector<int> except_ids, float except_value, bool protect_eos) {
    //     size_t limit = this->numel();
    //     const int eos_id = 2;

    //     float* cpu_value = new float[limit];
    //     for (size_t i=0; i<limit; i++) {
    //         if (protect_eos && i == eos_id) {
    //             continue;
    //         }
    //         cpu_value[i] = value;
    //     }
    //     for (int i2=0; i2< except_ids.size(); i2++) {
    //         if (i2 < limit) {
    //             int except_id = except_ids[i2];
    //             if (except_id < 0) {
    //                 // logits_mask的vocab id <0,表示使用了logits_mask模板
    //                 if (logits_mask_template_except_ids.find(except_id) != logits_mask_template_except_ids.end()) {
    //                     for (int j=0; j<limit; j++) {
    //                         if (protect_eos && j == eos_id) {
    //                             continue;
    //                         }
    //                         cpu_value[j] = logits_mask_template_base_values[except_id];
    //                     }
    //                     for (int j2=0; j2<logits_mask_template_except_ids[except_id].size(); j2++) {
    //                         cpu_value[logits_mask_template_except_ids[except_id][j2]] = logits_mask_template_except_values[except_id]; 
    //                     }
    //                 } else {
    //                     char* logits_mask_path = std::getenv("LITEQWEN_MASK_DIR"); // 应该已经在python代码中设置好资源路径的环境变量。
    //                     if (logits_mask_path != nullptr) {
    //                         auto mask_json_path = std::string(logits_mask_path) + std::string("/logits_masks.json");
    //                         std::ifstream input(mask_json_path.c_str());
    //                         std::string error;
    //                         std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    //                         auto result = json11::Json::parse(content, error);

    //                         for (auto item : result.array_items()) {
    //                             auto template_name = std::string(item["name"].string_value());
    //                             int template_id = item["template_id"].int_value();
    //                             printf("loading and buffering logits_mask template json, name=%s, template_id=%i from %s\n", template_name.c_str(), template_id, mask_json_path.c_str());
    //                             std::vector<int> except_token_ids;
    //                             for (auto tk_id : item["ids"].array_items()) {
    //                                 except_token_ids.push_back(tk_id.int_value());
    //                             }
    //                             float base_value = static_cast<float>(item["default"].number_value());
    //                             float except_value = static_cast<float>(item["value"].number_value());

    //                             logits_mask_template_except_ids[template_id] = except_token_ids;
    //                             logits_mask_template_except_values[template_id] = except_value;
    //                             logits_mask_template_base_values[template_id] = base_value;
    //                             logits_mask_names[template_id] = template_name;
    //                         }
    //                     }

    //                     // 已经加载完模板内的mask规则，可以赋值更新 mask。
    //                     if (logits_mask_template_except_ids.find(except_id) != logits_mask_template_except_ids.end()) {
    //                         for (int j=0; j<limit; j++) {
    //                             if (protect_eos && j == eos_id) {
    //                                 continue;
    //                             }
    //                             cpu_value[j] = logits_mask_template_base_values[except_id];
    //                         }
    //                         for (int j2=0; j2<logits_mask_template_except_ids[except_id].size(); j2++) {
    //                             cpu_value[logits_mask_template_except_ids[except_id][j2]] = logits_mask_template_except_values[except_id]; 
    //                         }
    //                     }
    //                 }
    //             } else {
    //                 // 在模板规则应用完后可以从except_ids后续的ids里修改except_value
    //                 cpu_value[except_id] = except_value;
    //             }
    //         }
    //     }

    //     if (this->dtype == liteqwen::DataType::FLOAT16) {
    //         UploadCastFp32ToFp16Data(this->cudaData, cpu_value, this->gpu_id, 0, 0, this->numel());
    //     } else if (this->dtype == liteqwen::DataType::FLOAT32) {
    //         UploadData(this->dtype, this->cudaData, (uint8_t*)cpu_value, this->gpu_id, 0, 0, this->numel());
    //     } else {
    //         printf("TODO: not implemented fill dtype...");
    //     }
    // }

    void Data::Free() {
        if (!(this->is_nested) && this->cudaData != nullptr) {
            // printf("freeing:%p\n", this->cudaData);
            CudaFree(this->cudaData, this->managed, this->gpu_id);
            this->cudaData = nullptr;
        }
    }

    void Data::Free(bool should_free) {
        if (!should_free) {
            return;
        }
        if (!(this->is_nested) && this->cudaData != nullptr) {
            // printf("freeing:%p\n", this->cudaData);
            CudaFree(this->cudaData, this->managed, this->gpu_id);
            this->cudaData = nullptr;
        }
    }

    void Data::exportFloatMatrix(float *array, int row, int col, std::string file_name)
    {
        std::fstream f;
        f.open(file_name.c_str(), std::ios::out);

        for (size_t y = 0; y < row; y++)
        {
            for (int x = 0; x < col; x++)
            {
                f<< array[y*col + x] <<"|";
            }
            f<<std::endl;
        }
        f.close();
        return;
    }

    void Data::export_2d(std::string filepath) {
        int64_t num_dim  = this->shape.size();
        int row_num = 1;
        int col_num = 1;
        std::vector<std::string> tensor_shape;
        std::string delim = "_";
        for (int64_t di=0; di<num_dim; di++){
            if (di < num_dim-1) {
                row_num *= this->shape[di];
            } else {
                col_num = this->shape[di];
            }
            tensor_shape.push_back(std::to_string(this->shape[di]));
        }
        std::string shape_print = liteqwen::join(tensor_shape, delim);
        printf("downloading tensor shape=%s\n", shape_print.c_str());
        int uint8_size = row_num * col_num * this->unitSize / this->unitSizeDiv;
        int uint8_len = uint8_size / sizeof(uint8_t);
        this->cpuData = new uint8_t[uint8_len];
        DeviceSynchronize();
        DownloadData(this->dtype, this->cpuData, this->cudaData, 0, 0, this->numel());
        if (this->dtype == liteqwen::DataType::FLOAT16) {
            float* printing_data = new float[row_num * col_num];
            printf("converting fp16 to fp32 and export to file=%s\n", (filepath+std::string(".")+shape_print+std::string(".txt")).c_str());
            CPUConvertFp16ToFp32(printing_data, this->cpuData, this->dtype, row_num*col_num);
            this->exportFloatMatrix(printing_data, row_num, col_num, filepath+std::string(".")+shape_print+std::string(".txt"));
            free(printing_data);
        } else if (this->dtype == liteqwen::DataType::FLOAT32) {
            printf("export directly as fp32\n");
            float* casted = (float*)this->cpuData;
            this->exportFloatMatrix(casted, row_num, col_num, filepath+std::string(".")+shape_print+std::string(".txt"));
        }

        free(this->cpuData);
        this->cpuData = nullptr;
        DeviceSynchronize();
    }

    void Data::print() {
        this->print(std::string(""));
    }

    void Data::print(std::string data_name, bool should_prt) {
        if(should_prt) {
            this->print(data_name);
        }
    }

    void Data::const_print(std::string data_name) const {
        size_t numel = this->numel();
        int64_t num_dim = this->shape.size();
        std::vector<std::string> tensor_shape;
        std::string delim = ",";
        for (int64_t di=0; di<num_dim; di++){
            int size = shape.at(di);
            tensor_shape.push_back(std::to_string(size));
        }
        std::string shape_print = liteqwen::join(tensor_shape, delim);

        std::stringstream ss;
        if (this->gpu_id <0) {
            ss << std::hex << this->cpuData;
        } else {
            ss << std::hex << this->cudaData;
        }
        printf((data_name + std::string("(device:") + std::to_string(this->gpu_id) + std::string(",&:")+ ss.str() + std::string(",dtype:") + DtypeToString(this->dtype) + (")[") + shape_print + std::string("]\n")).c_str());
        if (this->gpu_id >=0 && this->cudaData == nullptr) {
            printf("illegal empty cudaData to print\n");
        }
        if (numel > 0) {
            int col_size = (this->shape.back());
            int nrow = numel / col_size;

            if ((int)this->strides.size() == 1) {
                // 1d tensor, only 1 row.
                if (this->gpu_id == -1) {
                    print_cpu_row(std::string("[d0,s0,r0]"), this->dtype, this->cpuData, 0, col_size, std::min(static_cast<unsigned long>(5), numel/2));
                } else {
                    PrintRow(std::string("[d0,s0,r0]") ,this->dtype, this->cudaData, 0, col_size, std::min(static_cast<unsigned long>(5), numel/2));
                }
            } else {
                std::map<int, std::string> printable_rows = std::map<int, std::string>{};
                for (int dim_i=0; dim_i<((int)this->strides.size()-1); dim_i++) {
                    int slice_stride = this->get_stride(dim_i);
                    int per_slice_rows = slice_stride / col_size;
                    int nslice = numel / slice_stride;
                    for (int si=0; si<nslice;si++) {
                        if (si<3 || si >= nslice-3) {
                            int offset = si * slice_stride;
                            int ri = per_slice_rows * si;
                            if (printable_rows.find(ri) == printable_rows.end()) {
                                printable_rows[ri] = string_format<int, int, int>("[d%i,s%i,r%i]", dim_i, si, ri);
                            }
                        }
                    }
                }
                for (int i=0; i<nrow; i++) {
                    auto prt_record = printable_rows.find(i);
                    if (prt_record!= printable_rows.end()) {
                        if (this->gpu_id == -1) { 
                            print_cpu_row(prt_record->second ,this->dtype, this->cpuData, prt_record->first, col_size, std::min(static_cast<unsigned long>(5), numel/2));
                        } else {
                            PrintRow(prt_record->second, this->dtype, this->cudaData, prt_record->first, col_size, std::min(static_cast<unsigned long>(5), numel/2));
                        }
                    }
                }
            }
        }
        printf("==============================\n");
    }

    void Data::check_value(std::string tag) {
        size_t numel = this->numel();
        int64_t num_dim = this->shape.size();
        std::vector<std::string> tensor_shape;
        std::string delim = ",";
        for (int64_t di=0; di<num_dim; di++){
            int size = shape.at(di);
            tensor_shape.push_back(std::to_string(size));
        }
        std::string shape_print = liteqwen::join(tensor_shape, delim);
        std::string prt_str = (std::string("checking value: ") + tag + "["+ shape_print + "]\n");
        printf(prt_str.c_str());
        CheckGPUValues(this->dtype, this->numel(), this->cudaData);
        printf("==============\n");
    }

    void Data::print(std::string data_name) {
        this->print(data_name, 5);
    }

    void Data::print(std::string data_name, int prt_len) {
        // bool print_all_rows = true;
        size_t numel = this->numel();
        int64_t num_dim = this->shape.size();
        std::vector<std::string> tensor_shape;
        std::string delim = ",";
        for (int64_t di=0; di<num_dim; di++){
            int size = shape.at(di);
            tensor_shape.push_back(std::to_string(size));
        }
        std::string shape_print = liteqwen::join(tensor_shape, delim);

        std::stringstream ss;
        if (this->gpu_id <0) {
            ss << std::hex << this->cpuData;
        } else {
            ss << std::hex << this->cudaData;
        }
        printf((data_name + std::string("(device:") + std::to_string(this->gpu_id) + std::string(",&:")+ ss.str() + std::string(",dtype:") + DtypeToString(this->dtype) + (")[") + shape_print + std::string("]\n")).c_str());
        if (this->gpu_id >=0 && this->cudaData == nullptr) {
            printf("illegal empty cudaData to print\n");
        }
        if (numel > 0) {
            int col_size = (this->shape.back());
            int nrow = numel / col_size;
            if ((int)this->strides.size() == 1) {
                // 1d tensor, only 1 row.
                if (this->gpu_id == -1) {
                    print_cpu_row(std::string("[d0,s0,r0]") ,this->dtype, this->cpuData, 0, col_size, std::min(static_cast<unsigned long>(prt_len), numel/2));
                } else {
                    PrintRow(std::string("[d0,s0,r0]") ,this->dtype, this->cudaData, 0, col_size, std::min(static_cast<unsigned long>(prt_len), numel/2));
                }
            } else {
                std::map<int, std::string> printable_rows = std::map<int, std::string>{};
                for (int dim_i=0; dim_i<((int)this->strides.size()-1); dim_i++) {
                    int slice_stride = this->get_stride(dim_i);
                    int per_slice_rows = slice_stride / col_size;
                    int nslice = numel / slice_stride;
                    for (int si=0; si<nslice;si++) {
                        if (si<3 || si >= nslice-3 ) { // || print_all_rows
                            int offset = si * slice_stride;
                            int ri = per_slice_rows * si;
                            if (printable_rows.find(ri) == printable_rows.end()) {
                                printable_rows[ri] = string_format<int, int, int>("[d%i,s%i,r%i]", dim_i, si, ri);
                            }
                        }
                    }
                }
                for (int i=0; i<nrow; i++) {
                    auto prt_record = printable_rows.find(i);
                    if (prt_record!= printable_rows.end()) {
                        if (this->gpu_id == -1) { 
                            print_cpu_row(prt_record->second ,this->dtype, this->cpuData, prt_record->first, col_size, std::min(static_cast<unsigned long>(prt_len), numel/2));
                        } else {
                            PrintRow(prt_record->second, this->dtype, this->cudaData, prt_record->first, col_size, std::min(static_cast<unsigned long>(prt_len), numel/2));
                        }
                    }
                }
            }
        }
        printf("==============================\n");
    }

    void setup_gpu_cublas_handler(int gpu_id) {
        setup_cublas_handler(gpu_id);
    }

    void cpu_embedding_copy(uint8_t* cpu_embeddings, uint8_t* source_embed, int* cpu_input_ids, int lookup_len, int channel) {
        size_t embedding_data_size = 2; // sizeof(half) = 2
        size_t cpy_size = embedding_data_size * channel;
        for (int i=0; i<lookup_len; i++) {
            int token_id = cpu_input_ids[i];
            size_t write_token_offset = embedding_data_size * i * channel;
            size_t read_token_offset = embedding_data_size * token_id * channel;
            // printf("t=%i, token_id=%i, read=%i, write=%i\n", i, token_id, read_token_offset, write_token_offset);
            // for (int j = 0; j<channel; j++) {
            //     cpu_embeddings[write_token_offset+j] = source_embed[read_token_offset+j];
            // }
            uint8_t* write_row = cpu_embeddings + write_token_offset;
            uint8_t* read_row = source_embed + read_token_offset;
            std::memcpy(write_row, read_row, cpy_size);
        }
    }
}