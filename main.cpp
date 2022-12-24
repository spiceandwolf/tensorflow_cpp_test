//参考tensorflow 2.0 官方示例
//https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <istream>
#include <map>
#include <memory>
#include <new>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <regex>
#include <mutex>
#include <pthread.h>

#include "cc/framework/scope.h"
#include "core/framework/tensor_shape.h"
#include "core/platform/errors.h"
#include "core/platform/mutex.h"
#include "core/platform/path.h"
#include "core/platform/status.h"
#include "core/platform/str_util.h"
#include "core/platform/tstring.h"
#include "core/public/session_options.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/parsing_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using namespace tensorflow;
using namespace ops;
using namespace tensorflow::str_util;

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::tstring;

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status = 
        tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return ::tensorflow::OkStatus();
}

// 使用正则表达式分割字符串
void Stringsplit(const string& str, const string& split, std::vector<string>* res) {
    std::regex reg(split);  // 匹配split
    std::sregex_token_iterator pos(str.begin(), str.end(), reg, -1);                  
    for (decltype(pos) end; pos != end; ++pos) {    // 自动推导类型
        res->emplace_back(pos->str());
    }
}

// 读取CSV文件，result中存储sql和对应的基数 功能和ReadTensorFromSQL类似
static Status ReadEntireCSV(const string& file_name, 
                            std::vector<std::vector<string>*>* tables, 
                            std::vector<std::vector<string>*>* joins,
                            std::vector<std::vector<std::vector<string>*>*>* predicates,
                            std::vector<string>* labels) {
    std::fstream fs(file_name);
    if (!fs) {
        return tensorflow::errors::NotFound("File ", file_name, " not found.");
    }

    // .csv中每一行数据row，row[0]是表的集合，
    //                      row[1]是连接的集合，
    //                      row[2]是谓词的集合，
    //                      row[3]是标签的集合
    string line;
    // std::regex ns_reg("#"); // 按"#"分割
    // std::regex comman_reg(","); // 按","分割

    while (std::getline(fs, line)) {
        // result->push_back(line);
        std::vector<string> data_raw;
        Stringsplit(line, "#", &data_raw);
        // 分割出tables
        std::vector<string>* table = new std::vector<string>();
        Stringsplit(data_raw[0], ",", table);
        tables->emplace_back(table);
        // 分割出joins
        std::vector<string>* query_joins = new std::vector<string>();
        Stringsplit(data_raw[1], ",", query_joins);
        joins->emplace_back(query_joins);
        // 分割出predicates
            // 先分出的是单独的item
            // 每3个item才组成一个完整的predicate
        std::vector<string>* predicate_items = new std::vector<string>();
        std::vector<std::vector<string>*>* predicate_row = new std::vector<std::vector<string>*>();
        Stringsplit(data_raw[2], ",", predicate_items);
        
        for(int i = 0; i < predicate_items->size(); i = i + 3) { 
            
            std::vector<string>* query_predicates = new std::vector<string>();
            if (predicate_items->size() % 3 != 0) {
                string no_predicate = "no_predicate";
                query_predicates->emplace_back(no_predicate);
                predicate_row->emplace_back(query_predicates);
                break;
            }
            query_predicates->emplace_back(predicate_items->at(i));
            query_predicates->emplace_back(predicate_items->at(i + 1));
            query_predicates->emplace_back(predicate_items->at(i + 2));
            predicate_row->emplace_back(query_predicates);
        }
        predicates->emplace_back(predicate_row);
        // 分割出labels
        if (std::stoll(data_raw[3]) < 1) {
            return tensorflow::errors::InvalidArgument("Queries must have non-zero cardinalities");
        }
        labels->emplace_back(data_raw[3]);
    } 
    
    return ::tensorflow::OkStatus();
}

// 读取bitmaps文件，result中存储对应的bitmaps
static Status ReadEntireBitmaps(const string& file_name, const int num_materialized_samples, 
                                const std::vector<std::vector<string>*>& tables, 
                                std::vector<std::vector<std::vector<tensorflow::uint8>*>*>* result) {
    // tensorflow::uint64 file_size = 0;
    // TF_RETURN_IF_ERROR(env->GetFileSize(file_name, &file_size));
    std::fstream fs(file_name, std::ios::in | std::ios::binary);
    if (!fs) {
        return tensorflow::errors::NotFound("File ", file_name, " not found.");
    }
    
    result->clear();
    // string contents;
    // contents.resize((tensorflow::uint64)4);
    // tensorflow::StringPiece data;
    int num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3);

    // std::unique_ptr<tensorflow::RandomAccessFile> file;
    // TF_RETURN_IF_ERROR(env->NewRandomAccessFile(file_name, &file));
    // int offset = 0;
    
    //读取每个query的bitmaps
    for(int i = 0; i < tables.size(); i++) {
        // data.empty();
        // TF_RETURN_IF_ERROR(file->Read(offset, 4, &data, &(contents)[0]));
        // offset = offset + 4;
        char data[4];
        fs.read(data, 4);
        // 可能需要做反序处理
        // string four_bytes(data.rbegin(), data.rend());
        std::swap(data[0], data[3]);
        std::swap(data[1], data[2]);
        // int num_bitmaps_curr_query = std::stoi(four_bytes.data()); 
        int num_bitmaps_curr_query = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
        
        // tensorflow::uint8[num_bitmaps_curr_query][num_bytes_per_bitmap * 8] bitmaps;
        std::vector<std::vector<tensorflow::uint8>*>* bitmaps = new std::vector<std::vector<tensorflow::uint8>*>();
        //读取每个bitmap
        for(int j = 0; j < num_bitmaps_curr_query; j++) {
            // Read bitmap
            // tensorflow::StringPiece bitmap_bytes;
            // TF_RETURN_IF_ERROR(file->Read(offset, num_bytes_per_bitmap, &bitmap_bytes, &(contents)[0]));
            // offset = offset + num_bytes_per_bitmap;
            char bitmap_bytes[num_bytes_per_bitmap];
            
            fs.read(bitmap_bytes, num_bytes_per_bitmap);
            std::vector<tensorflow::uint8>* bitmap = new std::vector<tensorflow::uint8>(); 
            //把一个bitmap中的每一位以二进制展开
            for(int k = 0; k < num_bytes_per_bitmap; k++) {
                // 以uint8形式二进制展开
                for (int l = 7; l >= 0; l--) {
                    // bitmaps[j][7 - l + 8 * k] = (bitmap_bytes[k] >> l) & 1;
                    bitmap->emplace_back((bitmap_bytes[k] >> l) & 1);
                }               
            }
            bitmaps->emplace_back(bitmap);         
        }
        result->emplace_back(bitmaps);
    }

    return ::tensorflow::OkStatus();
}

// 生成onehot
std::vector<float>* idx_to_onehot(const int idx, const int num_elements) {
    std::vector<float>* onehot = new std::vector<float>();
    for (int i = 0; i < num_elements; i++) {
        if (i == idx) {
            onehot->emplace_back(1.0);
        } else {
            onehot->emplace_back(0.0);
        }
    }
    return onehot;
}

// 获取要编码的集合
    // idx2thing可以直接用c++ set的内部方法取得
std::map<string, int>* get_set_encoding(const std::set<string>& source_set, bool onehot) {
    if (!onehot) {
        int32_t num_elements = source_set.size();
        std::vector<string> source_list;
        source_list.assign(source_set.begin(), source_set.end());
        std::sort(source_list.begin(), source_list.end());
        // Build map from s to i
        std::map<string, int>* thing2idx = new std::map<string, int>();
        int i = 0;
        for(auto iter = source_list.begin(); iter !=  source_list.end(); iter++) {
            thing2idx->insert(std::pair<string, int>(iter->data(), i));
            i++;
        }
        return thing2idx;
    }
    
    return nullptr;
}

// 获取要编码的集合
    // idx2thing可以直接用c++ set的内部方法取得
std::map<string, std::vector<float>*>* get_set_encoding(const std::set<string>& source_set) {
    int32_t num_elements = source_set.size();
    std::vector<string> source_list;
    source_list.assign(source_set.begin(), source_set.end());
    std::sort(source_list.begin(), source_list.end());
    // Build map from s to i
    std::map<string, std::vector<float>*>* thing2vec = new std::map<string, std::vector<float>*>();
    int i = 0;
    for(auto iter = source_list.begin(); iter !=  source_list.end(); iter++) {
        thing2vec->
            insert(std::pair<string, std::vector<float>*>(iter->data(), idx_to_onehot(i, num_elements)));
        i++;
    }
    //进行onehot编码

    return thing2vec;
}

// 获取所有列名
std::set<string>* get_all_column_names(std::vector<std::vector<std::vector<string>*>*>* predicates) {
    std::set<string>* column_names = new std::set<string>();
    for(int i = 0; i < predicates->size(); i++) {
        std::vector<std::vector<string>*>* query = predicates->at(i);
        for(int j = 0; j < query->size(); j++) {
            std::vector<string>* query_predicates = query->at(j);
            if (query_predicates->size() == 3) {
                string column_name = query_predicates->at(0);
                column_names->insert(column_name);
            }
        }
    }

    return column_names;
}

// 获取所有操作符
std::set<string>* get_all_operators(std::vector<std::vector<std::vector<string>*>*>* predicates) {
    std::set<string>* operators = new std::set<string>();
    for(int i = 0; i < predicates->size(); i++) {
        std::vector<std::vector<string>*>* query = predicates->at(i);
        for(int j = 0; j < query->size(); j++) {
            std::vector<string>* query_predicates = query->at(j);
            if (query_predicates->size() == 3) {
                string op = query_predicates->at(1);
                operators->insert(op);
            }
        }
    }

    return operators;
}

// 获取所有表名
std::set<string>* get_all_table_names(std::vector<std::vector<string>*>* tables) {
    std::set<string>* table_names = new std::set<string>();
    for(int i = 0; i < tables->size(); i++) {
        std::vector<string>* query = tables->at(i);
        for(int j = 0; j < query->size(); j++) {
            string table = query->at(j);
            table_names->insert(table);
        }
    }
    return table_names;
}

// 获取所有连接
std::set<string>* get_all_joins(std::vector<std::vector<string>*>* joins) {
    std::set<string>* join_set = new std::set<string>();
    for(int i = 0; i < joins->size(); i++) {
        std::vector<string>* query = joins->at(i);
        for(int j = 0; j < query->size(); j++) {
            string query_joins = query->at(j);
            join_set->insert(query_joins);
        }
    }
    return join_set;
}

// 获取每列的最大最小值 
static Status ReadColumn_min_max_Vals(const string& file_name, 
                                        std::map<string, std::pair<float, float>*>* column_min_max_vals) {
    std::fstream fs(file_name);
    if (!fs) {
        return tensorflow::errors::NotFound("File ", file_name, " not found.");
    }
    
    column_min_max_vals->clear();
    string line;
    std::getline(fs, line);
    while(std::getline(fs, line)) {
        std::vector<string> data_raw;
        Stringsplit(line, ",", &data_raw);
        std::pair<float, float>* column_min_max_val 
            = new std::pair<float, float>(std::stof(data_raw[1]), std::stof(data_raw[2]));
        column_min_max_vals->insert(std::pair<string, std::pair<float, float>*>(data_raw[0], column_min_max_val));
    }

    return ::tensorflow::OkStatus();
}

// 归一化数据
float normalize_data(const float val, const string column_name, 
                const std::map<string, std::pair<float, float>*>* column_min_max_vals) {
    float min_val = column_min_max_vals->at(column_name)->first;
    float max_val = column_min_max_vals->at(column_name)->second;
    float val_norm = 0.0;
    if (max_val > min_val) {
        val_norm = (val - min_val) / (max_val - min_val);
    }
    return val_norm;
}

// 制作dataset,  Add zero-padding and wrap as tensor dataset.
// samples_test, predicates_test, joins_test,
Status make_dataset(std::vector<std::vector<std::vector<float>*>*>* samples, 
                std::vector<std::vector<std::vector<float>*>*>* predicates, 
                std::vector<std::vector<std::vector<float>*>*>* joins, 
                int max_num_joins, 
                int max_num_predicates, 
                std::vector<std::pair<std::string, Tensor>>& out_tensors) {
    
    int num_tensors = samples->size();
    std::vector<std::vector<float>*>* sample_masks = new std::vector<std::vector<float>*>();
    std::vector<std::vector<float>*>* predicate_masks = new std::vector<std::vector<float>*>();
    std::vector<std::vector<float>*>* join_masks = new std::vector<std::vector<float>*>();

    //在vector的形式下pad
    for(int i = 0; i < num_tensors; i++) {     
       
        std::vector<std::vector<float>*>* query_samples = samples->at(i);
        std::vector<std::vector<float>*>* query_predicates = predicates->at(i);
        std::vector<std::vector<float>*>* query_joins = joins->at(i);      

        int sample_size = query_samples->at(0)->size();
        int predicate_size = query_predicates->at(0)->size();
        int join_size = query_joins->at(0)->size();

        //构造mask
        std::vector<float>* sample_mask = new std::vector<float>();
        for(int j = 0; j < query_samples->size(); j++) {
            sample_mask->emplace_back(1);
        }

        std::vector<float>* predicate_mask = new std::vector<float>();
        for(int j = 0; j < query_predicates->size(); j++) {
            predicate_mask->emplace_back(1);
        }
        
        std::vector<float>* join_mask = new std::vector<float>();
        for(int j = 0; j < query_joins->size(); j++) {
            join_mask->emplace_back(1);
        }
        // for (int j = 0; j < query_predicates->size(); j++) {
        //     //float item = query_predicates->at(j);
        //     //predicate_tensors.flat<float>()(j) = item;
        //     predicate_masks.flat<float>()(j) = 1.0;
            
        // }
        // for (int j = 0; j < query_joins->size(); j++) {
        //     //float item = query_joins->at(j);
        //     //join_tensors.flat<float>()(j) = item;
        //     join_masks.flat<float>()(j) = 1.0;
            
        // }

        // using namespace ops;
        // Scope root = Scope::NewRootScope();

        // auto sample_tensor_v = Concat(root.WithOpName("input_sample_tensor"), {sample_tensors}, 0);
        // auto sample_mask_v = Concat(root.WithOpName("input_sample_mask"), {sample_mask_tensors}, 0);
        // auto predicate_tensor_v = Concat(root.WithOpName("input_predicate_tensor"), {predicate_tensors}, 0);
        // auto predicate_mask_v = Concat(root.WithOpName("input_predicate_mask"), {predicate_masks}, 0);
        // auto join_tensor_v = Concat(root.WithOpName("input_join_tensor"), {join_tensors}, 0);
        // auto join_mask_v = Concat(root.WithOpName("input_join_mask"), {join_masks}, 0);

        // 填充sample_mask和sample_tensor
        // 填充predicate_mask和predicate_tensor
        // 填充predicate_mask和predicate_tensor
        int num_pad_sample = max_num_joins + 1 - query_samples->size();       
        // auto sample_tensor_padded = Pad(root.WithOpName("sample_tensor_pad"), sample_tensor_v, {0, num_pad_sample, 0, 0});
        // auto sample_mask_padded = Pad(root.WithOpName("sample_mask_pad"), sample_mask_v, {0, num_pad_sample, 0, 0});
        for(int j = 0; j < num_pad_sample; j++) {
            std::vector<float>* sample_padded = new std::vector<float>();
            for(int k = 0; k < sample_size; k++) {
                sample_padded->emplace_back(0);
            }
            query_samples->emplace_back(sample_padded);
            sample_mask->emplace_back(0);
        }
        sample_masks->emplace_back(sample_mask);

        int num_pad_predicates = max_num_predicates - query_predicates->size();       
        // auto predicate_tensor_padded = Pad(root.WithOpName("predicate_tensor_pad"), predicate_tensor_v, {0, num_pad_predicates, 0, 0});
        // auto predicate_mask_padded = Pad(root.WithOpName("predicate_mask_pad"), predicate_mask_v, {0, num_pad_predicates, 0, 0});
        for(int j = 0; j < num_pad_predicates; j++) {
            std::vector<float>* predicate_padded = new std::vector<float>();
            for(int k = 0; k < predicate_size; k++) {
                predicate_padded->emplace_back(0);
            }
            query_predicates->emplace_back(predicate_padded);
            predicate_mask->emplace_back(0);
        }
        predicate_masks->emplace_back(predicate_mask);

        int num_pad_join = max_num_joins - query_joins->size();       
        // auto join_tensor_padded = Pad(root.WithOpName("join_tensor_pad"), join_tensor_v, {0, num_pad_join, 0, 0});
        // auto join_mask_padded = Pad(root.WithOpName("join_mask_pad"), join_mask_v, {0, num_pad_join, 0, 0});
        for(int j = 0; j < num_pad_join; j++) {
            std::vector<float>* join_padded = new std::vector<float>();
            for(int k = 0; k < join_size; k++) {
                join_padded->emplace_back(0);
            }
            query_joins->emplace_back(join_padded);
            join_mask->emplace_back(0);
        }
        join_masks->emplace_back(join_mask);
    }

    int num_samples = samples->at(0)->size();
    int num_predicates = predicates->at(0)->size();
    int num_joins = joins->at(0)->size();

    int sample_size = samples->at(0)->at(0)->size();
    int predicate_size = predicates->at(0)->at(0)->size();
    int join_size = joins->at(0)->at(0)->size();

    // 构造tensor 
    tensorflow::Tensor sample_tensors(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_tensors, num_samples, sample_size}));
    tensorflow::Tensor sample_mask_tensors(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_tensors, num_samples, 1}));
    tensorflow::Tensor predicate_tensors(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_tensors, num_predicates, predicate_size}));
    tensorflow::Tensor predicate_mask_tensors(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_tensors, num_predicates, 1}));
    tensorflow::Tensor join_tensors(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_tensors, num_joins, join_size}));
    tensorflow::Tensor join_mask_tensors(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_tensors, num_joins, 1}));
        
    for(int i = 0; i < num_tensors; i++) {
        for (int j = 0; j < num_samples; j++) {
            auto sample_tensor_mapped = sample_tensors.tensor<float, 3>();
            auto sample_masks_mapped = sample_mask_tensors.tensor<float, 3>();
            for(int k = 0; k < sample_size; k++){
                const float item = samples->at(i)->at(j)->at(k);
                sample_tensor_mapped(i, j, k) = item;  
            }   
            sample_masks_mapped(i, j, 0) = sample_masks->at(i)->at(j);
        }
        for (int j = 0; j < num_predicates; j++) {
            auto predicate_tensor_mapped = predicate_tensors.tensor<float, 3>();
            auto predicate_masks_mapped = predicate_mask_tensors.tensor<float, 3>();
            for(int k = 0; k < predicate_size; k++){
                const float item = predicates->at(i)->at(j)->at(k);
                predicate_tensor_mapped(i, j, k) = item;  
            }   
            predicate_masks_mapped(i, j, 0) = predicate_masks->at(i)->at(j);
        }
        for (int j = 0; j < num_joins; j++) {
            auto join_tensor_mapped = join_tensors.tensor<float, 3>();
            auto join_masks_mapped = join_mask_tensors.tensor<float, 3>();
            for(int k = 0; k < join_size; k++){
                const float item = joins->at(i)->at(j)->at(k);
                join_tensor_mapped(i, j, k) = item;  
            }   
            join_masks_mapped(i, j, 0) = join_masks->at(i)->at(j);
        }

    }

    // 填充target_tensor 可能用不上
    // auto target_tensor = ZerosLike(root.WithOpName("target_tensor_init"), sample_tensors);
    tensorflow::Tensor target_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_joins, 1}));
    for (int j = 0; j < num_joins; j++) {
        auto target_tensor_mapped = target_tensor.tensor<float, 2>();
           
        target_tensor_mapped(j, 0) = 0;
    }

    //制作inputs
    out_tensors = {
        {"serving_default_samples:0", sample_tensors},
        {"serving_default_predicates:0", predicate_tensors},
        {"serving_default_joins:0", join_tensors},
        {"serving_default_sample_mask:0", sample_mask_tensors},
        {"serving_default_predicate_mask:0", predicate_mask_tensors},
        {"serving_default_join_mask:0", join_mask_tensors},
    };

    return ::tensorflow::OkStatus();
}

// 给data部分编码
void encode_data(std::vector<std::vector<std::vector<string>*>*> * predicates, 
            std::vector<std::vector<string>*>* joins, 
            std::map<string, std::pair<float, float>*>* column_min_max_vals, 
            std::map<string, std::vector<float>*>* column2vec, 
            std::map<string, std::vector<float>*>* op2vec, 
            std::map<string, std::vector<float>*>* join2vec,
            std::vector<std::vector<std::vector<float>*>*>* predicates_enc, 
            std::vector<std::vector<std::vector<float>*>*>* joins_enc
            ) {
    for(int i = 0; i < predicates->size(); i++) {
        std::vector<std::vector<string>*>* query_pred = predicates->at(i);
        std::vector<std::vector<float>*>* pred_vec = new std::vector<std::vector<float>*>();
        for(int j = 0; j < query_pred->size(); j++) {
            std::vector<string>* query_predicates = query_pred->at(j);
            std::vector<float>* items_vec = new std::vector<float>(); 
            if (query_predicates->size() == 3) {
                // Proper query_predicates
                string column = query_predicates->at(0);
                string op = query_predicates->at(1);
                string val = query_predicates->at(2);
                float norm_val = normalize_data(std::stof(val), column, column_min_max_vals);
                for(int k = 0; k < column2vec->at(column)->size(); k++) {
                    items_vec->emplace_back(column2vec->at(column)->at(k));
                }
                for(int k = 0; k < op2vec->at(op)->size(); k++) {
                    items_vec->emplace_back(op2vec->at(op)->at(k));
                }
                items_vec->emplace_back(norm_val);
            } else {
                for (int k = 0; k < column2vec->size() + op2vec->size() + 1; k++) {
                    items_vec->emplace_back(0.0);
                }
            }
            pred_vec->emplace_back(items_vec);
        }
        predicates_enc->emplace_back(pred_vec);
        
        // Join instruction
        std::vector<string>* query_join = joins->at(i);
        std::vector<std::vector<float>*>* join_vec = new std::vector<std::vector<float>*>();
        for(int j = 0; j < query_join->size(); j++) {
            std::vector<float>* query_joins = join2vec->at(query_join->at(j));
            join_vec->emplace_back(query_joins);
        }
        joins_enc->emplace_back(join_vec);
    }
}

// 给label部分编码
void encode_samples(std::vector<std::vector<string>*>* tables,
                    std::vector<std::vector<std::vector<tensorflow::uint8>*>*>* samples, 
                    std::map<string, std::vector<float>*>* table2vec, 
                    std::vector<std::vector<std::vector<float>*>*>* samples_enc) {
    samples_enc->clear();
    //对每个query编码
    for (int i = 0; i < tables->size(); i++) {
        std::vector<string>* query = tables->at(i);
        std::vector<std::vector<tensorflow::uint8>*>* query_samples = samples->at(i);
        std::vector<std::vector<float>*>* query_vec = new std::vector<std::vector<float>*>();
        //对每个query中涉及的每个table进行编码
        for(int j = 0; j < query->size(); j++) {  
            std::vector<float>* sample_vec = new std::vector<float>();
            // Append table one-hot vector
            std::vector<float>* table_vec = table2vec->at(query->at(j));
            for(int k = 0; k < table_vec->size(); k++) {
                sample_vec->emplace_back(table_vec->at(k));
            }
            // Append bit vector
            std::vector<tensorflow::uint8>* bitmap = query_samples->at(j);
            for (int k = 0; k < bitmap->size(); k++) {
                sample_vec->emplace_back(bitmap->at(k));
            }
            query_vec->emplace_back(sample_vec);
        }
        samples_enc->emplace_back(query_vec);
    }
}

// Given an sql , read in the data, try to decode it ,
// decode it to the requested size, and then scale the values as desired.
// Status ReadTensorFromSQL(const string& file_name, const int input_height, 
//                                 const int input_width, const float input_mean,
//                                 const float input_std, std::vector<Tensor>* out_tensors) {
//     auto root = tensorflow::Scope::NewRootScope(); // NOLINT(build/namespaces) https://www.coder.work/article/3331082
//     using namespace ::tensorflow::ops;

//     string input_name = "file_reader";
//     string output_name = "decoded";

//     // read file_name into a tensor named input
//     Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
//     //TF_RETURN_IF_ERROR(ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

//     // use a placeholder to read input data
//     auto file_reader = Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

//     std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{"input", input},};
    
//     // Now try to figure out what kind of file it is and decode it.
//     // 编码sql...
//     // .csv中每一行数据row，row[0]是表的集合，
//     //                      row[1]是连接的集合，
//     //                      row[2]是谓词的集合，
//     //                      row[3]是标签的集合
//     tensorflow::OutputList csv_reader;
    
//     if (EndsWith(file_name, ".csv")) {
//         csv_reader = DecodeCSV(root.WithOpName("CSV_reader"), file_reader, 
//             {tensorflow::Input("tables"), 
//                                 tensorflow::Input("joins"), 
//                                 tensorflow::Input("predicates"), 
//                                 tensorflow::Input((int64_t)1.0)}, 
//                                 tensorflow::ops::DecodeCSV::FieldDelim("#")).output;
//     }

//     // This runs the GraphDef network definition that we've just constructed, and
//     // returns the results in the output tensor.
//     tensorflow::GraphDef graph;
//     TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
//     std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
//     TF_RETURN_IF_ERROR(session->Create(graph));
//     TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));

//     return ::tensorflow::OkStatus();
// }

// Analyzes the q-error of the Inception graph
Status CkeckLablesQError(const std::vector<Tensor>& outputs, int expected, bool* is_expected) {
    *is_expected = false;
    Tensor indices;
    Tensor scores;
    const int how_many_labels = 1;

    // 计算q-error...

    return ::tensorflow::OkStatus();
}

int main(int argc, char* argv[]) {

    // 读取模型前的准备工作
    string JOB_sqls = "tensorflow_c++_test/model/mscn/job-light.csv";
    string JOB_bitmap = "tensorflow_c++_test/model/mscn/job-light.bitmaps";
    string train_sqls = "tensorflow_c++_test/model/mscn/train.csv";
    string train_bitmap = "tensorflow_c++_test/model/mscn/train.bitmaps";
    string column_min_max_vals_file ="tensorflow_c++_test/model/mscn/column_min_max_vals.csv";
    string graph = "tensorflow_c++_test/model/mscn/tfmodel";
    string input_layer = "input";
    string output_layer = "StatefulPartitionedCall:0";
    bool self_test = false;
    string root_dir = "/home/hdd/user1/oblab/";
    std::vector<Flag> flag_list = {
        Flag("sqls", &JOB_sqls, "sqls to be processed"),
        Flag("bitmap", &JOB_bitmap, "bitmap to be processed"),
        Flag("column_min_max_vals_file", &column_min_max_vals_file, "column_min_max_vals_file to be processed"),
        Flag("graph", &graph, "graph to be executed"),
        Flag("input_layer", &input_layer, "name of input layer"),
        Flag("output_layer", &output_layer, "name of output layer"),
        Flag("self_test", &self_test, "run a self test"),
        Flag("root_dir", &root_dir,
           "interpret data and model file names relative to this directory"),
    };

    // 把已有数据转化成cmdline所需参数
    // string usage = tensorflow::Flags::Usage(argv[0], flag_list);
    // const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
    // if (!parse_result) {
    //     LOG(ERROR) << usage;
    //     return -1;
    // } 

    // We need to call this to set up global state for TensorFlow.
    // tensorflow::port::InitMain(argv[0], &argc, &argv);
    // if (argc > 1) {
    //     LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    //     return -1;
    // }

    // 读取SQL语句并转化成用于模型的编码
    // 先读训练集生成min_val, max_val, table2vec, column2vec, op2vec, join2vec
    // 再读测试集生成joins, predicates, tables, samples, label
        // 综述
        // [Learned Cardinality Estimation: A Design Space Exploration and A Comparative Evaluation]，
        // [Learned Cardinality Estimation: An In-depth Study]
        // 里有将SQL转化为csv的方法，并且第2篇里有生成bitmap的方法
    // 从csv和bitmap中读取数据然后编码
    // 对每一个编码成的向量进行预测
    string sqls_path = tensorflow::io::JoinPath(root_dir, train_sqls);
    std::vector<std::vector<string>*> * tables = new std::vector<std::vector<string>*>();
    std::vector<std::vector<string>*> * joins = new std::vector<std::vector<string>*>();
    std::vector<std::vector<std::vector<string>*>*> * predicates = new std::vector<std::vector<std::vector<string>*>*>();
    std::vector<string> * labels = new std::vector<string>();
    int num_materialized_samples = 1000;
    // 从csv中读取数据 注意读取的应该是训练用的csv，这里没改
    Status read_csv_status = ReadEntireCSV(sqls_path, tables, joins, predicates, labels);
    if (!read_csv_status.ok()) {
        LOG(ERROR) << read_csv_status;
        return -1;
    }
    
    // 从bitmap中读取数据
    string bitmap_path = tensorflow::io::JoinPath(root_dir, train_bitmap);
    std::vector<std::vector<std::vector<tensorflow::uint8>*>*>* samples 
        = new std::vector<std::vector<std::vector<tensorflow::uint8> *> *>();
    Status read_bitmap_status = ReadEntireBitmaps(bitmap_path, num_materialized_samples, *tables, samples);
    if (!read_csv_status.ok()) {
        LOG(ERROR) << read_csv_status;
        return -1;
    }

    // Get feature encoding and proper normalization
        // 需要tables, samples, table2vec
        // 需要predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec
        // 需要label, min_val, max_val
    std::map<string, std::vector<float>*>* column2vec = get_set_encoding(*get_all_column_names(predicates));
    std::map<string, std::vector<float>*>* table2vec = get_set_encoding(*get_all_table_names(tables));
    std::map<string, std::vector<float>*>* op2vec = get_set_encoding(*get_all_operators(predicates));
    std::map<string, std::vector<float>*>* join2vec = get_set_encoding(*get_all_joins(joins));

    // std::cout << "column2vec数量" << column2vec->size() << std::endl;
    // std::cout << "table2vec数量" << table2vec->size() << std::endl;
    // std::cout << "op2vec数量" << op2vec->size() << std::endl;
    // std::cout << "join2vec数量" << join2vec->size() << std::endl;
    // std::cout << "第一个column2vec" << std::endl;
    // for (int j = 0; j < column2vec->begin()->second->size(); j++) {
    //     std::cout << column2vec->begin()->second->at(j) << " ";
    // }
    // std::cout <<  std::endl;
    // std::cout << "第一个table2vec" << std::endl;
    // for (int j = 0; j < table2vec->begin()->second->size(); j++) {
    //     std::cout << table2vec->begin()->second->at(j) << " ";
    // }
    // std::cout <<  std::endl;
    // std::cout << "第一个op2vec" << std::endl;
    // for (int j = 0; j < op2vec->begin()->second->size(); j++) {
    //     std::cout << op2vec->begin()->second->at(j) << " ";
    // }
    // std::cout <<  std::endl;
    // std::cout << "第一个join2vec" << std::endl;
    // for (int j = 0; j < join2vec->begin()->second->size(); j++) {
    //     std::cout << join2vec->begin()->second->at(j) << " ";
    // }

    // Get min and max values for each column ：计算column_min_max_vals
    string column_min_max_vals_path = tensorflow::io::JoinPath(root_dir, column_min_max_vals_file);
    std::map<string, std::pair<float, float>*>* column_min_max_vals = new std::map<string, std::pair<float, float>*>();
    Status read_column_min_max_vals_status 
        = ReadColumn_min_max_Vals(column_min_max_vals_path, column_min_max_vals);
    if (!read_column_min_max_vals_status.ok()) {
        LOG(ERROR) << read_column_min_max_vals_status;
        return -1;
    }

    // std::cout << "第一个column_min_max_vals" << std::endl;
    // std::cout << "t.id" << " " << column_min_max_vals->find("t.id")->second->first << " " << (int)column_min_max_vals->find("t.id")->second->second;

    // Load test data
    string test_sqls_path = tensorflow::io::JoinPath(root_dir, JOB_sqls);
    std::vector<std::vector<string>*> * test_tables = new std::vector<std::vector<string>*>();
    std::vector<std::vector<string>*> * test_joins = new std::vector<std::vector<string>*>();
    std::vector<std::vector<std::vector<string>*>*> * test_predicates = new std::vector<std::vector<std::vector<string>*>*>();
    std::vector<string> * test_labels = new std::vector<string>();
    // 从csv中读取数据 注意读取的应该是训练用的csv，这里没改
    read_csv_status = ReadEntireCSV(test_sqls_path, test_tables, test_joins, test_predicates, test_labels);
    if (!read_csv_status.ok()) {
        LOG(ERROR) << read_csv_status;
        return -1;
    }
    // 从bitmap中读取数据
    string test_bitmap_path = tensorflow::io::JoinPath(root_dir, JOB_bitmap);
    std::vector<std::vector<std::vector<tensorflow::uint8>*>*>* test_samples 
        = new std::vector<std::vector<std::vector<tensorflow::uint8> *> *>();
    read_bitmap_status = ReadEntireBitmaps(test_bitmap_path, num_materialized_samples, *test_tables, test_samples);
    if (!read_csv_status.ok()) {
        LOG(ERROR) << read_csv_status;
        return -1;
    }

    // Get feature encoding and proper normalization
    std::vector<std::vector<std::vector<float>*>*>* samples_test = new std::vector<std::vector<std::vector<float>*>*>();
    encode_samples(test_tables, test_samples, table2vec, samples_test);
    std::vector<std::vector<std::vector<float>*>*>* predicates_test = new std::vector<std::vector<std::vector<float>*>*>();
    std::vector<std::vector<std::vector<float>*>*>* joins_test = new std::vector<std::vector<std::vector<float>*>*>();
    encode_data(test_predicates, test_joins, column_min_max_vals, column2vec, op2vec, join2vec, 
                predicates_test, joins_test);

    // std::cout << "第一个samples_test"  << " " << samples_test->at(0)->size() 
    //     << " " << samples_test->at(0)->at(0)->size() << std::endl;
    // for(int i = 0; i < samples_test->at(0)->at(0)->size(); i++) {
    //     std::cout << samples_test->at(0)->at(0)->at(i) << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "第一个predicates_test"  << " " << predicates_test->at(0)->size() 
    //     << " " << predicates_test->at(0)->at(0)->size() << std::endl;
    // for(int i = 0; i < predicates_test->at(0)->at(0)->size(); i++) {
    //     std::cout << predicates_test->at(0)->at(0)->at(i) << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "第一个joins_test"  << " " << joins_test->at(0)->size() 
    //     << " " << joins_test->at(0)->at(0)->size() << std::endl;
    // for(int i = 0; i < joins_test->at(0)->at(0)->size(); i++) {
    //     std::cout << joins_test->at(0)->at(0)->at(i) << " ";
    // }

    int max_num_predicates = 0;
    int max_num_joins = 0;
    for (int i = 0; i < predicates_test->size(); i++) {
        if (max_num_predicates < predicates_test->at(i)->size()) {
            max_num_predicates = predicates_test->at(i)->size();
        }
    }
    for (int i = 0; i < joins_test->size(); i++) {
        if (max_num_joins < joins_test->at(i)->size()) {
            max_num_joins = joins_test->at(i)->size();
        }
    }

    // Get test set predictions
    std::vector<std::pair<std::string, Tensor>> inputs;
    Status make_dataset_status = make_dataset(samples_test, 
                                    predicates_test, 
                                    joins_test, 
                                    max_num_joins, 
                                    max_num_predicates, 
                                    inputs);
    if (!make_dataset_status.ok()) {
        LOG(ERROR) << "Running model failed: " << make_dataset_status;
        return -1;
    }

    std::cout << inputs.at(0).second.DebugString() << std::endl;
    std::cout << inputs.at(1).second.DebugString() << std::endl;
    std::cout << inputs.at(2).second.DebugString() << std::endl;
    std::cout << inputs.at(3).second.DebugString() << std::endl;
    std::cout << inputs.at(4).second.DebugString() << std::endl;
    std::cout << inputs.at(5).second.DebugString() << std::endl;

    // 加载并初始化模型
    string graph_path = tensorflow::io::JoinPath(root_dir, graph);
    // Status load_graph_status = LoadGraph(graph_path, &session);
    // if (!load_graph_status.ok()) {
    //     LOG(ERROR) << load_graph_status;
    //     return -1;
    // }
    SessionOptions session_options;
    RunOptions run_option;
    SavedModelBundle bundle;
    constexpr char kSavedModelTagServe[] = "serve";
    Status session_status = LoadSavedModel(session_options, run_option, graph_path, {kSavedModelTagServe}, &bundle);
    if (!session_status.ok()) {
        LOG(ERROR) << tensorflow::errors::NotFound("Failed to load compute graph at '", graph_path, "'");
    }
    std::unique_ptr<Session> session(NewSession(session_options));
    session = std::move(bundle.session);
    
    auto sig_map = bundle.GetSignatures();
    auto model_def = sig_map.at("serving_default");

    printf("Model Signature\n");
    for (auto const& p : sig_map) {
        printf("key: %s\n", p.first.c_str());
    }

    printf("Model Input Nodes\n");
    for (auto const& p : model_def.inputs()) {
        printf("key: %s value: %s\n", p.first.c_str(), p.second.name().c_str());
    }

    printf("Model Output Nodes\n");
    for (auto const& p : model_def.outputs()) {
        printf("key: %s value: %s\n", p.first.c_str(), p.second.name().c_str());
    }
    
    // Actually run the tensor through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run(inputs, 
                                    {output_layer}, 
                                    {}, 
                                    &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }
    std::cout << outputs[0].DebugString() << std::endl;
    // This is for automated testing to make sure we get the expected result with the default settings.
    // 计算q-error
    // if (self_test) {
    //     bool expected_matches;
    //     Status check_status = CkeckLablesQError(outputs, 0, &expected_matches);
    // }
    

}
