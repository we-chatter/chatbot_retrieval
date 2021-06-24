# chatbot_retrieval

## 介绍
基于倒排索引和语义匹配的FAQ问答

## 架构

## 数据准备

### 数据

### ES

## 运行项目
    
    1、修改mysql和es的配置
    2、将数据导入到es

## 对话系统集成

原理：
    
    1、基于es的文档检索（更改为自定义分词器）进行粗排序
    2、基于bert的语义匹配精排序。由于候选答案可能较多，使用onnx进行模型加速