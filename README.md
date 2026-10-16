# RAG 知识库问答系统

基于 RAG（检索增强生成）架构的个人知识库问答应用。

## 功能

- 上传文档（Markdown / txt）
- 自动切块、向量化、存入向量库
- 基于知识库内容回答问题
- 显示答案参考的原文来源
- 数据持久化存储

## 技术栈

- 前端：HTML / CSS / JavaScript
- 后端：Python + FastAPI
- 向量数据库：ChromaDB
- 向量模型：sentence-transformers
- 大模型：DeepSeek API

## 架构

用户提问 → 向量化 → ChromaDB 检索相关段落 → 拼接提示词 → DeepSeek 生成答案

## 本地运行

1. 安装依赖
