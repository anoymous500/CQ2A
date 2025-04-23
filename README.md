# LLM-EVal
# Environment
`pip install -r requirements.txt`
# Dataset
下载任意数据集并根据提示进行预处理
# Run
在实际使用过程中，推荐按照如下顺序依次执行脚本：data_process.py、entity(rel)_extraction.py、q_generation.py、filter.py、test.py以完成测试用例的生成流程。在此基础上，可进一步调用 experiment 目录下的实验脚本，以开展本文所设定的各项实验任务
