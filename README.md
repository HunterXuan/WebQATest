# WebQATest
使用方法：
```python
# 生成测试集数据
python gen.py
# 训练模型
insqa_train.py
# 修改 service.py 内的模型载入路径后
python
imoprt service
# 随机打印若干 QA 对
service.print_random_qa()
# 获取最匹配的几个答案
service.get_top_answers('杨利伟是谁？')
```
