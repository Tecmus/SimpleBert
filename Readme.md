
### 简化版的bert实现。

现有bert实现结构和抽象方式都是基于google-bert进行修改的，为了模型转化对应方便，google-bert中的每一个scope,在transformers中抽象为一个module。
本实现进一步简化这些对象封装，增加可读性和改写的便捷性。

### 使用方式:
转化基于google-bert结构的预训练模型:

```shell
sh convert_tf_bert.sh <model_path> <output_model_path>
```

示例：

fine-tuning:

mrpc 任务：
```shell
sh run_example.sh
```


    