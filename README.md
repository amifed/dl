# deep learning for image processing

运行程序

```shell
python3 app.py
```

app.py 会调用系统命令，在后台执行 main.py 文件，main.py 中使用 `print()` 函数输出的内容都会写到 `.result\{执行脚本时间}` 文件夹下的 log.out 文件中，同时会保存训练结束后的模型

经卷积后的矩阵尺寸大小计算公式为

$$
N = (W - F + 2P) / S + 1
$$

> 输入图片大小 $W \times W$ > $\text{Filter}$ 大小 $F \times F$
> 步长 $S$ > $\text{padding}$ 的像素数 $P$

### 参考文章

- [动态调整学习率](!https://blog.csdn.net/qq_42079689/article/details/102806940)
